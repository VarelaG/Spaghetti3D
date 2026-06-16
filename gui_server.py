import os
import sys
import time
import logging
import threading
import webbrowser
from flask import Flask, jsonify, request, Response, send_from_directory

# Configurar logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Intentar importar módulos locales
try:
    from camera_stream import CameraStream
    from detector import NozzleDetector
    from yolo_classifier import YoloClassifier
    from classifier import AnomalyClassifier
    from logic import FailureLogic
    from printer_control import PrinterControl
    import cv2
except ImportError as e:
    logging.error(f"No se pudo cargar un módulo necesario: {e}")
    print("\n[ERROR] Por favor, asegúrate de estar en el directorio raíz del proyecto y tener instaladas las dependencias.")
    print("Corre: pip install -r requirements.txt\n")
    sys.exit(1)

app = Flask(__name__)

# Estado compartido para monitoreo concurrente
monitoring_active = False
monitor_thread = None
latest_frame = None
recalibrate_flag = False
frame_lock = threading.Lock()
printer_control = None
printer_config = {
    "host": None,
    "api_key": None,
    "printer_type": "octoprint",
    "auto_pause": False,
    "enabled": False
}

# Métricas iniciales
metrics = {
    "active": False,
    "score": 0.0,
    "avg_fail": 0.0,
    "threshold": 0.45,
    "fps": 0,
    "state": "DESCONECTADO",
    "printer_connected": False,
    "printer_action": None,
    "printer_error": None
}

def monitor_loop(src, use_calibrate=True, manual_threshold=None, model_path="models/spaghetti_pro.pt", use_enhance=False):
    """
    Bucle principal de procesamiento de imágenes y predicción IA en segundo plano.
    """
    global monitoring_active, latest_frame, metrics, recalibrate_flag, printer_control, printer_config
    
    logging.info(f"[BACKEND] Iniciando monitoreo de fuente: {src} (Modelo: {model_path}, Calibrar: {use_calibrate}, Realce CLAHE: {use_enhance}, Umbral: {manual_threshold})")
    
    # Reiniciar métricas
    metrics["active"] = True
    metrics["score"] = 0.0
    metrics["avg_fail"] = 0.0
    metrics["fps"] = 0
    metrics["state"] = "CALIBRANDO" if use_calibrate else "MONITOREANDO"
    metrics["printer_connected"] = printer_control is not None
    metrics["printer_action"] = None
    metrics["printer_error"] = None
    
    try:
        # Si la entrada es un número (ej. "0"), abrirla como índice de cámara web
        cam_src = int(src) if src.isdigit() else src
        cam = CameraStream(src=cam_src).start()
    except Exception as e:
        logging.error(f"[BACKEND] Error al abrir cámara/video '{src}': {e}")
        metrics["active"] = False
        metrics["state"] = f"ERROR: {e}"
        monitoring_active = False
        return

    detector = NozzleDetector()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    if not os.path.exists(model_path):
        logging.error(f"[BACKEND] Modelo no encontrado en: {model_path}")
        metrics["active"] = False
        metrics["state"] = "ERROR: Modelo no encontrado"
        monitoring_active = False
        cam.stop()
        return

    # Iniciar clasificador correspondiente (YOLO vs TFLite)
    if model_path.endswith(".pt"):
        if use_calibrate:
            classifier = YoloClassifier(model_path, use_enhancer=use_enhance)
        else:
            classifier = YoloClassifier(model_path, manual_threshold=manual_threshold, use_enhancer=use_enhance)
    else:
        # TFLite
        classifier = AnomalyClassifier(model_path, manual_threshold=manual_threshold, use_enhancer=use_enhance)
        classifier.is_calibrated = not use_calibrate
        if not use_calibrate and manual_threshold is not None:
            classifier.threshold = manual_threshold
        elif use_calibrate:
            classifier.calibration_limit = 45
    
    # Lógica de fallos: ventana temporal reactiva de 12 cuadros, alerta al superar el 45%
    logic = FailureLogic(window_size=12, alert_threshold=0.45)
    
    metrics["threshold"] = classifier.threshold
    
    fps_start_time = time.time()
    fps_counter = 0
    paused_on_error = False
    frozen_frame = None
    
    while monitoring_active:
        # Gestionar solicitud de recalibración/reset manual
        if recalibrate_flag:
            logging.info("[BACKEND] Reset de alerta y lógica solicitado por el usuario.")
            logic.reset()
            # Resetear calibración del clasificador sólo si iniciamos con ella
            if use_calibrate:
                classifier.processed_frames = 0
                classifier.max_baseline_score = 0.0
                classifier.is_calibrated = False
                metrics["state"] = "CALIBRANDO"
            else:
                classifier.is_calibrated = True
                metrics["state"] = "MONITOREANDO"
            paused_on_error = False
            frozen_frame = None
            recalibrate_flag = False
        
        if not paused_on_error:
            frame = cam.read()
            if frame is None:
                logging.info("[BACKEND] Fin del video / flujo finalizado.")
                break
                
            h_f, w_f = frame.shape[:2]
            
            # A. Detectar Cama de Impresión
            bed_bbox = detector.detect(frame)
            
            # B. Recortar ROI para la IA
            abn_crop = detector.get_abn(frame, bed_bbox, crop_size=classifier.input_shape)
            
            score = 0.0
            label = 0
            avg_fail = 0.0
            status_text = "ESTADO: IMPRESION SANA"
            status_color = (0, 255, 0)  # Verde
            state_label = "SANO"
            
            if abn_crop is not None:
                score, label = classifier.classify(abn_crop)
                
                if classifier.is_calibrated:
                    logic.update(label)
                    avg_fail = logic.get_average()
                    
                    if logic.should_alert():
                        status_text = "!!! ERROR DE IMPRESION DETECTADO !!!"
                        status_color = (0, 0, 255)  # Rojo
                        state_label = "FALLO CRÍTICO"
                        paused_on_error = True
                        frozen_frame = frame.copy()
                        # Si la impresora está configurada y la pausa automática está habilitada, enviamos la orden
                        if printer_control is not None and printer_config.get("auto_pause", False):
                            success = printer_control.pause_print()
                            metrics["printer_action"] = "pause_sent" if success else "pause_failed"
                            metrics["printer_error"] = None if success else "Error pausando impresora"
                        else:
                            metrics["printer_action"] = None
                            metrics["printer_error"] = None
                    elif avg_fail > 0.2:
                        status_text = "AVISO: ANOMALIA SOSPECHOSA"
                        status_color = (0, 165, 255)  # Naranja
                        state_label = "ADVERTENCIA"
                    else:
                        status_text = "ESTADO: IMPRESION SANA"
                        status_color = (0, 255, 0)  # Verde
                        state_label = "SANO"
                else:
                    # Fase de autocalibración de ruido
                    progress = int((classifier.processed_frames / classifier.calibration_limit) * 100)
                    status_text = f"AUTOCALIBRANDO RUIDO BASE... {progress}%"
                    status_color = (255, 140, 0)  # Naranja/Amarillo
                    state_label = "CALIBRANDO"
                    avg_fail = score
            
            # Actualizar umbral en las métricas en tiempo real
            metrics["threshold"] = classifier.threshold
            
            # Construir visualización sobre una copia
            display_frame = frame.copy()
            x, y, w, h = bed_bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, "CAMA BAJO MONITOREO", (x, max(20, y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Barra superior informativa
            cv2.rectangle(display_frame, (0, 0), (w_f, 50), (20, 20, 20), -1)
            cv2.putText(display_frame, status_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            if classifier.is_calibrated:
                cv2.putText(display_frame, f"Conf: {score:.2f} | Prom: {avg_fail:.2f}", (w_f - 260, 33), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(display_frame, f"Ruido: {score:.2f}", (w_f - 180, 33), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Miniatura de enfoque IA
            if abn_crop is not None:
                thumb = cv2.resize(abn_crop, (150, 150))
                display_frame[60:210, 20:170] = thumb
                cv2.rectangle(display_frame, (20, 60), (170, 210), (255, 255, 255), 1)
                cv2.putText(display_frame, "ENFOQUE IA", (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        else:
            # Estado congelado por Parada de Emergencia
            display_frame = frozen_frame.copy()
            h_f, w_f = display_frame.shape[:2]
            
            # Overlay translúcido rojo
            overlay = display_frame.copy()
            overlay[:] = (0, 0, 100)
            cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
            
            # Cartelera central
            cv2.rectangle(display_frame, (50, h_f//2 - 60), (w_f - 50, h_f//2 + 60), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (50, h_f//2 - 60), (w_f - 50, h_f//2 + 60), (0, 0, 255), 3)
            
            cv2.putText(display_frame, "!!! IMPRESION DETENIDA POR IA !!!", (w_f//2 - 220, h_f//2 - 15), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "FALLO CRITICO: ESPAGUETI DETECTADO", (w_f//2 - 200, h_f//2 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Pulse 'Reanudar Monitoreo' en el panel", (w_f//2 - 160, h_f//2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Barra superior
            cv2.rectangle(display_frame, (0, 0), (w_f, 50), (0, 0, 255), -1)
            cv2.putText(display_frame, "ESTADO: PARADA DE EMERGENCIA ACTIVA", (20, 33), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            score = 1.0
            avg_fail = 1.0
            state_label = "FALLO CRÍTICO"
            
        # Calcular FPS reales
        fps_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            current_fps = int(fps_counter / elapsed)
            fps_counter = 0
            fps_start_time = time.time()
        else:
            current_fps = metrics.get("fps", 0)
            
        # Actualizar métricas del estado
        metrics["score"] = score
        metrics["avg_fail"] = avg_fail
        metrics["fps"] = current_fps
        metrics["state"] = state_label
        
        # Codificar cuadro a JPG
        _, jpeg = cv2.imencode('.jpg', display_frame)
        jpeg_bytes = jpeg.tobytes()
        
        with frame_lock:
            latest_frame = jpeg_bytes
            
        # Control de FPS para reproducción de videos locales
        if cam.is_file:
            time.sleep(0.033) # Limitar a ~30 FPS para evitar sobrecarga y reproducir natural

    # Apagar cámara al salir
    logging.info("[BACKEND] Cerrando cámara/flujo y liberando recursos.")
    cam.stop()
    monitoring_active = False
    
    # Resetear estado
    with frame_lock:
        latest_frame = None
    metrics["active"] = False
    metrics["state"] = "DESCONECTADO"


# --- RUTAS DE FLASK ---

@app.route('/')
def index():
    """Sirve la página web de la interfaz local."""
    return send_from_directory('web', 'gui.html')

@app.route('/styles.css')
def styles():
    """Sirve los estilos de la interfaz."""
    return send_from_directory('web', 'styles.css')

@app.route('/api/videos', methods=['GET'])
def list_videos():
    """Busca y devuelve la lista de videos disponibles en el directorio del proyecto."""
    extensions = ('.mp4', '.mov', '.avi', '.mkv')
    files = os.listdir('.')
    videos = [f for f in files if f.lower().endswith(extensions) and os.path.isfile(f)]
    videos.sort()
    return jsonify(videos)

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Inicia el bucle de monitoreo con la fuente seleccionada y configuraciones de calibración."""
    global monitoring_active, monitor_thread, printer_control, printer_config
    
    if monitoring_active:
        return jsonify({"status": "error", "error": "El monitoreo ya está en ejecución"}), 400
        
    data = request.get_json() or {}
    source_val = data.get("source", "0")
    use_calibrate = data.get("calibrate", True)
    manual_th = data.get("threshold", None)
    model_val = data.get("model", "models/spaghetti_pro.pt")
    use_enhance = data.get("enhance", False)
    printer_host = data.get("printer_host", "").strip()
    printer_api_key = data.get("printer_api_key", "").strip()
    printer_type = data.get("printer_type", "octoprint")
    auto_pause = bool(data.get("auto_pause", False))

    if auto_pause and (not printer_host or not printer_api_key):
        return jsonify({"status": "error", "error": "Si activás la 'pausa automática' debés completar el Host del server de OctoPrint y su respectiva API Key."}), 400

    printer_config.update({
        "host": printer_host or None,
        "api_key": printer_api_key or None,
        "printer_type": printer_type,
        "auto_pause": auto_pause,
        "enabled": bool(printer_host and printer_api_key and auto_pause),
        "configured": bool(printer_host and printer_api_key)
    })

    if printer_config["configured"]:
        logging.info("[BACKEND] Verificando conexión con OctoPrint/Moonraker antes de iniciar monitoreo.")
        printer_control = PrinterControl(printer_host, printer_api_key, printer_type=printer_type)
        connected = printer_control.test_connection()
        metrics["printer_connected"] = connected
        if not connected:
            printer_control = None
            if auto_pause:
                monitoring_active = False
                return jsonify({"status": "error", "error": "No se pudo conectar con OctoPrint/Moonraker. Revisa la URL/API Key."}), 400
    else:
        printer_control = None
        metrics["printer_connected"] = False

    monitoring_active = True
    monitor_thread = threading.Thread(
        target=monitor_loop, 
        args=(source_val, use_calibrate, manual_th, model_val, use_enhance), 
        daemon=True
    )
    monitor_thread.start()

    printer_started = False
    if printer_control is not None and metrics["printer_connected"]:
        printer_started = printer_control.start_print()
        metrics["printer_action"] = "start_sent" if printer_started else "start_failed"
        metrics["printer_error"] = None if printer_started else "Error iniciando impresión"

    return jsonify({
        "status": "success",
        "message": f"Monitoreo iniciado con fuente {source_val}",
        "printer_connected": metrics["printer_connected"],
        "printer_started": printer_started
    })

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Detiene el bucle de monitoreo activo."""
    global monitoring_active, monitor_thread
    
    if not monitoring_active:
        return jsonify({"status": "error", "error": "El monitoreo no está activo"}), 400
        
    monitoring_active = False
    if monitor_thread:
        monitor_thread.join(timeout=3.0)
        
    return jsonify({"status": "success", "message": "Monitoreo detenido correctamente"})

@app.route('/api/recalibrate', methods=['POST'])
def reset_alert():
    """Solicita al bucle activo reiniciar la lógica de fallos y desactivar alarmas."""
    global recalibrate_flag
    if monitoring_active:
        recalibrate_flag = True
        return jsonify({"status": "success", "message": "Reseteo enviado con éxito"})
    return jsonify({"status": "error", "error": "El monitoreo no está en ejecución"}), 400

@app.route('/api/printer/resume', methods=['POST'])
def resume_printer():
    """Reanuda la impresión pausada en OctoPrint/Moonraker."""
    if printer_control is None:
        return jsonify({"status": "error", "error": "Impresora no configurada"}), 400

    success = printer_control.resume_print()
    if success:
        return jsonify({"status": "success", "message": "Impresión reanudada"})
    return jsonify({"status": "error", "error": "No se pudo reanudar la impresión"}), 500

@app.route('/api/printer/cancel', methods=['POST'])
def cancel_print_job():
    """Cancela el trabajo de impresión en OctoPrint/Moonraker."""
    if printer_control is None:
        return jsonify({"status": "error", "error": "Impresora no configurada"}), 400

    success = printer_control.cancel_print()
    if success:
        return jsonify({"status": "success", "message": "Impresión cancelada"})
    return jsonify({"status": "error", "error": "No se pudo cancelar la impresión"}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Devuelve las métricas en tiempo real en formato JSON."""
    return jsonify(metrics)

@app.route('/video_feed')
def video_feed():
    """Transmite el stream MJPEG generado por el bucle de IA."""
    def generate():
        global latest_frame, monitoring_active
        while monitoring_active:
            with frame_lock:
                frame_data = latest_frame
            if frame_data is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            # Pequeña pausa para no saturar la red local con envíos repetidos
            time.sleep(0.03)
            
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def open_browser():
    """Espera un segundo para que el servidor Flask inicie y abre el navegador."""
    time.sleep(1.0)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    # Lanzar el hilo que abre el navegador
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Ejecutar Flask
    print("\n" + "="*60)
    print("  INICIANDO SERVIDOR LOCAL DE SPAGHETTI3D")
    print("  Accede a la interfaz en: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
