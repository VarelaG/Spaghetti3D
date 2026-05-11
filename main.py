import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np

print("--- [SISTEMA DE MONITOREO UNIVERSAL AUTOCALIBRADO V6.1] ---")

try:
    from camera_stream import CameraStream
    from detector import NozzleDetector
    from classifier import AnomalyClassifier
    from yolo_classifier import YoloClassifier # Nuevo Motor Pro
    from logic import FailureLogic
except ImportError as e:
    print(f"[ERROR] No se pudo encontrar un módulo local: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AI 3D Printer Monitor - Self-Calibrating with Auto-Pause")
    parser.add_argument("--camera", type=str, default="0")
    parser.add_argument("--classifier_model", type=str, default="models/anomaly_efficientnet.tflite")
    parser.add_argument("--threshold", type=float, default=None, help="Forzar umbral manual (ej. 0.45)")
    parser.add_argument("--enhance", action="store_true", help="Activar modo de alto contraste (CLAHE)")
    parser.add_argument("--no_view", action="store_true", help="Desactivar visualizacion")
    args = parser.parse_args()

    try:
        cam_src = int(args.camera) if args.camera.isdigit() else args.camera
        cam = CameraStream(src=cam_src).start()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    detector = NozzleDetector()
    
    # CARGADOR DUAL DE INTELIGENCIA ARTIFICIAL
    model_path = args.classifier_model
    
    if model_path.endswith(".pt"):
        # MODO PROFESIONAL (Ultralytics YOLO)
        print(f"[*] Iniciando en Modo Profesional (YOLO PyTorch): {model_path}")
        manual_th = args.threshold if args.threshold is not None else 0.45
        classifier = YoloClassifier(model_path, manual_threshold=manual_th)
    else:
        # MODO LIGERO (TFLite)
        print(f"[*] Iniciando en Modo Ligero (EfficientNet TFLite): {model_path}")
        classifier = AnomalyClassifier(model_path, manual_threshold=args.threshold, use_enhancer=args.enhance)
    
    # Ajustamos la logica temporal para ser mas reactiva
    logic = FailureLogic(window_size=12, alert_threshold=0.45)
    
    print("\n>>> MONITOREO AUTOMATICO ACTIVO <<<")

    try:
        paused_on_error = False
        frozen_frame = None

        while True:
            # Si no está en pausa por error, lee un nuevo cuadro normalmente
            if not paused_on_error:
                frame = cam.read()
                if frame is None: break
                h_f, w_f = frame.shape[:2]
                
                # A. Obtener el cuadro de la cama de impresion
                bed_bbox = detector.detect(frame)
                
                # B. Recortar la zona para la IA
                abn_crop = detector.get_abn(frame, bed_bbox, crop_size=classifier.input_shape)
                
                status_text = "SISTEMA ACTIVO: ANALIZANDO..."
                status_color = (255, 191, 0)
                avg_fail = 0

                if abn_crop is not None:
                    score, label = classifier.classify(abn_crop)
                    
                    if classifier.is_calibrated:
                        logic.update(label)
                        avg_fail = logic.get_average()

                        if logic.should_alert():
                            status_text = "!!! ERROR DE IMPRESION DETECTADO !!!"
                            status_color = (0, 0, 255) # Rojo
                            paused_on_error = True  # Activamos parada de emergencia
                            frozen_frame = frame.copy() # Guardamos copia para congelar
                        elif avg_fail > 0.2:
                            status_text = "AVISO: ANOMALIA SOSPECHOSA"
                            status_color = (0, 165, 255) # Naranja
                        else:
                            status_text = "ESTADO: IMPRESION SANA"
                            status_color = (0, 255, 0) # Verde
                    else:
                        progress = int((classifier.processed_frames / classifier.calibration_limit) * 100)
                        status_text = f"AUTOCALIBRANDO RUIDO BASE... {progress}%"
                        status_color = (255, 140, 0)
                        avg_fail = score

            # --- Visualizacion ---
            if not args.no_view:
                # Si ocurrió una parada de emergencia, mostramos el cuadro congelado con cartelera gigante
                if paused_on_error and frozen_frame is not None:
                    display_frame = frozen_frame.copy()
                    
                    # 1. Crear un filtro translúcido rojo sobre toda la pantalla
                    overlay = display_frame.copy()
                    overlay[:] = (0, 0, 100) # Rojo oscuro
                    cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
                    
                    # 2. Dibujar cartelera gigante de Error en el centro
                    cv2.rectangle(display_frame, (50, h_f//2 - 60), (w_f - 50, h_f//2 + 60), (20, 20, 20), -1)
                    cv2.rectangle(display_frame, (50, h_f//2 - 60), (w_f - 50, h_f//2 + 60), (0, 0, 255), 3)
                    
                    cv2.putText(display_frame, "!!! IMPRESION DETENIDA POR IA !!!", (w_f//2 - 220, h_f//2 - 15), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_frame, "FALLO CRITICO: ESPAGUETI DETECTADO", (w_f//2 - 200, h_f//2 + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(display_frame, "Presione 'q' para salir", (w_f//2 - 100, h_f//2 + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Barra superior roja
                    cv2.rectangle(display_frame, (0, 0), (w_f, 50), (0, 0, 255), -1)
                    cv2.putText(display_frame, "ESTADO: PARADA DE EMERGENCIA ACTIVA", (20, 33), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow("MONITOR AUTOMATICO UNIVERSAL", display_frame)
                else:
                    # Mostrar stream normal activo
                    # Dibujar cuadro de cama
                    x, y, w, h = bed_bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "CAMA BAJO MONITOREO", (x, max(20, y - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Barra superior
                    cv2.rectangle(frame, (0, 0), (w_f, 50), (20, 20, 20), -1)
                    cv2.putText(frame, status_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                    
                    if classifier.is_calibrated:
                        cv2.putText(frame, f"Umbral IA: {classifier.threshold:.2f} | Conf: {avg_fail:.2f}", (w_f - 300, 33), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.putText(frame, f"Ruido: {avg_fail:.2f}", (w_f - 180, 33), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Miniatura
                    if abn_crop is not None:
                        thumb = cv2.resize(abn_crop, (150, 150))
                        frame[60:210, 20:170] = thumb
                        cv2.rectangle(frame, (20, 60), (170, 210), (255, 255, 255), 1)
                        cv2.putText(frame, "ENFOQUE IA", (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow("MONITOR AUTOMATICO UNIVERSAL", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
