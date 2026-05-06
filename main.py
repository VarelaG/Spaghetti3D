import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np

print("--- [SISTEMA DE MONITOREO UNIVERSAL AUTOCALIBRADO V6] ---")

try:
    from camera_stream import CameraStream
    from detector import NozzleDetector
    from classifier import AnomalyClassifier
    from logic import FailureLogic
except ImportError as e:
    print(f"[ERROR] No se pudo encontrar un módulo local: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AI 3D Printer Monitor - Universal Self-Calibrating")
    parser.add_argument("--camera", type=str, default="0")
    parser.add_argument("--classifier_model", type=str, default="models/anomaly_efficientnet.tflite")
    parser.add_argument("--no_view", action="store_true", help="Desactivar visualizacion")
    args = parser.parse_args()

    try:
        cam_src = int(args.camera) if args.camera.isdigit() else args.camera
        cam = CameraStream(src=cam_src).start()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    detector = NozzleDetector()
    classifier = AnomalyClassifier(args.classifier_model)
    logic = FailureLogic(window_size=15, alert_threshold=0.6)
    
    print("\n>>> MONITOREO AUTOMATICO ACTIVO <<<")

    try:
        while True:
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
                
                # Solo actualizamos la logica de fallos si la calibracion esta terminada
                if classifier.is_calibrated:
                    logic.update(label)
                    avg_fail = logic.get_average()

                    if logic.should_alert():
                        status_text = "!!! ERROR DE IMPRESION DETECTADO !!!"
                        status_color = (0, 0, 255) # Rojo
                    elif avg_fail > 0.2:
                        status_text = "AVISO: ANOMALIA SOSPECHOSA"
                        status_color = (0, 165, 255) # Naranja
                    else:
                        status_text = "ESTADO: IMPRESION SANA"
                        status_color = (0, 255, 0) # Verde
                else:
                    # Durante la calibracion inicial
                    progress = int((classifier.processed_frames / classifier.calibration_limit) * 100)
                    status_text = f"AUTOCALIBRANDO RU_DO BASE... {progress}%"
                    status_color = (255, 140, 0) # Azul/Cian profundo
                    avg_fail = score

            # --- Visualizacion ---
            if not args.no_view:
                # Dibujar cuadro de analisis (Verde)
                x, y, w, h = bed_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "CAMA BAJO MONITOREO", (x, max(20, y - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Barra superior
                cv2.rectangle(frame, (0, 0), (w_f, 50), (20, 20, 20), -1)
                cv2.putText(frame, status_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Mostrar umbral actual de la IA en la barra
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
