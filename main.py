import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np

print("--- [SISTEMA DE MONITOREO AUTOMATICO DE CAMA V5] ---")

try:
    from camera_stream import CameraStream
    from detector import NozzleDetector
    from classifier import AnomalyClassifier
    from logic import FailureLogic
except ImportError as e:
    print(f"[ERROR] No se pudo encontrar un módulo local: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AI 3D Printer Monitor - Bed Tracker")
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
                logic.update(label)
                avg_fail = logic.get_average()

                if logic.should_alert():
                    status_text = "!!! ERROR DE IMPRESION DETECTADO !!!"
                    status_color = (0, 0, 255)
                elif avg_fail > 0.2:
                    status_text = "AVISO: ANOMALIA SOSPECHOSA"
                    status_color = (0, 165, 255)
                else:
                    status_text = "ESTADO: IMPRESION SANA"
                    status_color = (0, 255, 0)

            # --- Visualizacion ---
            if not args.no_view:
                # Dibujar cuadro de analisis (Verde)
                x, y, w, h = bed_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "CAMA BAJO MONITOREO", (x, max(20, y - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Barra superior (Sin caracteres especiales para evitar ??)
                cv2.rectangle(frame, (0, 0), (w_f, 50), (20, 20, 20), -1)
                cv2.putText(frame, status_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, f"IA Conf: {avg_fail:.2f}", (w_f - 180, 33), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Miniatura
                if abn_crop is not None:
                    thumb = cv2.resize(abn_crop, (150, 150))
                    frame[60:210, 20:170] = thumb
                    cv2.rectangle(frame, (20, 60), (170, 210), (255, 255, 255), 1)
                    cv2.putText(frame, "ENFOQUE IA", (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("MONITOR AUTOMATICO", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
