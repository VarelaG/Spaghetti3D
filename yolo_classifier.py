import cv2
import logging
import os
from ultralytics import YOLO

class YoloClassifier:
    """
    Motor de inferencia profesional basado en Ultralytics YOLO.
    Detecta instancias específicas de fallos y devuelve la confianza máxima.
    """
    def __init__(self, weights_path, manual_threshold=0.45):
        self.model = None
        if not os.path.exists(weights_path):
            logging.error(f"Modelo YOLO no encontrado en: {weights_path}")
            return
            
        try:
            logging.info(f"Cargando motor YOLOv8/11 desde {weights_path}...")
            self.model = YOLO(weights_path)
            self.threshold = manual_threshold
            # Variables de compatibilidad para no romper main.py
            self.is_calibrated = True 
            self.processed_frames = 45
            self.calibration_limit = 45
            self.input_shape = (640, 640) # YOLO re-escala internamente, pero dejamos este default
            logging.info("Motor YOLO cargado y listo. GPU/NPU activa si está disponible.")
        except Exception as e:
            logging.error(f"Error al iniciar Ultralytics: {e}")

    def classify(self, img):
        if self.model is None or img is None:
            return 0.0, 0
            
        try:
            # Ejecutar inferencia (verbose=False oculta los logs de consola por frame)
            results = self.model.predict(img, conf=self.threshold, verbose=False)
            
            # Si detectó alguna caja
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Tomar la confianza mas alta de la lista de cajas encontradas
                score = float(results[0].boxes.conf.max())
                # Dibujar en la imagen opcionalmente (esto lo podemos ver despues)
                return score, 1
                
            return 0.0, 0
            
        except Exception as e:
            logging.error(f"Error en predicción YOLO: {e}")
            return 0.0, 0
