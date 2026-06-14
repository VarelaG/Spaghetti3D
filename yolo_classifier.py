import cv2
import logging
import os
from ultralytics import YOLO

class YoloClassifier:
    """
    Motor de inferencia profesional basado en Ultralytics YOLO.
    Detecta instancias específicas de fallos y devuelve la confianza máxima.
    Soporta calibración automática de ruido base para mitigar falsos positivos.
    """
    def __init__(self, weights_path, manual_threshold=None, use_enhancer=False):
        self.model = None
        self.use_enhancer = use_enhancer
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if not os.path.exists(weights_path):
            logging.error(f"Modelo YOLO no encontrado en: {weights_path}")
            return
            
        try:
            logging.info(f"Cargando motor YOLOv8/11 desde {weights_path}...")
            self.model = YOLO(weights_path)
            
            # Variables de calibración dinámica
            self.processed_frames = 0
            self.calibration_limit = 45
            self.max_baseline_score = 0.0
            self.input_shape = (640, 640)
            
            if manual_threshold is not None:
                self.threshold = float(manual_threshold)
                self.is_calibrated = True
                logging.info(f"--> MODO MANUAL YOLO: Umbral fijo establecido en {self.threshold}")
            else:
                self.threshold = 0.45  # Umbral base inicial
                self.is_calibrated = False
                logging.info("--> MODO AUTOMÁTICO YOLO: Iniciando calibración dinámica del ruido base.")
                
            logging.info("Motor YOLO cargado y listo. Aceleración local activa.")
        except Exception as e:
            logging.error(f"Error al iniciar Ultralytics: {e}")

    def classify(self, img):
        if self.model is None or img is None:
            return 0.0, 0
            
        try:
            # --- MEJORA COMERCIAL: Realce de Contraste Dinámico ---
            if self.use_enhancer:
                # Usamos CLAHE en el canal de luminosidad (Espacio LAB)
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = self.clahe.apply(lab[:,:,0])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # En la fase de calibración usamos un conf bajo (0.15) para poder registrar el ruido normal de la cama
            current_conf = 0.15 if not self.is_calibrated else self.threshold
            results = self.model.predict(img, conf=current_conf, verbose=False)
            
            score = 0.0
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Tomar la confianza más alta de las cajas encontradas
                score = float(results[0].boxes.conf.max())
                
            # --- Lógica de Calibración ---
            if not self.is_calibrated:
                self.processed_frames += 1
                if score > self.max_baseline_score:
                    self.max_baseline_score = score
                
                if self.processed_frames >= self.calibration_limit:
                    # Establecer umbral adaptativo = max_baseline + margen (0.08)
                    self.threshold = min(0.70, max(0.28, self.max_baseline_score + 0.08))
                    self.is_calibrated = True
                    logging.info(f"--> Calibración YOLO Completada: Umbral dinámico fijado en {self.threshold:.3f} (Ruido base max: {self.max_baseline_score:.3f})")
                return score, 0
            else:
                label = 1 if score >= self.threshold else 0
                return score, label
                
        except Exception as e:
            logging.error(f"Error en predicción YOLO: {e}")
            return 0.0, 0
