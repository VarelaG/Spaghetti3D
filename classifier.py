import numpy as np
import cv2
import logging
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        logging.warning("No se encontró tflite_runtime.")

class AnomalyClassifier:
    def __init__(self, model_path, manual_threshold=None, use_enhancer=False):
        self.interpreter = None
        self.use_enhancer = use_enhancer
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        if not os.path.exists(model_path): 
            logging.error(f"Modelo no encontrado en: {model_path}")
            return

        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape'][1:3]
            
            # Variables de Umbral
            self.processed_frames = 0
            self.calibration_limit = 45
            self.max_baseline_score = 0.0
            self.is_calibrated = False
            
            if manual_threshold is not None:
                self.threshold = float(manual_threshold)
                self.is_calibrated = True
                logging.info(f"--> MODO MANUAL: Umbral fijado por usuario en {self.threshold}")
            else:
                self.threshold = 0.35 # Base
                logging.info("--> MODO AUTOMATICO: Iniciando calibracion dinamica.")
                
            logging.info(f"IA lista con entrada: {self.input_shape}. Filtro realce: {use_enhancer}")
        except Exception as e:
            logging.error(f"Error carga TFLite: {e}")

    def preprocess(self, abn_frame):
        if abn_frame is None: return None
        img = cv2.resize(abn_frame, (self.input_shape[1], self.input_shape[0]))
        
        # --- MEJORA COMERCIAL: Realce de Contraste Dinámico ---
        if self.use_enhancer:
            # Usamos CLAHE en el canal de luminosidad (Espacio LAB)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = self.clahe.apply(lab[:,:,0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def classify(self, abn_frame):
        if self.interpreter is None or abn_frame is None:
            return 0.0, 0

        try:
            input_data = self.preprocess(abn_frame)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Extraccion robusta del score
            if len(output_data.shape) == 3 and output_data.shape[1] == 8:
                class_scores = output_data[0, 4:, :] 
                score = float(np.max(class_scores))
            else:
                flat_output = output_data.flatten()
                score = float(flat_output[0])
                
            # --- Lógica de Decisión ---
            if not self.is_calibrated:
                self.processed_frames += 1
                if score > self.max_baseline_score:
                    self.max_baseline_score = score
                
                if self.processed_frames >= self.calibration_limit:
                    # Clamp para no bloquear detecciones reales si el inicio tiene ruido
                    self.threshold = min(0.75, max(0.25, self.max_baseline_score + 0.12))
                    self.is_calibrated = True
                    logging.info(f"--> Calibracion Completada: Umbral fijado en {self.threshold:.3f}")
                return score, 0
            else:
                label = 1 if score >= self.threshold else 0 
                return score, label
                
        except Exception as e:
            logging.error(f"Error clasificación: {e}")
            return 0.0, 0
