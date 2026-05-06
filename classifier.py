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
    def __init__(self, model_path):
        self.interpreter = None
        if not os.path.exists(model_path): return

        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape'][1:3]
            
            # --- Variables de Autocalibracion Dinamica ---
            self.processed_frames = 0
            self.calibration_limit = 45  # Calibra durante los primeros ~3-4 segundos de video
            self.max_baseline_score = 0.0
            self.threshold = 0.35  # Valor inicial por defecto
            self.is_calibrated = False
            
            logging.info(f"IA lista con entrada: {self.input_shape}. Algoritmo de Autocalibracion Dinamica Activo.")
        except Exception as e:
            logging.error(f"Error carga TFLite: {e}")

    def preprocess(self, abn_frame):
        if abn_frame is None: return None
        img = cv2.resize(abn_frame, (self.input_shape[1], self.input_shape[0]))
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
            
            # Estructura YOLOv8 Detect [1, 8, 8400]
            if len(output_data.shape) == 3 and output_data.shape[1] == 8:
                class_scores = output_data[0, 4:, :] 
                score = float(np.max(class_scores))
            else:
                flat_output = output_data.flatten()
                score = float(flat_output[0])
                
            # --- FASE DE AUTOCALIBRACION DINAMICA (Universal) ---
            if not self.is_calibrated:
                self.processed_frames += 1
                if score > self.max_baseline_score:
                    self.max_baseline_score = score
                
                label = 0  # No disparamos errores durante la fase de calibracion inicial
                
                if self.processed_frames >= self.calibration_limit:
                    # Fijamos el umbral dinamico agregando un margen del 15% sobre el ruido base observado
                    self.threshold = min(0.88, max(0.20, self.max_baseline_score + 0.15))
                    self.is_calibrated = True
                    logging.info(f"--> Autocalibracion Completada. Ruido Base: {self.max_baseline_score:.4f} | Umbral Fijado: {self.threshold:.4f}")
            else:
                # Fase de Monitoreo Activo usando el umbral auto-aprendido
                label = 1 if score > self.threshold else 0 
                
            return score, label
        except Exception as e:
            logging.error(f"Error clasificación: {e}")
            return 0.0, 0
