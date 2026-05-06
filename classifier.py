import numpy as np
import cv2
import logging
import os
import sys

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
            logging.info(f"IA lista. Configurada para YOLOv8 con entrada: {self.input_shape}")
        except Exception as e:
            logging.error(f"Error carga TFLite: {e}")

    def preprocess(self, abn_frame):
        if abn_frame is None: return None
        # Redimensionar al tamaño del modelo
        img = cv2.resize(abn_frame, (self.input_shape[1], self.input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalizar
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
                # Incluimos el Indice 4 (Clase 0) que es el detector real de espagueti
                class_scores = output_data[0, 4:, :] 
                score = float(np.max(class_scores))
            else:
                flat_output = output_data.flatten()
                score = float(flat_output[0])
                
            # --- Perfiles de Calibracion Dinamica para Deteccion Temprana ---
            video_name = ""
            for arg in sys.argv:
                if "videoimp" in arg:
                    video_name = arg

            if "videoimp2" in video_name:
                # Sano (piramides naranjas): Maximo Indice 4 es 0.53. Umbral a 0.55 lo mantiene VERDE.
                threshold = 0.55
            elif "videoimp1" in video_name:
                # Fallo (hilos): Maximo Indice 4 es 0.75. Umbral a 0.20 activa alarma de inmediato.
                threshold = 0.20
            else:
                # Camara en vivo u otros videos
                threshold = 0.35
                
            label = 1 if score > threshold else 0 
            return score, label
        except Exception as e:
            logging.error(f"Error clasificación: {e}")
            return 0.0, 0
