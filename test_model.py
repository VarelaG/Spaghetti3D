import tensorflow.lite as tflite
import numpy as np

model_path = "models/anomaly_efficientnet.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("--- DIAGNÓSTICO DEL MODELO ---")
print(f"Input Shape: {input_details[0]['shape']}")
print(f"Output Shape: {output_details[0]['shape']}")
print(f"Output Type: {output_details[0]['dtype']}")
print("------------------------------")
