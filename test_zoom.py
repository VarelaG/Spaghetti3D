import cv2
import numpy as np
import tensorflow.lite as tflite

model_path = "models/anomaly_efficientnet.tflite"
video_path = "videoimp.mp4"

# Cargar el modelo
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

cap = cv2.VideoCapture(video_path)
max_score = 0.0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    img = cv2.resize(frame, (input_shape[1], input_shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    input_data = np.expand_dims(img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Tomar la probabilidad de anomalía (Clase 0)
    score = float(np.max(output_data[0, 4:, :]))
    if score > max_score:
        max_score = score

cap.release()
print(f"\n--- ANALISIS DE 'videoimp.mp4' (SANO CON ZOOM) ---")
print(f"Probabilidad Máxima Detectada: {max_score:.4f}")
print("---------------------------------------------------\n")
