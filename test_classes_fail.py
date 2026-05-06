import cv2
import numpy as np
import tensorflow.lite as tflite

model_path = "models/anomaly_efficientnet.tflite"
video_path = "videoimp1.mp4"

# Cargar el modelo
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

# Leer un frame de la mitad del video de fallos (donde ya hay hilos)
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2) # Ir al medio del video
ret, frame = cap.read()
cap.release()

if ret:
    img = cv2.resize(frame, (input_shape[1], input_shape[0]))
    
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    input_data = np.expand_dims(img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print("\n--- ANALISIS DE CLASES EN PIEZA CON FALLO ---")
    for idx, class_num in enumerate(range(4, 8)):
        class_max = np.max(output_data[0, class_num, :])
        print(f"Clase {idx} - Máxima Probabilidad Detectada: {class_max:.4f}")
    print("---------------------------------------------\n")
