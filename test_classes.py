import cv2
import numpy as np
import tensorflow.lite as tflite

model_path = "models/anomaly_efficientnet.tflite"
video_path = "videoimp2.mp4"

# 1. Cargar el modelo
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

# 2. Leer un frame del video del objeto sano
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    # Preprocesar igual que en el clasificador
    img = cv2.resize(frame, (input_shape[1], input_shape[0]))
    
    # Aplicar ecualización en canal Y para ver si el contraste influye
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    input_data = np.expand_dims(img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print("\n--- ANALISIS DE CLASES EN PIEZA SANA ---")
    print(f"Forma de la salida: {output_data.shape}")
    
    # Imprimir los valores máximos para cada una de las 4 clases
    for idx, class_num in enumerate(range(4, 8)):
        class_max = np.max(output_data[0, class_num, :])
        print(f"Clase {idx} - Máxima Probabilidad Detectada: {class_max:.4f}")
    print("----------------------------------------\n")
else:
    print("No se pudo cargar el video para la prueba.")
