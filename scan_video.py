import cv2
import numpy as np
import tensorflow.lite as tflite

model_path = "models/anomaly_efficientnet.tflite"

# Cargar el modelo
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    max_scores = [0.0, 0.0, 0.0, 0.0]
    max_frames = [0, 0, 0, 0]
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        img = cv2.resize(frame, (input_shape[1], input_shape[0]))
        
        # Preprocesamiento normalizado
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        input_data = np.expand_dims(img, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        for idx, class_num in enumerate(range(4, 8)):
            score = float(np.max(output_data[0, class_num, :]))
            if score > max_scores[idx]:
                max_scores[idx] = score
                max_frames[idx] = frame_count
                
    cap.release()
    return max_scores, max_frames

print("Analizando 'videoimp2.mp4' (SANO)...")
scores_sano, frames_sano = analyze_video("videoimp2.mp4")
print("Analizando 'videoimp1.mp4' (CON FALLO)...")
scores_fallo, frames_fallo = analyze_video("videoimp1.mp4")

print("\n=== COMPARATIVA DE DETECCION DE IA ===")
for i in range(4):
    print(f"Clase {i}:")
    print(f"  Sano (Max): {scores_sano[i]:.4f} en frame {frames_sano[i]}")
    print(f"  Fallo (Max): {scores_fallo[i]:.4f} en frame {frames_fallo[i]}")
print("======================================\n")
