import cv2
import os

video_path = "videoimp.mp4"
print(f"Verificando archivo: {video_path}")
print(f"Existe: {os.path.exists(video_path)}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: OpenCV no pudo abrir el video.")
else:
    ret, frame = cap.read()
    print(f"Lectura del primer cuadro: {ret}")
    if ret:
        print(f"Dimensiones: {frame.shape}")
    
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Total de cuadros detectados: {total_frames}")

cap.release()
