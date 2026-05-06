import cv2
import threading
import queue
import time
import logging

class CameraStream:
    def __init__(self, src=0, queue_size=10):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        
        # Si es un video, no queremos que el hilo salte cuadros
        self.is_file = isinstance(src, str) and not src.isdigit()
        
        if not self.stream.isOpened():
            logging.error(f"Fallo al abrir fuente: {src}")
            raise IOError("No se puede abrir el video")

        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        logging.info(f"Stream inicializado. Fuente: {src} (Es archivo: {self.is_file})")

    def start(self):
        self.thread.start()
        # Dar un pequeño margen para que el primer cuadro cargue
        time.sleep(0.5)
        return self

    def update(self):
        while True:
            if self.stopped:
                break

            if not self.queue.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    if self.is_file:
                        # Fin del video
                        self.stopped = True
                        break
                    else:
                        time.sleep(0.1)
                        continue
                
                self.queue.put(frame)
            else:
                # Si la cola está llena, esperamos un poco
                time.sleep(0.01)

    def read(self):
        try:
            return self.queue.get(timeout=2.0)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()
