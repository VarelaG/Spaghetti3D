import cv2
import numpy as np
import logging

class NozzleDetector:
    """
    Rastreador de la Cama de Impresión (ROI Estática).
    Define la zona de construcción principal de la impresora 3D para enfocar la IA
    en la pieza, sin importar el color del filamento ni la iluminación.
    """
    def __init__(self, model_path=None, confidence_threshold=0.5):
        logging.info("Rastreador de Cama de Impresión (ROI Estática) activado.")

    def detect(self, frame):
        """
        Devuelve un cuadro delimitador que cubre la zona de impresión central (el 80% de la cama).
        Esto asegura capturar la pieza completa sin importar su color.
        """
        h, w = frame.shape[:2]
        
        # Definir una ROI central muy robusta
        bx = int(w * 0.1)
        by = int(h * 0.15)
        bw = int(w * 0.8)
        bh = int(h * 0.75)
        
        return (bx, by, bw, bh)

    def get_abn(self, frame, bbox, crop_size=(640, 640)):
        """
        Recorta y redimensiona la zona de impresión para la IA.
        """
        if bbox is None:
            return None
            
        x, y, w, h = bbox
        crop = frame[y:y+h, x:x+w]
        
        if crop.size == 0:
            return None
            
        return cv2.resize(crop, crop_size)
