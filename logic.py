from collections import deque
import logging

class FailureLogic:
    """
    Módulo de lógica para mitigación de falsos positivos.
    Implementa un 'Rolling Prediction Average' para confirmar fallos sostenidos.
    """
    def __init__(self, window_size=30, alert_threshold=0.8):
        """
        :param window_size: Cantidad de predicciones recientes a considerar (ej. 30 cuadros).
        :param alert_threshold: Porcentaje (0.0 a 1.0) necesario para activar la alerta.
        """
        self.window = deque(maxlen=window_size)
        self.threshold = alert_threshold
        self.alert_active = False
        logging.info(f"Lógica de fallos inicializada: Ventana={window_size}, Umbral={alert_threshold*100}%")

    def update(self, prediction):
        """
        Añade una nueva predicción a la ventana.
        :param prediction: 1 si es anomalía, 0 si es normal.
        """
        self.window.append(prediction)

    def get_average(self):
        """Calcula el promedio de anomalías en la ventana actual."""
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    def should_alert(self):
        """
        Determina si el sistema debe emitir una alerta de fallo.
        Solo se activa si el promedio supera el umbral definido.
        """
        if len(self.window) < self.window.maxlen:
            return False # No alertar hasta tener la ventana llena

        avg = self.get_average()
        
        # Si el promedio supera el umbral
        if avg >= self.threshold:
            if not self.alert_active:
                logging.warning(f"¡ANOMALÍA DETECTADA! Confirmada por promedio sostenido: {avg:.2f}")
                self.alert_active = True
            return True
        
        # Histéresis: Solo bajamos la alerta si el promedio cae por debajo del 50% del umbral
        if self.alert_active and avg < (self.threshold * 0.5):
            self.alert_active = False
            logging.info("La anomalía ha desaparecido o el promedio se ha estabilizado.")
            
        return False

    def reset(self):
        """Limpia la ventana de predicciones (útil tras una intervención)."""
        self.window.clear()
        self.alert_active = False
        logging.info("Lógica de predicciones reiniciada.")

if __name__ == "__main__":
    logging.info("Módulo logic.py cargado.")
