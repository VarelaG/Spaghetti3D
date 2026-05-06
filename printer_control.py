import requests
import logging

class PrinterControl:
    """
    Módulo para controlar la impresora 3D mediante APIs de red.
    Soporta OctoPrint y Moonraker (Klipper).
    """
    def __init__(self, host, api_key=None, printer_type='octoprint'):
        """
        :param host: URL base de la impresora (ej. http://192.168.1.100)
        :param api_key: Clave API (requerida para OctoPrint, opcional para Moonraker)
        :param printer_type: 'octoprint' o 'moonraker'
        """
        self.host = host.rstrip('/')
        self.api_key = api_key
        self.type = printer_type.lower()
        self.timeout = 5 

        logging.info(f"Controlador de impresora configurado para {self.type} en {self.host}")

    def _send_request(self, endpoint, method='POST', json_data=None):
        """Manejo genérico de peticiones HTTP con excepciones."""
        headers = {}
        if self.api_key:
            headers['X-Api-Key'] = self.api_key

        try:
            url = f"{self.host}{endpoint}"
            if method == 'POST':
                response = requests.post(url, headers=headers, json=json_data, timeout=self.timeout)
            else:
                response = requests.get(url, headers=headers, timeout=self.timeout)

            # Para Moonraker, a veces se usa JSON-RPC sobre HTTP POST a /printer/print/pause
            # pero aquí usamos los endpoints REST directos que Moonraker también soporta.
            
            if response.status_code >= 200 and response.status_code < 300:
                return True
            else:
                logging.error(f"La impresora respondió con error {response.status_code}: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logging.error(f"Error de conexión con la impresora: {e}")
            return False

    def pause_print(self):
        """Pausa la impresión para inspección humana."""
        logging.info("Solicitando PAUSA de impresión...")
        if self.type == 'octoprint':
            return self._send_request('/api/job', json_data={"command": "pause", "action": "pause"})
        elif self.type == 'moonraker':
            return self._send_request('/printer/print/pause')
        return False

    def cancel_print(self):
        """Cancela la impresión para evitar desperdicio de filamento."""
        logging.warning("Solicitando CANCELACIÓN de impresión...")
        if self.type == 'octoprint':
            return self._send_request('/api/job', json_data={"command": "cancel"})
        elif self.type == 'moonraker':
            return self._send_request('/printer/print/cancel')
        return False

    def emergency_stop(self):
        """Parada de emergencia (M112)."""
        logging.critical("¡ENVIANDO PARADA DE EMERGENCIA!")
        if self.type == 'octoprint':
            return self._send_request('/api/printer/command', json_data={"command": "M112"})
        elif self.type == 'moonraker':
            return self._send_request('/printer/emergency_stop')
        return False

if __name__ == "__main__":
    logging.info("Módulo printer_control.py cargado.")
