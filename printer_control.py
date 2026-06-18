import requests
import logging
import urllib3

# Suppress InsecureRequestWarning when verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
            logging.info(f"[PRINTER] {method} {url}")
            if method == 'POST':
                response = requests.post(url, headers=headers, json=json_data, timeout=self.timeout, verify=False)
            else:
                response = requests.get(url, headers=headers, timeout=self.timeout, verify=False)

            if response.status_code >= 200 and response.status_code < 300:
                return True
            else:
                logging.error(f"[PRINTER] Error {response.status_code}: {response.text}")
                return False
        except requests.exceptions.SSLError as e:
            logging.error(f"[PRINTER] SSL Error (certificado no válido): {e}")
            return False
        except requests.exceptions.ConnectionError as e:
            logging.error(f"[PRINTER] Error de conexión (no se pudo alcanzar el host): {e}")
            return False
        except requests.exceptions.Timeout as e:
            logging.error(f"[PRINTER] Timeout: el servidor no respondió en {self.timeout}s: {e}")
            return False
        except requests.exceptions.RequestException as e:
            logging.error(f"[PRINTER] Error inesperado: {type(e).__name__}: {e}")
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

    def start_print(self):
        """Inicia o reanuda la impresión en el sistema de la impresora."""
        logging.info("Solicitando INICIO de impresión...")
        if self.type == 'octoprint':
            return self._send_request('/api/job', json_data={"command": "start"})
        elif self.type == 'moonraker':
            return self._send_request('/printer/print/start')
        return False

    def resume_print(self):
        """Reanuda la impresión pausada."""
        logging.info("Solicitando REANUDAR impresión...")
        if self.type == 'octoprint':
            return self._send_request('/api/job', json_data={"command": "pause", "action": "resume"})
        elif self.type == 'moonraker':
            return self._send_request('/printer/print/resume')
        return False

    def test_connection(self):
        """Verifica que el host de la impresora responda y la clave API sea válida."""
        logging.info("Verificando conexión con la impresora...")
        if self.type == 'octoprint':
            return self._send_request('/api/version', method='GET')
        elif self.type == 'moonraker':
            return self._send_request('/printer/info', method='GET')
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
