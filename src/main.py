# src/main.py
import sys
import cv2
import base64, requests, json
import resources_rc
# Referencia explícita para evitar warnings de linter; resources se registran al importarlos
_ = resources_rc
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice
from pathlib import Path
from PySide6.QtCore import Qt
import time

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QDialog, QPushButton
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 1=oculta INFO, 2=oculta también WARNING, 3=oculta ERROR

# silenciar logs de absl (los que imprime MediaPipe)
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# (opcional) por si tu runtime usa el logger estándar
import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import mediapipe as mp  # <-- después de silenciar



def load_ui(path):
    loader = QUiLoader()
    ui_file = QFile(str(path))
    ui_file.open(QIODevice.ReadOnly)
    window = loader.load(ui_file)
    ui_file.close()
    return window


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # variables
        self.total = 0
        self.cant_ok = 0

        base_dir = Path(__file__).resolve().parents[1]
        self.base_dir = base_dir
        self.ui = load_ui(base_dir / "ui" / "main_window.ui")
        self.pop_up = base_dir / "ui" /"popup_window.ui"
        # Buscar el QLabel del .ui
        self.label: QLabel = self.ui.findChild(QLabel, "lblCamera")

        # Qlabel que muestra el resultado de cada captura
        self.lbl_result: QLabel = self.ui.findChild(QLabel, "lblResult")

        # Imagen correcta
        self.correcto_icon = QPixmap(":/icons/correcto.png")
        self.correcto_icon = self.correcto_icon.scaled(self.lbl_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        #imagen error
        self.error_icon = QPixmap(":/icons/error.png")
        self.error_icon = self.error_icon.scaled(self.lbl_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Inicializar la cámara (0 = webcam principal)
        self.cap = cv2.VideoCapture(0)

        # --- MediaPipe Holistic (una sola instancia, estable) ---
        self.mp_holistic = mp.solutions.holistic
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,  # tracking entre frames
            model_complexity=1,  # 0=rápido, 1=mejor; probá 0 si tu CPU sufre
            smooth_landmarks=True,  # suaviza y reduce jitter
            enable_segmentation=False,
            refine_face_landmarks=False,  # sin cara
            min_detection_confidence=0.6,  # subí si hay ruido
            min_tracking_confidence=0.6
        )

        # Procesar 1 de cada N frames (si tu CPU lo necesita)
        self.frame_count = 0
        self.process_every = 2

        # Estado para evitar parpadeo: último resultado válido y “edad”
        self.last_valid_res = None
        self.last_valid_age = 0
        self.hold_max_frames = 6  # cuánto tiempo "sostener" el overlay si se pierde detección



        # para no saturar CPU: procesar 1 de cada N frames
        self.frame_count = 0
        self.process_every = 2  # probá 2 ó 3 si tu CPU sufre

        # guardar último frame
        self.last_frame_bgr = None

        # Crear un QTimer que actualice el frame cada 30 ms
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Timer de captura cada 3 segundos (no arrancar aquí; sólo durante la sesión)
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_to_base64)

        # Timer de sesión (11 segundos) - se crea pero se arranca en start_session
        self.session_timer = QTimer()
        self.session_timer.setSingleShot(True)
        # cuando expire, primero detener capture_timer y luego finalizar la sesión (evita captura en t~fin)
        self.session_timer.timeout.connect(self.session_expired)

        # Estado de sesión y almacenamiento
        self.session_active = False
        self.session_dir = None
        self.session_images = []           # lista de rutas de archivos guardados en la sesión
        self.session_predictions = {}      # mapa ruta -> prediction

        # Conectar botón "Comenzar" que en la UI se llama btnComenzar
        try:
            from PySide6.QtWidgets import QPushButton
            btn_start = self.ui.findChild(QPushButton, "btnComenzar")
            if btn_start:
                btn_start.clicked.connect(self.start_session)
            # Conectar botón reiniciar (si existe)
            btn_restart = self.ui.findChild(QPushButton, "btnReiniciar")
            if btn_restart:
                btn_restart.clicked.connect(self.restart_camera)
        except Exception:
            # si no existe el botón en la UI, no hacer nada (puede llamarse start_session manualmente)
            pass


        self.ui.show()

    def log_session_error(self, message: str):
        """Guardar mensajes de error en un archivo errors.log dentro de la sesión (no mostrar al usuario)."""
        try:
            if self.session_dir is not None:
                p = self.session_dir / "errors.log"
                with open(p, "a", encoding="utf-8") as f:
                    f.write(f"{int(time.time())}: {message}\n")
        except Exception as e:
            print(f"No se pudo escribir errors.log: {e}")

    def session_expired(self):
        """Handler al expirar el timer de sesión: detener capture_timer para evitar captura en el borde, luego finalizar."""
        print("Session timer expired: deteniendo capture_timer y finalizando sesión.")
        try:
            if self.capture_timer.isActive():
                self.capture_timer.stop()
        except Exception:
            pass
        # Llamar a end_session para hacer la limpieza restante
        self.end_session()

    def show_session_popup(self, proba: float):
        """
        Abre un popup modal usando ui/popup_window.ui.
        Si tu popup tiene:
          - QLabel con objectName 'lblFoto' -> muestra la última imagen correcta si existe
          - QLabel con objectName 'lblMensaje' -> muestra un resumen corto
          - QPushButton con objectName 'btnCerrar' -> cierra el popup
          - QPushButton con objectName 'btnReintentar' -> reinicia la cámara/sesión
        """
        try:
            # Cargar el UI del popup
            dialog_widget = load_ui(self.pop_up)  # debe ser un QDialog en el .ui; si es QWidget, igual funciona
            # Si la raíz no es QDialog, lo forzamos dentro de un QDialog simple
            if not isinstance(dialog_widget, QDialog):
                dlg = QDialog(self)
                dialog_widget.setParent(dlg)
                dialog_widget.setWindowFlags(dialog_widget.windowFlags() & ~Qt.Window)
                dlg.setWindowTitle("Resultados de la sesión")
                dlg.setModal(True)
                dlg_layout = dlg.layout()
                if dlg_layout is None:
                    from PySide6.QtWidgets import QVBoxLayout
                    dlg_layout = QVBoxLayout(dlg)
                dlg_layout.addWidget(dialog_widget)
                qdialog = dlg
            else:
                qdialog = dialog_widget
                qdialog.setModal(True)

            # Rellenar contenido dinámico si existen los widgets
            lbl_nota = qdialog.findChild(QLabel, "lblRes")

            if proba > 5:
                lbl_nota.setPixmap(self.correcto_icon)
            else:
                lbl_nota.setPixmap(self.error_icon)


            # Conexión de botones si existen
            btn_cerrar = qdialog.findChild(QPushButton, "btnCerrar")
            if btn_cerrar:
                btn_cerrar.clicked.connect(qdialog.accept)
            qdialog.exec()
        except Exception as e:
            print(f"[POPUP] No se pudo abrir el popup: {e}")

    def restart_camera(self):
        """Reinicia la cámara y limpia la vista final para poder intentar de nuevo."""
        print("Reiniciando cámara y limpiando vista.")
        try:

            self.total = 0
            self.cant_ok = 0

            # Si hay una sesión activa, cancelarla limpiamente
            if self.session_timer.isActive():
                self.session_timer.stop()
            if self.capture_timer.isActive():
                self.capture_timer.stop()
            self.session_active = False

            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

        # Reabrir cámara
        self.cap = cv2.VideoCapture(0)
        # limpiar vistas
        try:
            self.label.clear()
            self.lbl_result.clear()
        except Exception:
            pass

        # Asegurar que la vista previa vuelve a arrancar
        if not self.timer.isActive():
            self.timer.start(30)

        print("Cámara reiniciada.")

    def post_request(self, image: str):
        payload = {"image": image}  # ✅ clave correcta
        print("Enviando request al servidor.")
        try:
            r = requests.post("http://127.0.0.1:8000/predictOne", json=payload, timeout=120)
        except requests.RequestException as e:
            print(f"[POST] Error de red: {e}")
            return {"error": str(e)}

        try:
            data = r.json()
        except ValueError:
            return {"error": "Respuesta no es JSON", "raw": r.text}

        if not r.ok:
            # 👇 deja este print para ver el detalle de FastAPI
            print("[POST] HTTP error:", r.status_code, data)
            return {"error": "HTTP error", "status": r.status_code, "detail": data.get("detail", data)}

        label = data.get("label")
        if label is None:
            return {"error": "Campo 'label' ausente", "data": data}

        print("Resultado de prediccion:", label)
        return {"prediction": label, "raw": data}

    """def post_request(self, image: str):
        ""Envía la imagen al servidor y devuelve un dict consistente.
        En caso de éxito: {"prediction": <str>, "raw": <response-json>}.
        En caso de error: {"error": <mensaje>, ...}.
        ""
        payload = {"image": image}  # se pasa una imagen al request
        print("Enviando request al servidor.")
        try:
            r = requests.post("http://127.0.0.1:8000/predictOne", json=payload, timeout=120)

        except requests.RequestException as e:
            # Error de red / timeout
            print(f"[POST] Error de red: {e}")
            return {"error": str(e)}

        try:
            data = r.json()

        except ValueError:
            # No era JSON
            return {"error": "Respuesta no es JSON", "raw": r.text}

        if not r.ok:
            # FastAPI suele usar {"detail": ...}
            return {"error": "HTTP error", "status": r.status_code, "detail": data.get("detail", data)}

        prediction = data.get("prediction")

        if prediction is None:
            # No vino 'prediction'; devuelve lo que vino para depurar
            return {"error": "Campo 'prediction' ausente", "data": data}

        print("Resultado de prediccion:", prediction)
        return {"prediction": prediction, "raw": data}"""


    def set_icon_result(self, result: str):

        self.total += 1

        if result == "correcta":
            self.lbl_result.setPixmap(self.correcto_icon)
            self.cant_ok += 1
        else:
            self.lbl_result.setPixmap(self.error_icon)

    def start_session(self):
        """Iniciar una sesión de 11 segundos: guardar fotos cada 3s y evaluar."""
        self.label.clear()
        if self.session_active:
            print("Ya hay una sesión en curso.")
            return

        timestamp = int(time.time())
        session_name = f"session_{timestamp}"
        self.session_dir = self.base_dir / "captures" / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.session_active = True
        self.session_images = []
        self.session_predictions = {}

        # Asegurar que la cámara y timers están corriendo
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if not self.timer.isActive():
            self.timer.start(30)
        if not self.capture_timer.isActive():
            # TODO
            self.capture_timer.start(1500)

        # Hacer una captura inmediata al iniciar la sesión (t=0)
        try:
            self.capture_to_base64()
        except Exception as e:
            print(f"Error al capturar inmediatamente: {e}")

        # Iniciar timer de 12 segundos
        self.session_timer.start(100000)
        print(f"Sesión iniciada. Guardando en: {self.session_dir}")

    def end_session(self):
        """Finalizar sesión: detener cámara, timers y mostrar la última foto correcta (si existe)."""
        if not self.session_active:
            return

        self.session_active = False
        self.session_timer.stop()

        # Detener captura y actualización de frames y liberar la cámara
        if self.capture_timer.isActive():
            self.capture_timer.stop()
        if self.timer.isActive():
            self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # Buscar la última imagen marcada como 'correcta'
        last_correct = None
        for path in reversed(self.session_images):
            pred = self.session_predictions.get(str(path))
            if pred == "correcta":
                last_correct = path
                break

        if last_correct:
            pix = QPixmap(str(last_correct))
            pix = pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(pix)
            print(f"Mostrando última foto correcta: {last_correct}")
        else:
            # No mostrar foto final si no hay correcta (limpiar o dejar como estaba)
            self.label.clear()
            print("No se encontró ninguna foto correcta en la sesión.")

        print("Sesión finalizada.")

        try:
            proba = 10 * self.cant_ok / self.total

            self.total = 0
            self.cant_ok = 0

            print("proba ",proba)
        except Exception as e:
            print(f"Error al calcular el puntaje: {e}")

        self.show_session_popup(proba)

    def capture_to_base64(self):
        """Toma el último frame, lo guarda en disco si hay sesión activa, lo envía y guarda la predicción."""
        if self.last_frame_bgr is None:
            return

        # Codificar a JPEG en memoria (calidad 90)
        ok, buf = cv2.imencode(".jpg", self.last_frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            print("No se pudo codificar el frame.")
            return

        # Si hay una sesión activa, guardar la imagen en disco
        saved_path = None
        if self.session_active and self.session_dir is not None:
            idx = len(self.session_images) + 1
            filename = self.session_dir / f"frame_{idx:03d}_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
            try:
                with open(filename, "wb") as f:
                    f.write(buf.tobytes())
                saved_path = filename
                self.session_images.append(filename)
            except Exception as e:
                print(f"No se pudo guardar la imagen en disco: {e}")

        b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")

        result = self.post_request(b64_str)
        # Manejar errores del post_request
        if isinstance(result, dict) and "error" in result:
            # Registrar error para revisar luego (no mostrar al usuario)
            err_msg = result.get("error")
            print(f"[PREDICT] Error: {err_msg}")
            # registrar en archivo de la sesión
            try:
                self.log_session_error(str(err_msg))
            except Exception:
                pass
            # marcar la predicción como error si se guardó la imagen
            if saved_path is not None:
                self.session_predictions[str(saved_path)] = f"ERROR: {err_msg}"
            return

        # extraer prediction en los distintos formatos posibles
        prediction = None
        if isinstance(result, dict) and "prediction" in result:
            prediction = result["prediction"]
        elif isinstance(result, str):
            prediction = result
        else:
            print(f"[PREDICT] formato inesperado: {result}")
            return

        print(f"[PREDICT] prediction={prediction}")

        # guardar la predicción asociada al archivo si fue guardado
        if saved_path is not None:
            self.session_predictions[str(saved_path)] = prediction

        # Actualizar el ícono de resultado
        self.set_icon_result(prediction)


    """def update_frame(self):
        ok, frame_bgr = self.cap.read()
        if not ok:
            return

        self.last_frame_bgr = frame_bgr  # ¡guardar último frame!

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))"""

    def update_frame(self):
        ok, frame_bgr = self.cap.read()
        if not ok:
            return

        # Guardamos el frame limpio para enviar al backend
        self.last_frame_bgr = frame_bgr

        self.frame_count += 1
        do_process = (self.frame_count % self.process_every == 0)

        draw_on = frame_bgr.copy()

        # Procesá en baja resolución para ganar FPS (mismo aspect ratio)
        # y dibujá SIEMPRE sobre el tamaño original para que no pierda nitidez.
        if do_process:
            h0, w0 = frame_bgr.shape[:2]
            small = cv2.resize(frame_bgr, (w0 // 2, h0 // 2))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = self.holistic.process(small_rgb)

            has_any = (res.pose_landmarks is not None) or \
                      (res.left_hand_landmarks is not None) or \
                      (res.right_hand_landmarks is not None)

            if has_any:
                # Guardamos como último válido y reseteamos la edad
                self.last_valid_res = res
                self.last_valid_age = 0
            else:
                # Si no hubo detección, envejecemos el último válido
                self.last_valid_age += 1
        else:
            # No procesamos este frame; si no hay nuevo res, solo envejecemos si ya veníamos sosteniendo
            if self.last_valid_res is not None:
                self.last_valid_age += 1

        # Elegimos qué resultados dibujar:
        res_to_draw = None
        if do_process:
            # Si procesamos recién, usamos el res actual si tiene algo;
            # si no, probamos el último válido si está dentro del hold.
            if 'res' in locals():
                any_now = (res.pose_landmarks is not None) or \
                          (res.left_hand_landmarks is not None) or \
                          (res.right_hand_landmarks is not None)
                if any_now:
                    res_to_draw = res
                elif self.last_valid_res is not None and self.last_valid_age <= self.hold_max_frames:
                    res_to_draw = self.last_valid_res
        else:
            # En frames no procesados, como mínimo intentamos dibujar el último válido dentro del hold
            if self.last_valid_res is not None and self.last_valid_age <= self.hold_max_frames:
                res_to_draw = self.last_valid_res

        # Dibujo sobre el frame original (draw_on)
        if res_to_draw is not None:
            ds = self.mp_draw.DrawingSpec(thickness=2, circle_radius=2)
            # Pose
            if res_to_draw.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    image=draw_on,
                    landmark_list=res_to_draw.pose_landmarks,
                    connections=self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=ds,
                    connection_drawing_spec=ds
                )
            # Mano izquierda
            if res_to_draw.left_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image=draw_on,
                    landmark_list=res_to_draw.left_hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=ds,
                    connection_drawing_spec=ds
                )
            # Mano derecha
            if res_to_draw.right_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image=draw_on,
                    landmark_list=res_to_draw.right_hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=ds,
                    connection_drawing_spec=ds
                )
        else:
            # Si ya superamos el hold, reseteamos para no acumular edad infinita
            if self.last_valid_age > self.hold_max_frames:
                self.last_valid_res = None
                self.last_valid_age = 0

        # Mostrar en QLabel
        frame_rgb = cv2.cvtColor(draw_on, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        """Cerrar cámara al cerrar la app"""
        try:
            if self.capture_timer.isActive():
                self.capture_timer.stop()
        except Exception:
            pass
        try:
            if self.timer.isActive():
                self.timer.stop()
        except Exception:
            pass
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

        try:
            if hasattr(self, "holistic") and self.holistic is not None:
                self.holistic.close()
        except Exception:
            pass

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    sys.exit(app.exec())
