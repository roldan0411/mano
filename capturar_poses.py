"""
============================================================
 PRÓTESIS MIOELÉCTRICA CON RECONOCIMIENTO DE GESTOS POR IA
 D3 — Captura de poses reales de la mano con la webcam
------------------------------------------------------------
 Roldán Rodríguez — Ing. CCIA
============================================================

Qué hace:
  Detecta tu mano con MediaPipe Hands (21 puntos 3D) y guarda las poses
  etiquetadas con el gesto que estás haciendo, en el mismo formato que el
  dataset sintético D2 (origen en la muñeca, escala muñeca -> nudillo medio).

Controles (con la ventana de video activa):
  0-6     elegir gesto: 0 reposo · 1 abierta · 2 puño · 3 pinza ·
          4 señalar · 5 flexión de muñeca · 6 extensión de muñeca
  ESPACIO grabar / pausar
  q       guardar y salir

Consejo: antebrazo quieto y vertical, palma hacia la cámara, buena luz.
Haz cada gesto también suave y fuerte, y las transiciones entre ellos.

Usar (con el kernel del proyecto):
  C:\\Users\\rolda\\gemelo_mano\\venv\\Scripts\\python.exe capturar_poses.py
  ... --camara 1          (otra cámara)
"""

import argparse
import time
import urllib.request

import h5py
import numpy as np

from gemelo import config
from gemelo.datos_poses import normalizar
import ia_mano as ia

URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")


class Detector:
    """MediaPipe Hands: usa la API clásica si existe; si no, la de Tasks."""

    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.tipo = "solutions"
            self.hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=1,
                                                  min_detection_confidence=0.6, min_tracking_confidence=0.6)
        else:
            self.tipo = "tasks"
            ruta = config.DATOS / "mediapipe" / "hand_landmarker.task"
            if not ruta.exists():
                ruta.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(URL_MODELO, ruta)
            from mediapipe.tasks.python import BaseOptions, vision
            op = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(ruta)),
                                              running_mode=vision.RunningMode.VIDEO, num_hands=1)
            self.hands = vision.HandLandmarker.create_from_options(op)

    def detectar(self, rgb, t_ms):
        """-> (puntos 3D en metros (21,3) o None, puntos de imagen (21,3), lateralidad)"""
        if self.tipo == "solutions":
            r = self.hands.process(rgb)
            if not r.multi_hand_world_landmarks:
                return None, None, None
            mundo = r.multi_hand_world_landmarks[0].landmark
            img = r.multi_hand_landmarks[0].landmark
            lado = r.multi_handedness[0].classification[0].label
        else:
            imagen = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            r = self.hands.detect_for_video(imagen, int(t_ms))
            if not r.hand_world_landmarks:
                return None, None, None
            mundo, img = r.hand_world_landmarks[0], r.hand_landmarks[0]
            lado = r.handedness[0][0].category_name
        return (np.array([[p.x, p.y, p.z] for p in mundo]), np.array([[p.x, p.y, p.z] for p in img]), lado)


def a_marco_gemelo(mundo_m, lado):
    """MediaPipe (x derecha, y abajo, metros) -> marco del gemelo (y hacia los dedos, cm).
    La mano izquierda se refleja para que todas queden como la derecha."""
    P = mundo_m * 100.0
    P = np.c_[P[:, 0], -P[:, 1], -P[:, 2]]
    if lado == "Left":
        P[:, 0] *= -1
    return P


def main():
    ap = argparse.ArgumentParser(description="Captura de poses de mano (D3)")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--salida", default=str(config.DATOS / "D3_poses_webcam.h5"))
    ap.add_argument("--hz", type=float, default=15, help="poses guardadas por segundo")
    args = ap.parse_args()
    import cv2

    config.crear_carpetas()
    det = Detector()
    cam = cv2.VideoCapture(args.camara)
    if not cam.isOpened():
        raise SystemExit(f"No pude abrir la cámara {args.camara}")
    sesion = int(time.time())
    gesto, grabando = 0, False
    filas = {"landmarks": [], "landmarks_cm": [], "imagen": [], "gesto": [], "tiempo": [], "lado": []}
    ultimo = 0.0
    t0 = time.time()
    print(f"Detector MediaPipe ({det.tipo}). Gesto actual: {ia.GESTOS[gesto]}. ESPACIO para grabar.")
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ahora = time.time()
        mundo, img, lado = det.detectar(rgb, (ahora - t0) * 1000)
        if img is not None:
            h, w = frame.shape[:2]
            for x, y, _ in img:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (52, 120, 214), -1)
            if grabando and ahora - ultimo >= 1 / args.hz:
                P = a_marco_gemelo(mundo, lado)
                filas["landmarks_cm"].append(P)
                filas["landmarks"].append(normalizar(P[None])[0])
                filas["imagen"].append(img)
                filas["gesto"].append(gesto)
                filas["tiempo"].append(ahora)
                filas["lado"].append(0 if lado == "Right" else 1)
                ultimo = ahora
        estado = "GRABANDO" if grabando else "pausa"
        cv2.putText(frame, f"{gesto}: {ia.NOMBRES[ia.GESTOS[gesto]]}  [{estado}]  muestras: {len(filas['gesto'])}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if grabando else (60, 60, 60), 2)
        cv2.imshow("Captura de poses - gemelo de la mano (q para salir)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            grabando = not grabando
        if ord("0") <= k <= ord("6"):
            gesto = k - ord("0")
            print("Gesto:", ia.GESTOS[gesto])
    cam.release()
    cv2.destroyAllWindows()
    if not filas["gesto"]:
        print("No se grabó nada.")
        return
    nuevos = {k: np.asarray(v) for k, v in filas.items()}
    nuevos["sesion"] = np.full(len(nuevos["gesto"]), sesion, np.int64)
    with h5py.File(args.salida, "a") as f:
        for k, v in nuevos.items():
            v = v.astype(np.float32) if v.dtype == np.float64 else v
            if k not in f:
                f.create_dataset(k, data=v, maxshape=(None,) + v.shape[1:], chunks=True)
            else:
                f[k].resize(f[k].shape[0] + len(v), axis=0)
                f[k][-len(v):] = v
        f.attrs["manifiesto"] = ('{"version": "D3-1.0", "fuente": "webcam + MediaPipe Hands", '
                                 '"normalizacion": "igual que D2", "gestos": ' + str(ia.GESTOS).replace("'", '"') + "}")
    print(f"Guardadas {len(nuevos['gesto'])} poses en {args.salida}")


if __name__ == "__main__":
    main()
