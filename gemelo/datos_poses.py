"""
D2 — Dataset de poses de mano (21 puntos, topología MediaPipe Hands)
====================================================================

Cinemática de la mano del gemelo (gemelo_digital.puntos_mano) llevada a los
21 puntos que entrega MediaPipe, para que los datos sintéticos y los de la
webcam (D3, capturar_poses.py) estén en el mismo formato:

   0 muñeca · 1-4 pulgar (CMC, MCP, IP, punta) · 5-8 índice (MCP, PIP, DIP,
   punta) · 9-12 medio · 13-16 anular · 17-20 meñique

Cada muestra trae los 6 grados de libertad que la generaron (5 dedos y
muñeca), así se puede medir con R² qué tanto una red recupera la pose.
Varía por "mano": tamaño, largo de falanges, separación de los dedos,
temblor y orientación de la cámara.
"""

import numpy as np

from . import config  # noqa: F401
from .datos_emg import POSES_6, POSE_REPOSO, NOMBRES_DOF, dividir_por_persona, guardar_h5
import ia_mano as ia

VERSION = "D2-1.0"
BASES = np.array([[-3.2, 2.2, -0.4], [-2.4, 8.0, 0], [-0.8, 8.3, 0], [0.8, 8.0, 0], [2.3, 7.4, 0]])
FALANGES = np.array([[3.0, 2.4, 1.9], [4.0, 2.4, 1.8], [4.4, 2.8, 1.9], [4.1, 2.6, 1.8], [3.2, 2.0, 1.6]])
FLEX_MAX = np.radians([80, 100, 70])
SEPARACION = np.radians([0, -10, -3, 4, 11])      # abanico de los dedos con la mano abierta
MUNECA = np.array([-0.3, -0.4, 0.0])
CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
DEDO_DE_PUNTO = np.array([-1] + [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4)


def _rot_x(p, ang):
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return np.stack([p[..., 0], p[..., 1] * c - p[..., 2] * s, p[..., 1] * s + p[..., 2] * c], axis=-1)


def landmarks(dof, escala=None, largos=None, bases=None, separacion=None):
    """dof (n, 6) -> puntos (n, 21, 3) en cm, antebrazo fijo y muñeca rotando."""
    dof = np.atleast_2d(dof).astype(np.float64)
    n = len(dof)
    escala = np.ones(n) if escala is None else escala
    largos = np.broadcast_to(FALANGES, (n, 5, 3)) if largos is None else largos
    bases = np.broadcast_to(BASES, (n, 5, 3)) if bases is None else bases
    separacion = np.broadcast_to(SEPARACION, (n, 5)) if separacion is None else separacion
    P = np.zeros((n, 21, 3))
    P[:, 0] = MUNECA
    for i in range(5):
        f = np.clip(dof[:, i], 0, 1)
        p = bases[:, i].copy()
        P[:, 1 + 4 * i] = p
        if i == 0:                                   # pulgar: se opone hacia la palma
            d0, d1 = np.array([-0.55, 0.83, 0.0]), np.array([0.75, 0.35, -0.55])
            acum = np.zeros(n)
            for k in range(3):
                acum = np.minimum(1.0, acum + f * 0.45)
                d = (1 - acum)[:, None] * d0 + acum[:, None] * d1
                p = p + largos[:, 0, k:k + 1] * d / np.linalg.norm(d, axis=1, keepdims=True)
                P[:, 2 + k] = p
        else:
            ang = np.zeros(n)
            sep = separacion[:, i] * (1 - 0.8 * f)       # al cerrar, los dedos se juntan
            for k in range(3):
                ang = ang + f * FLEX_MAX[k]
                d = np.stack([np.sin(sep) * np.cos(ang), np.cos(sep) * np.cos(ang), -np.sin(ang)], axis=1)
                p = p + largos[:, i, k:k + 1] * d
                P[:, 2 + 4 * i + k] = p
    P = _rot_x(P - MUNECA, np.radians(55) * np.clip(dof[:, 5], -1, 1)) + MUNECA
    return P * escala[:, None, None]


def normalizar(P):
    """Misma normalización para sintético y webcam: origen en la muñeca (punto 0) y
    escala = distancia muñeca -> MCP del medio (punto 9). No quita la rotación,
    porque ahí está la flexión/extensión de la muñeca."""
    P = np.asarray(P, np.float64)
    P = P - P[:, :1]
    esc = np.linalg.norm(P[:, 9], axis=1)
    return (P / np.maximum(esc, 1e-6)[:, None, None]).astype(np.float32)


def _rotacion_camara(rng, n, grados=10):
    a = np.radians(rng.uniform(-grados, grados, (n, 3)))
    R = np.empty((n, 3, 3))
    for k in range(n):
        cx, cy, cz = np.cos(a[k])
        sx, sy, sz = np.sin(a[k])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R[k] = Rz @ Ry @ Rx
    return R


def generar_d2(n=60000, n_manos=600, semilla=2027, ruta=None):
    """60.000 poses de 600 manos virtuales: 75 % gestos (con amplitud parcial) y
    25 % transiciones entre dos gestos."""
    config.crear_carpetas()
    ruta = ruta or config.DATOS / "D2_poses_sinteticas.h5"
    rng = np.random.default_rng(semilla)
    mano = rng.integers(0, n_manos, n)
    escala_m = rng.uniform(0.85, 1.15, n_manos)
    largos_m = FALANGES * rng.normal(1, 0.05, (n_manos, 5, 3))
    bases_m = BASES + rng.normal(0, 0.15, (n_manos, 5, 3))
    sep_m = SEPARACION * rng.uniform(0.6, 1.4, (n_manos, 5))

    g = rng.integers(0, ia.NUM_GESTOS, n)
    amp = np.where(rng.random(n) < 0.2, 1.0, rng.uniform(0.3, 1.0, n))
    dof = POSE_REPOSO + amp[:, None] * (POSES_6[g] - POSE_REPOSO)
    mezcla = rng.random(n) < 0.25                       # transiciones A -> B
    g2 = rng.integers(0, ia.NUM_GESTOS, n)
    t = rng.uniform(0, 1, n)
    dof = np.where(mezcla[:, None], (1 - t)[:, None] * POSES_6[g] + t[:, None] * POSES_6[g2], dof)
    gesto = np.where(mezcla & (t >= 0.5), g2, g)
    dof = dof + rng.normal(0, 0.04, dof.shape) * np.r_[np.ones(5), 1.5]
    dof = np.c_[np.clip(dof[:, :5], 0, 1), np.clip(dof[:, 5], -1, 1)]

    P = landmarks(dof, escala_m[mano], largos_m[mano], bases_m[mano], sep_m[mano])
    P = np.einsum("nij,nkj->nki", _rotacion_camara(rng, n), P)        # orientación de la cámara
    P = P + rng.normal(0, 0.12, P.shape)                                # temblor / ruido de detección
    d = {"landmarks": normalizar(P), "dof": dof.astype(np.float32), "gesto": gesto.astype(np.int8),
         "amplitud": np.where(mezcla, np.nan, amp).astype(np.float32), "mezcla": mezcla,
         "persona": mano.astype(np.int32)}
    d["split"] = dividir_por_persona(d["persona"], semilla)
    guardar_h5(d, ruta, {"version": VERSION, "semilla": semilla, "n_manos": n_manos, "gestos": ia.GESTOS,
                         "dof": NOMBRES_DOF, "puntos": 21, "topologia": "MediaPipe Hands",
                         "normalizacion": "origen muñeca, escala |p9 - p0|, sin quitar rotación",
                         "fuente": "cinemática del gemelo digital", "autor": config.AUTOR})
    return ruta
