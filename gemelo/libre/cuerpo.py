"""
El cuerpo de la mano bebé: de 10 músculos a la pose
===================================================

La mano no conoce esta función: solo activa músculos (0 a 1) y ve dónde
quedan sus dedos y su muñeca. Son los mismos 10 músculos del gemelo:

  FDS, FDP   flexores de los 4 dedos (comparten tendones: se mueven juntos)
  FPL / EPL  flexor / extensor largo del pulgar
  EDC        extensor común de los 4 dedos
  EIP        extensor propio del índice (lo único que lo separa de los demás)
  FCR, FCU   flexores de la muñeca      ECR, ECU   extensores de la muñeca

Con esa anatomía unas poses son fáciles (puño, mano abierta), otras exigen
coordinar músculos (señalar, pulgar arriba) y otras son imposibles (cerrar
solo el anular). Eso es lo que la mano tiene que descubrir.
"""

import numpy as np

from .. import config  # noqa: F401
from ..datos_emg import POSES_6
import ia_mano as ia
import simulador as sm

MUSCULOS = sm.NOMBRES_MUSCULOS            # FDS FDP FPL FCR FCU EDC EIP EPL ECR ECU
I = {m: k for k, m in enumerate(MUSCULOS)}
GANANCIA_DEDO = np.array([1.0, 1.06, 1.12, 1.18])   # índice, medio, anular, meñique
# reparto de cada músculo por dedo (índice, medio, anular, meñique): el flexor profundo (FDP)
# actúa sobre todo en el índice y el superficial (FDS) en los otros tres; así la pinza existe,
# pero solo coordinando varios músculos
W_FDS = np.array([0.6, 1.2, 1.2, 1.1])
W_FDP = np.array([1.5, 0.8, 0.8, 0.8])
W_EDC = np.array([1.0, 1.5, 1.5, 1.5])
W_EIP = np.array([1.7, 0.15, 0.15, 0.15])
DOF = ["pulgar", "índice", "medio", "anular", "meñique", "muñeca"]


def pose(a, rng=None):
    """a: (n, 10) activaciones en [0, 1] -> (n, 6) pose (5 dedos 0-1, muñeca -1 a 1)."""
    a = np.clip(np.atleast_2d(a), 0, 1)
    m = lambda k: a[:, I[k]]
    muneca = np.tanh(1.7 * (m("ECR") + m("ECU") - m("FCR") - m("FCU")) + 0.25 * m("EDC") - 0.25 * m("FDS"))
    dedos = []
    for d in range(4):
        fuerza = 1.1 * (W_FDS[d] * m("FDS") + W_FDP[d] * m("FDP")) - W_EDC[d] * m("EDC") - W_EIP[d] * m("EIP")
        fuerza = fuerza * GANANCIA_DEDO[d] + 0.3 * muneca          # tenodesis: extender la muñeca cierra los dedos
        dedos.append(0.25 + np.where(fuerza >= 0, 0.75, 0.25) * np.tanh(1.4 * fuerza))
    fp = 1.4 * m("FPL") + 0.35 * m("FDS") - 1.5 * m("EPL")
    pulgar = 0.25 + np.where(fp >= 0, 0.75, 0.25) * np.tanh(1.5 * fp)
    p = np.c_[pulgar, np.stack(dedos, axis=1), muneca]
    if rng is not None:
        p = p + rng.normal(0, 0.008, p.shape)
    return np.c_[np.clip(p[:, :5], 0, 1), np.clip(p[:, 5], -1, 1)]


def balbuceo(rng, n=1):
    """Activaciones al azar, como un bebé: pocos músculos a la vez, con fuerza variable."""
    on = rng.random((n, len(MUSCULOS))) < rng.uniform(0.15, 0.5, (n, 1))
    return (on * rng.uniform(0.2, 1.0, (n, len(MUSCULOS)))).astype(np.float32)


def distancia(p, q):
    """Distancia entre poses: los dedos pesan 1 y la muñeca (que va de -1 a 1) la mitad."""
    d = np.asarray(p) - np.asarray(q)
    return np.sqrt(np.sum(d[..., :5] ** 2, axis=-1) + (0.5 * d[..., 5]) ** 2)


def describir(p):
    """Nombre en palabras para una pose descubierta (sin usar la lista de gestos)."""
    p = np.asarray(p, float)
    cerca = int(np.argmin(distancia(POSES_6, p)))
    if distancia(POSES_6[cerca], p) < 0.22:
        return ia.NOMBRES[ia.GESTOS[cerca]], ia.GESTOS[cerca]
    estirados = [DOF[k] for k in range(5) if p[k] < 0.2]
    cerrados = [DOF[k] for k in range(5) if p[k] > 0.7]
    partes = []
    if p[0] < 0.2 and all(p[k] > 0.6 for k in range(1, 5)):
        partes.append("pulgar arriba")
    elif p[1] < 0.25 and all(p[k] > 0.6 for k in range(2, 5)):
        partes.append("señala con el índice")
    elif len(cerrados) >= 4:
        partes.append("mano cerrada")
    elif len(estirados) >= 4:
        partes.append("mano abierta")
    elif cerrados:
        partes.append("cierra " + ", ".join(cerrados))
    else:
        partes.append("dedos a medio cerrar")
    if p[5] > 0.45:
        partes.append("muñeca hacia el dorso")
    elif p[5] < -0.45:
        partes.append("muñeca hacia la palma")
    texto = ", ".join(partes)
    return texto[0].upper() + texto[1:], None


def describir_libre(p):
    """La pose en palabras, dedo por dedo, sin compararla con ninguna lista de gestos."""
    p = np.asarray(p, float)
    estados = {"estirado": [], "a medias": [], "cerrado": []}
    for k in range(5):
        estados["estirado" if p[k] < 0.3 else "cerrado" if p[k] > 0.7 else "a medias"].append(DOF[k])
    plural = {"estirado": "estirados", "a medias": "a medias", "cerrado": "cerrados"}
    if any(len(v) == 5 for v in estados.values()):
        texto = "todos los dedos " + plural[next(e for e, v in estados.items() if len(v) == 5)]
    else:
        partes = []
        for e, v in estados.items():
            if v:
                nombres = v[0] if len(v) == 1 else ", ".join(v[:-1]) + (" e " if v[-1].startswith("í") else " y ") + v[-1]
                partes.append(f"{nombres} {e if len(v) == 1 else plural[e]}")
        texto = "; ".join(partes)
    if p[5] > 0.35:
        texto += "; muñeca hacia el dorso"
    elif p[5] < -0.35:
        texto += "; muñeca hacia la palma"
    return texto[0].upper() + texto[1:]


class Mapa:
    """Plano 2D fijo para dibujar todo lo que la mano puede hacer (PCA de poses alcanzables)."""

    def __init__(self, semilla=7, n=30_000, celdas=36):
        rng = np.random.default_rng(semilla)
        P = pose(balbuceo(rng, n))
        self.media = P.mean(0)
        _, _, Vt = np.linalg.svd(P - self.media, full_matrices=False)
        self.ejes = Vt[:2]
        Z = self.proyectar(P)
        self.lo, self.hi = Z.min(0) - 0.05, Z.max(0) + 0.05
        self.celdas = celdas
        self.alcanzables = np.unique(self.celda(Z))
        self.nube = Z[rng.choice(len(Z), 2500, replace=False)]
        self.gestos = self.proyectar(POSES_6)

    def proyectar(self, P):
        return (np.atleast_2d(P) - self.media) @ self.ejes.T

    def celda(self, Z):
        ij = np.floor((Z - self.lo) / (self.hi - self.lo) * self.celdas).astype(int)
        ij = np.clip(ij, 0, self.celdas - 1)
        return ij[:, 0] * self.celdas + ij[:, 1]
