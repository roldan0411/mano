"""
Fotos de los 7 gestos de cada campeona (mano InMoov)
====================================================

Para cada gesto se toma la pose que la campeona predice (mediana en el examen,
ventanas ya sostenidas) y se dibuja la mano InMoov en esa pose, con la pose
objetivo en azul transparente al lado de la palma. También una hoja con los 7.
"""

import threading

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .. import mano_inmoov
import ia_mano as ia

_PIEZAS = None
_LOCK = threading.Lock()
LUZ = np.array([0.3, 0.5, -0.8]) / np.linalg.norm([0.3, 0.5, -0.8])


def _piezas():
    global _PIEZAS
    with _LOCK:
        if _PIEZAS is None:
            _PIEZAS = mano_inmoov.construir(caras_falange=900, caras_palma=2400)
    return _PIEZAS


def _dibujar_mano(ax, piezas, plantilla, dof, color, alfa=1.0):
    T = mano_inmoov.poses_mundo(dof, plantilla)
    for k, m in piezas.items():
        mm = m.copy().apply_transform(T[k])
        tono = 0.45 + 0.55 * np.clip(mm.face_normals @ LUZ, 0, 1)
        c = np.clip(np.array(color)[None, :] * tono[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(mm.vertices[mm.faces], facecolors=np.c_[c, np.full(len(c), alfa)],
                                             linewidths=0))


def foto_gesto(ruta, dof, gesto, titulo, subtitulo):
    piezas, plantilla = _piezas()
    fig = Figure(figsize=(4.2, 4.6), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    _dibujar_mano(ax, piezas, plantilla, dof, (0.86, 0.84, 0.79))
    ax.set_xlim(-70, 70)
    ax.set_ylim(-20, 200)
    ax.set_zlim(-110, 110)
    ax.set_box_aspect((140, 220, 220))
    ax.view_init(elev=18, azim=-118, vertical_axis="y")
    ax.set_axis_off()
    fig.text(0.05, 0.95, titulo, fontsize=12, fontweight="bold", va="top")
    fig.text(0.05, 0.90, subtitulo, fontsize=9, color="#52514e", va="top")
    fig.subplots_adjust(0, 0, 1, 0.88)
    fig.savefig(ruta)
    return ruta


def fotos_campeona(carpeta, probs, pose, examen, n_campeona, gen):
    """7 fotos (una por gesto) + hoja de contacto, con la predicción de la campeona en el examen."""
    y = examen["gesto"].astype(int)
    sostenida = ~examen["transicion"]
    rutas = []
    for g in range(ia.NUM_GESTOS):
        m = (y == g) & sostenida
        dof = np.median(pose[m], axis=0)
        acierto = float(np.mean(probs[m].argmax(1) == g))
        rutas.append(foto_gesto(carpeta / f"gesto_{g}_{ia.GESTOS[g]}.png", dof, g, ia.NOMBRES[ia.GESTOS[g]],
                                f"Campeona #{n_campeona} · generación {gen} · acierto {acierto * 100:.0f} %"))
    # hoja de contacto
    import matplotlib.image as mpimg
    fig = Figure(figsize=(14, 4.6), dpi=90)
    FigureCanvasAgg(fig)
    for i, r in enumerate(rutas):
        ax = fig.add_subplot(1, 7, i + 1)
        ax.imshow(mpimg.imread(r))
        ax.set_axis_off()
    fig.suptitle(f"Campeona #{n_campeona} · generación {gen}", fontsize=13, fontweight="bold")
    fig.subplots_adjust(0, 0, 1, 0.9, 0.02, 0)
    fig.savefig(carpeta / "hoja_gestos.png")
    return rutas
