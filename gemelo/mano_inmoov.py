"""
Mano InMoov i2 (derecha) para el visor 3D en vivo
=================================================

Las piezas STL vienen acostadas en sus placas de impresión. Este módulo:

  1. Las reconoce (falange proximal/media, bisagra distal, tapa de punta,
     cubiertas de la palma) por archivo y tamaño.
  2. Las lleva a un marco canónico: origen en el eje de su bisagra
     proximal, +Y hacia la punta, +Z hacia el dorso. El extremo con
     horquilla es el distal (allí entra la lengüeta de la falange siguiente).
  3. Las arma con la MISMA cinemática del gemelo (datos_poses.landmarks):
     flexión de cada dedo en 3 articulaciones (80°, 100°, 70°), pulgar en
     oposición y muñeca ±55°.
  4. Exporta mallas simplificadas + la plantilla de articulaciones a JSON
     para Three.js (reportes/vivo/mano_inmoov.json).

Unidades: milímetros.
"""

import base64
import json
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from . import config

ZIP_DEFECTO = config.RAIZ_PROYECTO / "Inmoov-hand-i2.zip"
FLEX_MAX = np.radians([80, 100, 70])
DEDOS = ["pulgar", "indice", "medio", "anular", "menique"]
# base de cada dedo en el marco de la palma (x lateral, y hacia los dedos, z dorso), mm
BASE_DEDOS = {"indice": (-31.5, 112.0, 0.0), "medio": (-10.5, 116.0, 0.0), "anular": (10.5, 112.0, 0.0),
              "menique": (31.5, 104.0, 0.0)}
SEPARACION = np.radians([0, -10, -3, 4, 11])
BASE_PULGAR = (-44.0, 38.0, -9.0)
DIR_PULGAR = (np.array([-0.55, 0.83, 0.0]), np.array([0.75, 0.35, -0.55]))   # abierto -> en oposición


def extraer(zip_ruta=None, destino=None):
    zip_ruta = Path(zip_ruta or ZIP_DEFECTO)
    destino = Path(destino or config.DATOS / "inmoov_zip")
    carpeta = destino / "Inmoov-hand-i2" / "i2_RightHand"
    if not carpeta.exists():
        destino.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_ruta) as z:
            z.extractall(destino)
    return carpeta


def _piezas(carpeta, archivo):
    m = trimesh.load(Path(carpeta) / archivo, force="mesh")
    return sorted([c for c in m.split(only_watertight=False) if len(c.faces) > 300],
                  key=lambda c: (round(c.bounds.mean(0)[0]), round(c.bounds.mean(0)[1])))


def _extremo_horquilla(c):
    """'+y' o '-y': en la horquilla no hay material en el centro de la pieza."""
    v, b = c.vertices, c.bounds
    cx = b.mean(0)[0]
    frac = {}
    for lado, y in (("-y", b[0, 1]), ("+y", b[1, 1])):
        cerca = np.abs(v[:, 1] - y) < 5
        frac[lado] = ((np.abs(v[:, 0] - cx) < 2.5) & cerca).sum() / max(cerca.sum(), 1)
    return min(frac, key=frac.get)


def _ancho_arriba(c):
    """Ancho relativo de los últimos 3 mm en Z: la bisagra termina en una lengüeta angosta (~0.35),
    la tapa de la punta no (~0.65)."""
    v = c.vertices
    arriba = v[v[:, 2] > c.bounds[1, 2] - 3]
    return float(np.ptp(arriba[:, 0]) / c.extents[0])


def _canonica_falange(c):
    """Falange impresa a lo largo de Y: origen en el eje proximal, +Y hacia la horquilla (distal)."""
    m = c.copy()
    b = m.bounds
    m.apply_translation(-np.array([b.mean(0)[0], 0, b.mean(0)[2]]))
    if _extremo_horquilla(c) == "-y":                      # horquilla abajo: girar 180° en Z
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
    r = m.extents[2] / 2                                   # extremos redondeados alrededor de la bisagra
    m.apply_translation([0, -(m.bounds[0, 1] + r), 0])
    largo = m.bounds[1, 1] - r                             # de bisagra a bisagra
    return m, float(largo)


def _canonica_punta(bisagra, tapa):
    """Bisagra distal (lengüeta arriba en Z) + tapa encajada abajo -> una sola punta, +Y hacia afuera."""
    b = bisagra.copy()
    t = tapa.copy()
    zb = b.bounds
    pivote_z = zb[1, 2] - 5.0
    b.apply_translation(-np.array([zb.mean(0)[0], zb.mean(0)[1], pivote_z]))
    zt = t.bounds
    # la tapa se apoya con su base en el extremo inferior de la bisagra (2 mm de encaje) y crece hacia afuera
    t.apply_translation(-np.array([zt.mean(0)[0], zt.mean(0)[1], zt[0, 2]]))
    t.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    t.apply_translation([0, 0, b.bounds[0, 2] + 2.0])
    punta = trimesh.util.concatenate([b, t])
    # Z de impresión (-) -> +Y canónico (hacia la punta)
    punta.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return punta


def _canonica_palma(cubierta, dorso, nudillos=None):
    """Cubiertas acostadas (grosor en Y, alto en Z) -> marco de la palma: y hacia los dedos,
    z hacia el dorso; la palmar queda hacia -z y la dorsal hacia +z."""
    giro = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])   # (x, y, z) -> (x, z, -y)
    partes = {}
    for nombre, pieza, lado in (("palma", cubierta, -1), ("dorso", dorso, 1)):
        p = pieza.copy()
        b = p.bounds
        p.apply_translation(-np.array([b.mean(0)[0], b[0, 1], b[0, 2]]))
        p.apply_transform(giro)                 # ahora el alto va en +y y el grosor en -z
        if lado > 0:                            # la dorsal se refleja al otro lado del plano medio
            p.apply_transform(np.diag([1, 1, -1, 1]))
            p.invert()
        partes[nombre] = p
    if nudillos is not None:
        k = nudillos.copy()
        b = k.bounds
        k.apply_translation(-np.array([b.mean(0)[0], b[0, 1], b[0, 2]]))
        k.apply_transform(giro)
        k.apply_transform(np.diag([1, 1, -1, 1]))
        k.invert()
        k.apply_translation([0, 96, 0])
        partes["nudillos"] = k
    return partes


def construir(carpeta=None, caras_falange=2600, caras_palma=7000):
    """Reconoce y orienta todas las piezas. Devuelve {nombre: malla canónica} y la plantilla."""
    carpeta = carpeta or extraer()
    fal = _piezas(carpeta, "i2_FingersX5V2.stl")
    puntas = _piezas(carpeta, "i2_FingersTipX5V2.stl")
    proximales = sorted([c for c in fal if c.extents[1] > 45], key=lambda c: -c.extents[0])
    medias = [c for c in fal if c.extents[1] < 45]
    bisagras = sorted([c for c in puntas if _ancho_arriba(c) < 0.5], key=lambda c: -c.extents[2])
    tapas = sorted([c for c in puntas if c not in bisagras], key=lambda c: -c.extents[0])
    piezas, largos = {}, {}
    for i, dedo in enumerate(DEDOS):
        p, l0 = _canonica_falange(proximales[i])        # la proximal más ancha es la del pulgar
        piezas[f"{dedo}_0"] = p
        if dedo == "pulgar":
            piezas[f"{dedo}_1"] = _canonica_punta(bisagras[0], tapas[0])
            largos[dedo] = [l0, 30.0]
        else:
            m, l1 = _canonica_falange(medias[i - 1])
            piezas[f"{dedo}_1"] = m
            piezas[f"{dedo}_2"] = _canonica_punta(bisagras[i], tapas[i])
            largos[dedo] = [l0, l1, 30.0]
    cubierta = _piezas(carpeta, "i2_PalmCoverV2.stl")[0]
    dorso = _piezas(carpeta, "i2_HandCoverV2.stl")[0]
    nudillos = max(_piezas(carpeta, "i2_CoverFingerV3.stl"), key=lambda c: c.extents[0])
    piezas.update(_canonica_palma(cubierta, dorso, nudillos))
    for k, m in piezas.items():
        n = caras_palma if k in ("palma", "dorso", "nudillos") else caras_falange
        if len(m.faces) > n:
            try:
                piezas[k] = m.simplify_quadric_decimation(face_count=n)
            except Exception:
                pass
    plantilla = {"largos": largos, "base_dedos": BASE_DEDOS, "separacion": SEPARACION.tolist(),
                 "flex_max": FLEX_MAX.tolist(), "base_pulgar": BASE_PULGAR,
                 "dir_pulgar": [d.tolist() for d in DIR_PULGAR], "muneca_max": float(np.radians(55)),
                 "pivote_muneca": [0.0, 0.0, 0.0]}
    return piezas, plantilla


# ------------------------------------------------------------------
# Cinemática en Python (la misma que hace el visor en JavaScript)
# ------------------------------------------------------------------
def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _t(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def _de_y_a(d):
    """Rotación que lleva +Y a la dirección d (para el pulgar)."""
    d = d / np.linalg.norm(d)
    y = np.array([0.0, 1.0, 0.0])
    eje = np.cross(y, d)
    s, c = np.linalg.norm(eje), float(y @ d)
    if s < 1e-9:
        return np.eye(4)
    return trimesh.transformations.rotation_matrix(np.arctan2(s, c), eje / s)


def poses_mundo(dof, plantilla):
    """dof (6,) -> {pieza: matriz 4x4 en el mundo}."""
    dof = np.asarray(dof, float)
    raiz = _rx(plantilla["muneca_max"] * np.clip(dof[5], -1, 1))
    T = {"palma": raiz, "dorso": raiz, "nudillos": raiz}
    for i, dedo in enumerate(DEDOS):
        f = float(np.clip(dof[i], 0, 1))
        L = plantilla["largos"][dedo]
        if dedo == "pulgar":
            d0, d1 = (np.array(v) for v in plantilla["dir_pulgar"])
            m = raiz @ _t(plantilla["base_pulgar"])
            acum = 0.0
            for k in range(2):
                acum = min(1.0, acum + f * 0.45 * (1.5 if k else 1.0))
                rot = _de_y_a((1 - acum) * d0 + acum * d1)
                T[f"pulgar_{k}"] = m @ rot
                m = m @ rot @ _t([0, L[k], 0]) @ np.linalg.inv(rot)
            continue
        sep = plantilla["separacion"][i] * (1 - 0.8 * f)
        m = raiz @ _t(plantilla["base_dedos"][dedo]) @ _rz(-sep)
        for k in range(3):
            m = m @ _rx(-f * plantilla["flex_max"][k])
            T[f"{dedo}_{k}"] = m
            m = m @ _t([0, L[k], 0])
    return T


def ensamblar(piezas, plantilla, dof):
    T = poses_mundo(dof, plantilla)
    return trimesh.util.concatenate([piezas[k].copy().apply_transform(T[k]) for k in piezas])


# ------------------------------------------------------------------
# Exportación para el visor web
# ------------------------------------------------------------------
def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def exportar_web(ruta, piezas=None, plantilla=None):
    if piezas is None:
        piezas, plantilla = construir()
    datos = {"unidades": "mm", "fuente": "InMoov i2 hand (Gael Langevin, inmoov.fr) — CC BY-NC",
             "plantilla": plantilla, "piezas": {}}
    for k, m in piezas.items():
        datos["piezas"][k] = {"posiciones": _b64(m.vertices.astype(np.float32)),
                              "indices": _b64(m.faces.astype(np.uint32)), "n_vertices": int(len(m.vertices)),
                              "n_caras": int(len(m.faces))}
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(json.dumps(datos), encoding="utf-8")
    return ruta
