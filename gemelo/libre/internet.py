"""
Lo que la mano bebé aprende de internet: manos de personas reales (HaGRID v2)
=============================================================================

HaGRID v2 (Kapitanov et al., WACV 2024; licencia CC BY-SA 4.0): más de medio
millón de fotos de personas haciendo 33 gestos cotidianos, cada mano con sus
21 puntos (MediaPipe). Aquí NO se usan las fotos: solo los 21 puntos.

De cada mano humana se saca "cuánto se dobla cada dedo" (0 = estirado,
1 = cerrado), que es lo que la mano bebé intenta imitar con sus músculos:

  dedo:   recto = la distancia de la base a la punta es igual al largo
          de sus tres falanges (eso no cambia aunque el dedo apunte a la
          cámara); doblado = la cuerda es mucho más corta que el largo.
  pulgar: lo mismo + qué tanto cruza la palma (oposición).
  muñeca: las fotos no muestran el antebrazo → queda en 0 (recta).
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np

from .. import config

URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
ZIP = config.RAIZ_DATOS / "datos" / "externos" / "hagrid" / "annotations.zip"
NPZ = config.RAIZ_DATOS / "datos" / "externos" / "hagrid" / "manos_internet.npz"
POR_GESTO = 600
NOMBRES = {
    "call": "llamar por teléfono", "dislike": "no me gusta", "fist": "puño", "four": "cuatro", "grabbing": "agarrar",
    "grip": "sujetar", "hand_heart": "medio corazón", "hand_heart2": "medio corazón (otro)", "holy": "manos juntas",
    "like": "me gusta", "little_finger": "meñique arriba", "middle_finger": "dedo medio", "mute": "silencio", "ok": "ok",
    "one": "uno", "palm": "palma abierta", "peace": "paz", "peace_inverted": "paz al revés", "point": "señalar",
    "rock": "rock", "stop": "alto", "stop_inverted": "alto al revés", "take_picture": "tomar una foto",
    "three": "tres", "three2": "tres (con pulgar)", "three3": "tres (otro)", "three_gun": "pistola de tres",
    "thumb_index": "pulgar e índice", "thumb_index2": "pulgar e índice (otro)", "timeout": "tiempo fuera",
    "two_up": "dos arriba", "two_up_inverted": "dos arriba al revés", "xsign": "cruz"}
DEDOS = {1: [5, 6, 7, 8], 2: [9, 10, 11, 12], 3: [13, 14, 15, 16], 4: [17, 18, 19, 20]}
CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def _rectitud(P):
    seg = np.linalg.norm(np.diff(P, axis=-2), axis=-1).sum(-1)
    cuerda = np.linalg.norm(P[..., -1, :] - P[..., 0, :], axis=-1)
    return cuerda / np.maximum(seg, 1e-9)


def a_pose(L):
    """L: (n, 21, 2) puntos de la mano en la foto -> (n, 6) pose para la mano bebé."""
    L = np.asarray(L, float)
    palma = np.linalg.norm(L[:, 9] - L[:, 0], axis=-1)[:, None]
    dedos = []
    for f, idx in DEDOS.items():
        curl = np.clip((0.97 - _rectitud(L[:, idx])) / 0.55, 0, 1)
        # muy cerrado: la punta queda más cerca de la muñeca que la articulación media
        cerrado = np.linalg.norm(L[:, idx[3]] - L[:, 0], axis=-1) < np.linalg.norm(L[:, idx[1]] - L[:, 0], axis=-1)
        dedos.append(np.where(cerrado, np.maximum(curl, 0.85), curl))
    curl_p = np.clip((0.97 - _rectitud(L[:, [1, 2, 3, 4]])) / 0.45, 0, 1)
    oposicion = np.clip(1.15 - np.linalg.norm(L[:, 4] - L[:, 13], axis=-1)[:, None][:, 0] / palma[:, 0], 0, 1)
    pulgar = np.maximum(curl_p, oposicion)
    return np.c_[pulgar, np.stack(dedos, 1), np.zeros(len(L))].astype(np.float32)


def normalizar_2d(L):
    """Para dibujar: muñeca abajo al centro, escala de la palma, dedos hacia arriba."""
    L = np.asarray(L, float)
    L = L - L[:, :1]
    L[..., 1] *= -1                                    # en las fotos y crece hacia abajo
    ang = np.arctan2(L[:, 9, 0], L[:, 9, 1])
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    R = np.stack([np.c_[c, -s], np.c_[s, c]], 1)       # rota para que el dedo medio apunte arriba
    L = np.einsum("nij,nkj->nki", R, L)
    return (L / np.maximum(np.linalg.norm(L[:, 9], axis=-1), 1e-9)[:, None, None]).astype(np.float32)


def procesar(zip_ruta=ZIP, salida=NPZ, por_gesto=POR_GESTO, semilla=0):
    """Lee cada gesto del .zip sin descomprimirlo todo y guarda una muestra compacta."""
    rng = np.random.default_rng(semilla)
    Ls, clases, usuarios = [], [], []
    orden = sorted(NOMBRES)
    with zipfile.ZipFile(zip_ruta) as z:
        archivos = {Path(n).stem: n for n in z.namelist() if n.endswith(".json") and "/train/" in n.replace("\\", "/")}
        for k, g in enumerate(orden):
            if g not in archivos:
                continue
            with z.open(archivos[g]) as f:
                datos = json.load(io.TextIOWrapper(f, encoding="utf-8"))
            manos = []
            for item in datos.values():
                for etiqueta, lm in zip(item.get("labels", []), item.get("hand_landmarks", [])):
                    if etiqueta == g and lm is not None and len(lm) == 21:
                        manos.append((lm, item.get("user_id", "")))
            sel = rng.choice(len(manos), min(por_gesto, len(manos)), replace=False)
            for i in sel:
                Ls.append(manos[i][0])
                clases.append(k)
                usuarios.append(manos[i][1])
            del datos
    L = np.asarray(Ls, np.float32)
    np.savez_compressed(salida, landmarks=L, landmarks_vista=normalizar_2d(L), pose=a_pose(L), clase=np.array(clases, np.int16),
                        claves=np.array(orden), nombres=np.array([NOMBRES[g] for g in orden]),
                        usuario=np.array(usuarios), fuente=np.array("HaGRID v2 (Kapitanov et al., WACV 2024) · CC BY-SA 4.0"))
    return salida


# ------------------------------------------------------------------
# Descarga parcial: del .zip remoto (719 MB) solo se leen el índice y los
# primeros megas de cada gesto, que ya traen miles de manos
# ------------------------------------------------------------------
def _rango(url, ini, fin, intentos=6):
    """Bytes [ini, fin] del archivo remoto (curl con reintentos: el servidor a veces corta)."""
    import subprocess
    import time as _t
    for k in range(intentos):
        r = subprocess.run(["curl", "-s", "-f", "--max-time", "120", "--retry", "3", "--retry-all-errors",
                            "-r", f"{ini}-{fin}", url], capture_output=True)
        if r.returncode == 0 and len(r.stdout) == fin - ini + 1:
            return r.stdout
        _t.sleep(2 + 3 * k)
    raise ConnectionError(f"No se pudo leer el rango {ini}-{fin} de {url}")


def _tamano(url):
    import re
    import subprocess
    r = subprocess.run(["curl", "-s", "-D", "-", "-o", "NUL", "--retry", "5", "--retry-all-errors", "-r", "0-0", url],
                       capture_output=True, text=True)
    return int(re.search(r"Content-Range: bytes 0-0/(\d+)", r.stdout, re.I).group(1))


def _indice_zip(url):
    import struct
    total = _tamano(url)
    cola = _rango(url, total - 70_000, total - 1)
    k = cola.rfind(b"PK\x05\x06")
    tam_cd, ofs_cd = struct.unpack("<II", cola[k + 12:k + 20])
    cd = _rango(url, ofs_cd, ofs_cd + tam_cd - 1)
    entradas, i = {}, 0
    while cd[i:i + 4] == b"PK\x01\x02":
        metodo, = struct.unpack("<H", cd[i + 10:i + 12])
        comp, = struct.unpack("<I", cd[i + 20:i + 24])
        n, e, c = struct.unpack("<HHH", cd[i + 28:i + 34])
        ofs, = struct.unpack("<I", cd[i + 42:i + 46])
        nombre = cd[i + 46:i + 46 + n].decode("utf-8", "replace")
        entradas[nombre] = (metodo, comp, ofs)
        i += 46 + n + e + c
    return entradas


def _items_parciales(url, metodo, comp, ofs, megas):
    """Descomprime solo el comienzo del JSON de un gesto y devuelve los registros completos."""
    import struct
    import zlib
    cab = _rango(url, ofs, ofs + 29)
    n, e = struct.unpack("<HH", cab[26:30])
    ini = ofs + 30 + n + e
    datos = _rango(url, ini, ini + min(comp, int(megas * 1e6)) - 1)
    texto = (zlib.decompressobj(-15).decompress(datos) if metodo == 8 else datos).decode("utf-8", "ignore")
    dec, i, items = json.JSONDecoder(), texto.index("{") + 1, []
    while True:
        try:
            while texto[i] in " \r\n\t,":
                i += 1
            _, i = dec.raw_decode(texto, i)                       # la clave (id de la foto)
            i = texto.index(":", i) + 1
            while texto[i] in " \r\n\t":
                i += 1
            valor, i = dec.raw_decode(texto, i)
            items.append(valor)
        except (ValueError, IndexError):
            return items                                           # el texto quedó cortado: se usan los completos


def descargar_muestra(salida=NPZ, por_gesto=POR_GESTO, megas=4.0, semilla=0, progreso=print):
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)
    entradas = _indice_zip(URL)
    orden = sorted(NOMBRES)
    Ls, clases = [], []
    for k, g in enumerate(orden):
        nombre = next((n for n in entradas if n.replace("\\", "/").endswith(f"train/{g}.json")), None)
        if nombre is None:
            continue
        items = _items_parciales(URL, *entradas[nombre], megas)
        manos = [lm for it in items for et, lm in zip(it.get("labels", []), it.get("hand_landmarks", []))
                 if et == g and lm is not None and len(lm) == 21]
        sel = rng.choice(len(manos), min(por_gesto, len(manos)), replace=False) if manos else []
        Ls += [manos[i] for i in sel]
        clases += [k] * len(sel)
        progreso(f"  {g:18s} {len(manos):6d} manos leídas · {len(sel)} guardadas")
    L = np.asarray(Ls, np.float32)
    np.savez_compressed(salida, landmarks=L, landmarks_vista=normalizar_2d(L), pose=a_pose(L), clase=np.array(clases, np.int16),
                        claves=np.array(orden), nombres=np.array([NOMBRES[g] for g in orden]),
                        fuente=np.array("HaGRID v2 (Kapitanov et al., WACV 2024) · CC BY-SA 4.0"))
    return salida


def cargar():
    if not NPZ.exists():
        return None
    d = np.load(NPZ, allow_pickle=False)
    return {k: d[k] for k in d.files}
