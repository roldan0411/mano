"""Rutas y semillas del proyecto.

El código vive en OneDrive (se sincroniza); los datasets y la caché viven
fuera, en C:\\Users\\<usuario>\\gemelo_mano, porque pesan y OneDrive bloquea
archivos grandes mientras los sube. Se puede cambiar con la variable de
entorno GEMELO_DATOS.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RAIZ_DATOS = Path(os.environ.get("GEMELO_DATOS", Path.home() / "gemelo_mano"))
DATOS = RAIZ_DATOS / "datos"
CACHE = RAIZ_DATOS / "cache"
MODELOS = RAIZ_PROYECTO / "modelos"      # exportables (van al repositorio)
REPORTES = RAIZ_PROYECTO / "reportes"    # figuras y animaciones HTML

if str(RAIZ_PROYECTO) not in sys.path:   # ia_mano.py y simulador.py están en la raíz
    sys.path.insert(0, str(RAIZ_PROYECTO))

AUTOR = "Roldán Rodríguez — Ing. CCIA"


def crear_carpetas():
    for c in (DATOS, CACHE, MODELOS, REPORTES):
        c.mkdir(parents=True, exist_ok=True)


def fijar_semillas(semilla=42):
    random.seed(semilla)
    np.random.seed(semilla)
    try:
        import keras
        keras.utils.set_random_seed(semilla)
    except ImportError:
        pass


def guardar_json(datos, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2, default=_a_json)


def _a_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def vigilar_padre(pid, creado):
    """Para los procesos de simulación (initializer del ProcessPoolExecutor): si el proceso que los
    creó muere, terminan solos. Sin esto, cerrar el motor a la fuerza los dejaba huérfanos."""
    import os
    import threading
    import time

    import psutil

    def vigilar():
        while True:
            try:
                vivo = psutil.Process(pid).create_time() == creado
            except psutil.Error:
                vivo = False
            if not vivo:
                os._exit(0)
            time.sleep(2)

    threading.Thread(target=vigilar, daemon=True, name="vigilar-padre").start()


def pool_vigilado(procesos):
    """ProcessPoolExecutor cuyos procesos se cierran solos si muere el que los creó."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    import psutil
    yo = os.getpid()
    return ProcessPoolExecutor(max_workers=procesos, initializer=vigilar_padre, initargs=(yo, psutil.Process(yo).create_time()))
