"""
G2 · Estabilidad: memoria y avance de las aplicaciones abiertas durante un tiempo largo
======================================================================================

Cada minuto anota la memoria (RSS) de cada proceso de las aplicaciones del gemelo,
la memoria de la GPU y cuánto avanzó cada una (generaciones del motor, de la
neuroevolución e intentos de la mano bebé). Si la memoria crece sin parar, hay una
fuga; si el avance se detiene, algo se colgó.

  python pruebas/monitor_estabilidad.py 60      (minutos)
"""

import csv
import pickle
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil

RAIZ = Path.home() / "gemelo_mano" / "motor"
SALIDA = Path(__file__).parent / "resultados" / "G2_estabilidad.csv"


def _rol(p):
    try:
        cmd = " ".join(p.cmdline())
    except psutil.Error:
        return None
    if "app_gemelo.py" in cmd:
        return "ventana --libre" if "--libre" in cmd else "ventana --neuro" if "--neuro" in cmd else "ventana principal"
    if "spawn_main" in cmd:
        return "proceso hijo"
    return None


def avance():
    gen = neuro = bebe = None
    try:
        gen = sqlite3.connect(RAIZ / "gemelo.db").execute("SELECT MAX(id) FROM generaciones").fetchone()[0]
    except Exception:
        pass
    try:
        with open(RAIZ / "neuro" / "poblacion.pkl", "rb") as f:
            neuro = pickle.load(f).gen
    except Exception:
        pass
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        with open(RAIZ / "bebe" / "bebe.pkl", "rb") as f:
            bebe = pickle.load(f).intentos
    except Exception:
        pass
    return gen, neuro, bebe


def gpu_mb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def main(minutos):
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(SALIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["minuto", "pid", "rol", "rss_mb", "cpu_pct", "gpu_mb", "gen_motor", "gen_neuro", "intentos_bebe"])
        for k in range(minutos + 1):
            procs = [(p, _rol(p)) for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower().startswith("python")]
            procs = [(p, r) for p, r in procs if r]
            for p, _ in procs:
                try:
                    p.cpu_percent(None)
                except psutil.Error:
                    pass
            time.sleep(1)
            g, n, b = avance()
            gm = gpu_mb()
            for p, r in procs:
                try:
                    w.writerow([k, p.pid, r, round(p.memory_info().rss / 1e6, 1), p.cpu_percent(None), gm, g, n, b])
                except psutil.Error:
                    pass
            f.flush()
            time.sleep(max(0, t0 + 60 * (k + 1) - time.time()))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
