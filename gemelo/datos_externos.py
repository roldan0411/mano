"""
D4 — EMG real de fuentes públicas, llevado al formato del gemelo
================================================================

UCI "EMG data for gestures" (Lobov et al., Sensors 2018; licencia CC BY 4.0)
  36 personas, pulsera MYO de 8 canales alrededor del antebrazo.
  Clases: 1 reposo, 2 puño, 3 flexión, 4 extensión, 5 desviación radial,
  6 desviación cubital, 7 mano abierta, 0 sin marcar.

Para parecerse a la prótesis (2 canales):
  - Por persona se elige el canal más "flexor" (más activo en puño que en
    extensión) como canal 1 y el más "extensor" como canal 2.
  - Se pasa a 1 kHz uniforme y a cuentas del ADC con la ganancia del
    circuito real (AD620 ×106 · pasa-banda ×10 → ~1061 V/V).
  - Gestos que la mano no tiene (desviaciones) y lo no marcado quedan con
    gesto = -1: sirven para el aprendizaje NO supervisado (SimCLR).

Otras fuentes (requieren registro, se cargan cuando estén en disco):
  NinaPro DB2 (ninapro.hevs.ch) — 12 canales + guante de datos (R² de pose real).
  GRABMyo (PhysioNet) — 43 personas, 16 gestos.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .datos_emg import dividir_por_persona, guardar_h5
import ia_mano as ia

URL_UCI = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"
GANANCIA_CIRCUITO = (1 + 49.4e3 / 470) * 10            # ≈ 1061 V/V
CUENTAS_POR_VOLT = GANANCIA_CIRCUITO / 3.3 * 4095
MAPA_UCI = {1: "reposo", 2: "puno", 3: "flexion", 4: "extension", 7: "abierta"}
DESCARTE_MS = 300


def descargar_uci(destino=None):
    destino = Path(destino or config.DATOS / "externos" / "uci_emg")
    carpeta = destino / "EMG_data_for_gestures-master"
    if carpeta.exists():
        return carpeta
    destino.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL_UCI, timeout=120) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(destino)
    return carpeta


def _leer(archivo):
    d = pd.read_csv(archivo, sep="\t")
    t = d["time"].to_numpy(float)
    rejilla = np.arange(t[0], t[-1], 1000 / ia.FS)
    canales = np.stack([np.interp(rejilla, t, d[f"channel{i}"].to_numpy(float)) for i in range(1, 9)], axis=1)
    clase = d["class"].to_numpy()[np.clip(np.searchsorted(t, rejilla, side="right") - 1, 0, len(t) - 1)]
    return canales, clase


def _elegir_canales(registros):
    """Canal 1 = el más flexor (puño/extensión), canal 2 = el más extensor."""
    mav = {2: [], 4: []}
    for x, c in registros:
        for k in mav:
            if (c == k).any():
                mav[k].append(np.abs(x[c == k]).mean(axis=0))
    if not mav[2] or not mav[4]:
        return 0, 4
    puno, ext = np.mean(mav[2], axis=0), np.mean(mav[4], axis=0)
    c1 = int(np.argmax(puno / ext))
    r = ext / puno
    r[c1] = -np.inf
    return c1, int(np.argmax(r))


def generar_d4_uci(paso_ms=50, max_ventanas=100_000, semilla=2028, ruta=None):
    """Ventanas de 200 ms con paso de 50 ms (más solapadas que en D1 para llegar a
    ~50-100 mil ventanas con los 36 sujetos)."""
    config.crear_carpetas()
    ruta = ruta or config.DATOS / "D4_emg_uci.h5"
    carpeta = descargar_uci()
    out = {k: [] for k in ("x_a", "gesto", "clase_original", "persona", "repeticion", "ventana_idx")}
    canales_elegidos = {}
    rep = 0
    for sujeto in sorted(p for p in carpeta.iterdir() if p.is_dir()):
        pid = int(sujeto.name)
        registros = [_leer(a) for a in sorted(sujeto.glob("*.txt"))]
        c1, c2 = _elegir_canales(registros)
        canales_elegidos[pid] = (c1 + 1, c2 + 1)
        for x, clase in registros:
            senal = np.clip(np.round(2048 + x[:, [c1, c2]] * CUENTAS_POR_VOLT), 0, 4095).astype(np.int16)
            cortes = np.flatnonzero(np.diff(clase) != 0) + 1
            for ini, fin in zip(np.r_[0, cortes], np.r_[cortes, len(clase)]):
                seg = senal[ini + DESCARTE_MS:fin]
                if len(seg) < ia.VENTANA:
                    continue
                idx = np.arange(0, len(seg) - ia.VENTANA + 1, paso_ms)
                v = np.stack([seg[i:i + ia.VENTANA] for i in idx])
                k = int(clase[ini])
                g = ia.GESTOS.index(MAPA_UCI[k]) if k in MAPA_UCI else -1
                out["x_a"].append(v)
                out["gesto"].append(np.full(len(v), g, np.int8))
                out["clase_original"].append(np.full(len(v), k, np.int8))
                out["persona"].append(np.full(len(v), pid, np.int32))
                out["repeticion"].append(np.full(len(v), rep, np.int32))
                out["ventana_idx"].append(np.minimum(np.arange(len(v)), 127).astype(np.int8))
                rep += 1
    d = {k: np.concatenate(v) for k, v in out.items()}
    if len(d["gesto"]) > max_ventanas:                      # recorte uniforme, conserva a todos
        sel = np.sort(np.random.default_rng(semilla).choice(len(d["gesto"]), max_ventanas, replace=False))
        d = {k: v[sel] for k, v in d.items()}
    d["caracteristicas"] = ia.caracteristicas_lote(d["x_a"]).astype(np.float32)
    d["split"] = dividir_por_persona(d["persona"], semilla)
    guardar_h5(d, ruta, {"version": "D4-UCI-1.0", "fuente": "UCI EMG data for gestures (id 481)",
                         "url": URL_UCI, "licencia": "CC BY 4.0",
                         "cita": "Lobov S. et al. Latent Factors Limiting the Performance of sEMG-Interfaces. "
                                 "Sensors 2018;18(4):1122",
                         "gestos": ia.GESTOS, "gesto_-1": "sin marcar o gesto que la mano no tiene",
                         "paso_ms": paso_ms, "canales_elegidos_por_persona": canales_elegidos,
                         "cuentas_por_volt": CUENTAS_POR_VOLT, "autor": config.AUTOR})
    return ruta
