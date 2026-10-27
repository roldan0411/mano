"""
D1 — Dataset EMG sintético del gemelo digital
==============================================

Usa la misma física que simulador.py (10 músculos, 10 electrodos, circuito
AD620 + pasa-banda, ADC de 12 bits a 1 kHz), pero genera cada repetición
DOS VECES con las mismas fuentes musculares y distinto sensor:

  vista A y vista B = la MISMA contracción de la MISMA persona, vista con
  otra colocación de electrodos, otra piel, otro ruido, otro 60 Hz y otro
  artefacto de movimiento.

Ese par es el "positivo físico" del aprendizaje contrastivo: obliga al
codificador a ser invariante justo a las fuentes de ruido del brazo real.

Cada ventana (200 ms) trae además etiquetas continuas para medir R²:
  pose     6 grados de libertad (5 dedos 0-1 y muñeca -1 a 1)
  fuerza   nivel de contracción efectivo (0 en reposo)
  progreso qué tanto avanzó la activación desde el inicio del gesto
"""

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import h5py
import numpy as np
from scipy import signal

from . import config  # noqa: F401  (agrega la raíz del proyecto a sys.path)
import ia_mano as ia
import simulador as sm

VERSION = "D1-1.0"
CALENTAMIENTO_MS = 100      # deja asentar los filtros
REPOSO_MS = 400             # reposo antes de cada gesto
GESTO_MS = 2500             # duración del gesto
INICIO = CALENTAMIENTO_MS + REPOSO_MS
N_VENTANAS = (GESTO_MS - ia.VENTANA) // ia.PASO + 1   # 24 ventanas por repetición
TAU = 0.06                  # constante de activación muscular (s)
NOMBRES_DOF = ["pulgar", "indice", "medio", "anular", "menique", "muneca"]
POSES_6 = np.array([sm.POSES_OBJETIVO[g][0] + [sm.POSES_OBJETIVO[g][1]] for g in ia.GESTOS])
POSE_REPOSO = POSES_6[ia.GESTOS.index("reposo")]

# progreso medio de la activación dentro de cada ventana (igual para todos los músculos)
_t_ms = np.arange(GESTO_MS)
_p = 1 - np.exp(-_t_ms / (TAU * 1000))
PROGRESO_VENTANA = np.array([_p[k * ia.PASO:k * ia.PASO + ia.VENTANA].mean() for k in range(N_VENTANAS)])

_FILTROS = {}


def _filtros_musculo(fatiga):
    clave = round(float(fatiga), 2)
    if clave not in _FILTROS:
        sos_l, norma_l = [], []
        for _, (lo, hi), _, _ in sm.MUSCULOS.values():
            k = 1 - 0.3 * clave
            sos = signal.butter(2, [lo * k, hi * k], btype="band", fs=sm.FS_INTERNA, output="sos")
            imp = signal.sosfilt(sos, np.r_[1.0, np.zeros(4000)])
            sos_l.append(sos)
            norma_l.append(np.sqrt(np.sum(imp ** 2)))
        _FILTROS[clave] = (sos_l, np.array(norma_l))
    return _FILTROS[clave]


class Persona:
    """Una persona virtual: ganancias musculares, patrón propio de cada gesto y piel."""

    def __init__(self, cond, rng):
        v = cond.variabilidad
        M = len(sm.MUSCULOS)
        self.cond, self.rng = cond, rng
        self.ganancia = np.exp(rng.normal(0, 0.35 * v, M))
        self.patron = {g: np.clip(a * np.exp(rng.normal(0, 0.25 * v, M)) + rng.normal(0, 0.03 * v, M), 0, 1)
                       for g, a in sm.ACTIVACION.items()}
        self.piel = np.exp(rng.normal(0, 0.2 * v, sm.NUM_CANALES))
        self.W = self.colocar_electrodos()

    def colocar_electrodos(self):
        d = self.cond.desplazamiento
        forma = sm.ACOPLE_BASE.shape
        return sm.ACOPLE_BASE * np.exp(self.rng.normal(0, 0.6 * d, forma)) + d * 0.15 * self.rng.random(forma)


def _fuentes(persona, gesto, intensidad, fatiga, rng):
    """Señal de cada músculo en voltios a 4 kHz: (N, 10)."""
    D = sm.DECIMACION
    N = (INICIO + GESTO_MS) * D
    t = np.arange(N) / sm.FS_INTERNA
    sos_l, normas = _filtros_musculo(fatiga)
    blanco = rng.normal(0, 1, (N, len(sos_l)))
    fuentes = np.stack([signal.sosfilt(s, blanco[:, m]) for m, s in enumerate(sos_l)], axis=1) / normas
    a0 = persona.patron["reposo"]
    a1 = np.clip(persona.patron[gesto] * (intensidad if gesto != "reposo" else 1.0), 0, 1)
    dt = np.clip(t - INICIO / 1000, 0, None)[:, None]
    activ = np.where(dt > 0, a1 + (a0 - a1) * np.exp(-dt / TAU), a0)
    fases = rng.uniform(0, 2 * np.pi, (2, len(sos_l)))
    mod = 1 + 0.08 * np.sin(2 * np.pi * 0.7 * t[:, None] + fases[0]) + 0.03 * np.sin(2 * np.pi * 10 * t[:, None] + fases[1])
    return sm.A_MVC * (1 + 0.3 * fatiga) * persona.ganancia * activ * mod * fuentes


def _sensor(s, W, piel, cond, rng):
    """Electrodos + circuito + ADC: de voltios musculares a cuentas (n, 10) a 1 kHz."""
    N = s.shape[0]
    C = W.shape[0]
    t = np.arange(N) / sm.FS_INTERNA
    v = (s @ W.T) * piel
    v += rng.normal(0, 3e-6 * cond.ruido, (N, C))
    v += cond.interferencia_60hz * rng.uniform(0.7, 1.3) * 1e-6 * \
        np.sin(2 * np.pi * 60 * t + rng.uniform(0, 2 * np.pi))[:, None] * sm.PESO_60HZ[:C]
    ta = np.clip(t - INICIO / 1000, 0, None)
    art = rng.uniform(0.05, 0.3) * 1e-3 * np.exp(-ta / 0.15) * np.sin(2 * np.pi * 4 * ta) * (ta > 0)
    v += art[:, None] * rng.uniform(0.5, 1.0, C)
    salida = np.empty((N // sm.DECIMACION, C))
    for c in range(C):
        y = np.clip(signal.sosfilt(sm.SOS_CADENA, v[:, c]), -7.5, 7.5)
        salida[:, c] = np.clip(1.65 + y[sm.DECIMACION - 1::sm.DECIMACION], 0, 3.3)
    cuentas = np.round(salida / 3.3 * 4095 + rng.normal(0, 1.5, salida.shape))
    return np.clip(cuentas, 0, 4095).astype(np.int16)


def _ventanas(senal):
    ini = INICIO + np.arange(N_VENTANAS) * ia.PASO
    return np.stack([senal[i:i + ia.VENTANA] for i in ini])


def condiciones_persona(nivel, rng):
    """Condiciones del nivel del currículo con ±15 % de variación por persona."""
    c = sm.NIVELES[nivel]
    j = lambda x: float(x * rng.uniform(0.85, 1.15))
    return replace(c, ruido=j(c.ruido), interferencia_60hz=j(c.interferencia_60hz),
                   desplazamiento=min(1.0, j(c.desplazamiento)), fatiga=min(1.0, j(c.fatiga)),
                   variabilidad=min(1.0, j(c.variabilidad)))


def simular_persona(args):
    """Genera todas las repeticiones de una persona (se corre en paralelo)."""
    id_persona, nivel, rondas, semilla = args
    rng = np.random.default_rng(semilla)
    cond = condiciones_persona(nivel, rng)
    p = Persona(cond, rng)
    piel_b = p.piel * np.exp(rng.normal(0, 0.1, sm.NUM_CANALES))
    out = {k: [] for k in ("x_a", "x_b", "gesto", "pose", "fuerza", "progreso", "intensidad",
                           "repeticion", "ventana_idx")}
    rep = 0
    for ronda in range(rondas):
        fatiga = min(1.0, cond.fatiga + 0.05 * ronda)
        for g in rng.permutation(ia.GESTOS):
            gi = ia.GESTOS.index(g)
            # 75 % fuerza normal, 25 % contracciones suaves (control proporcional)
            inten = rng.uniform(0.75, 1.25) if rng.random() < 0.75 else rng.uniform(0.35, 0.75)
            if g == "reposo":
                inten = 1.0
            s = _fuentes(p, g, inten, fatiga, rng)
            W_b = p.colocar_electrodos()          # recolocación para la vista B
            xa = _ventanas(_sensor(s, p.W, p.piel, cond, rng))
            xb = _ventanas(_sensor(s, W_b, piel_b, cond, rng))
            amp = 0.0 if g == "reposo" else min(1.0, inten)
            prog = PROGRESO_VENTANA
            pose = POSE_REPOSO + (amp * prog)[:, None] * (POSES_6[gi] - POSE_REPOSO)
            out["x_a"].append(xa)
            out["x_b"].append(xb)
            out["gesto"].append(np.full(N_VENTANAS, gi, np.int8))
            out["pose"].append(pose.astype(np.float32))
            out["fuerza"].append(((0.0 if g == "reposo" else inten) * prog).astype(np.float32))
            out["progreso"].append(prog.astype(np.float32))
            out["intensidad"].append(np.full(N_VENTANAS, inten, np.float32))
            out["repeticion"].append(np.full(N_VENTANAS, id_persona * 1000 + rep, np.int32))
            out["ventana_idx"].append(np.arange(N_VENTANAS, dtype=np.int8))
            rep += 1
    d = {k: np.concatenate(v) for k, v in out.items()}
    n = len(d["gesto"])
    d["persona"] = np.full(n, id_persona, np.int32)
    d["nivel"] = np.full(n, nivel + 1, np.int8)
    return d


def generar_d1(n_personas=160, rondas=3, semilla=2026, ruta=None, procesos=None, progreso=None):
    """160 personas × 3 rondas × 7 gestos × 24 ventanas ≈ 80.640 ventanas (dos vistas cada una)."""
    config.crear_carpetas()
    ruta = ruta or config.DATOS / "D1_emg_sintetico.h5"
    rng = np.random.default_rng(semilla)
    tareas = [(i, i % len(sm.NIVELES), rondas, int(rng.integers(1 << 31))) for i in range(n_personas)]
    procesos = procesos or max(1, (os.cpu_count() or 2) - 2)
    partes = []
    with ProcessPoolExecutor(max_workers=procesos) as ex:
        for k, d in enumerate(ex.map(simular_persona, tareas, chunksize=2)):
            partes.append(d)
            if progreso:
                progreso(k + 1, n_personas)
    d = {k: np.concatenate([p[k] for p in partes]) for k in partes[0]}
    d["caracteristicas"] = ia.caracteristicas_lote(d["x_a"]).astype(np.float32)
    d["transicion"] = d["progreso"] < 0.9
    d["split"] = dividir_por_persona(d["persona"], semilla)
    guardar_h5(d, ruta, {"version": VERSION, "semilla": semilla, "n_personas": n_personas, "rondas": rondas,
                         "gestos": ia.GESTOS, "dof": NOMBRES_DOF, "fs": ia.FS, "ventana": ia.VENTANA,
                         "paso": ia.PASO, "canales": sm.NUM_CANALES, "fuente": "gemelo digital (simulador.py)",
                         "vistas": "x_a y x_b: misma contracción, distinto sensor", "autor": config.AUTOR})
    return ruta


def dividir_por_persona(persona, semilla, fracciones=(0.70, 0.15)):
    """0 = entrenamiento, 1 = validación, 2 = prueba. Nunca se mezcla una persona."""
    ids = np.unique(persona)
    rng = np.random.default_rng(semilla + 1)
    rng.shuffle(ids)
    n_tr = int(round(len(ids) * fracciones[0]))
    n_va = int(round(len(ids) * fracciones[1]))
    grupo = {i: 0 for i in ids[:n_tr]} | {i: 1 for i in ids[n_tr:n_tr + n_va]} | {i: 2 for i in ids[n_tr + n_va:]}
    return np.array([grupo[i] for i in persona], np.int8)


def guardar_h5(d, ruta, manifiesto):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(np.ascontiguousarray(d[k]).tobytes()[:1_000_000])
    manifiesto = dict(manifiesto, n=int(len(next(iter(d.values())))), hash=h.hexdigest()[:16],
                      campos=sorted(d))
    with h5py.File(ruta, "w") as f:
        for k, v in d.items():
            f.create_dataset(k, data=v, compression="gzip" if v.nbytes > 1e6 else None, compression_opts=4
                             if v.nbytes > 1e6 else None)
        f.attrs["manifiesto"] = json.dumps(manifiesto, ensure_ascii=False)


def cargar_h5(ruta):
    with h5py.File(ruta, "r") as f:
        d = {k: f[k][()] for k in f.keys()}
        man = json.loads(f.attrs["manifiesto"])
    return d, man


def indices_contexto(ventana_idx, k=3):
    """Para cada ventana, los índices de sus k-1 ventanas anteriores de la misma repetición
    (se repite la primera cuando no hay suficientes). Devuelve (n, k), la última columna es ella."""
    i = np.arange(len(ventana_idx))
    vi = ventana_idx.astype(int)
    return np.stack([i - np.minimum(k - 1 - j, vi) for j in range(k)], axis=1)
