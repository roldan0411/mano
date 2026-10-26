"""
La pista de la neuroevolución: una persona del gemelo haciendo gestos
=====================================================================

Como el circuito de los carros, pero para una prótesis: una persona virtual
se pone los electrodos, se calibra (20 s) y hace 21 gestos al azar (cada uno
3 veces, con fuerza variable). Las manos solo "escuchan" los 2 canales EMG
(24 entradas: 8 características calibradas × 3 ventanas) y tienen que hacer
lo que la persona quiere, ventana a ventana (cada 100 ms).

Una mano CHOCA cuando hace el gesto equivocado durante medio segundo seguido
(5 ventanas). Las primeras ventanas de cada gesto (transición, el músculo
todavía no llegó) no cuentan ni a favor ni en contra.
"""

import numpy as np

import ia_mano as ia
import simulador as sm
from .. import datos_emg as de
from ..motor import gemelo2

VENTANAS = 14               # 1,4 s por gesto
REPETICIONES = 3            # 7 gestos × 3 = 21 gestos: ~29 s de pista
CHOQUE = 5                  # ventanas seguidas equivocada = choca (0,5 s)
DT = ia.PASO / 1000


def _orden(rng):
    """Cada gesto 3 veces, al azar, sin repetir el mismo dos veces seguidas."""
    while True:
        o = rng.permutation(np.repeat(np.arange(ia.NUM_GESTOS), REPETICIONES))
        if np.all(o[1:] != o[:-1]):
            return o


def recorrido(semilla, senal=False):
    rng = np.random.default_rng(semilla)
    amp = gemelo2.Amplificador()
    nivel = int(rng.integers(0, len(sm.NIVELES)))
    cond = de.condiciones_persona(nivel, rng)
    p = gemelo2.Persona(cond, rng)
    ses = p.colocar(rng)
    desfase = rng.normal(0, amp.desfase_mv_sd * 1e-3, 2)
    F_cal = []
    for g in ia.GESTOS:                                           # calibración de 20 s al ponerse los electrodos
        cuentas, _ = gemelo2.sensor_por_etapas(de._fuentes(p, g, 1.0, cond.fatiga, rng), ses, cond, amp, rng, desfase)
        F_cal.append(ia.caracteristicas_lote(de._ventanas(cuentas)))
    cal = gemelo2.calibracion(np.concatenate(F_cal))
    F, gesto, pose, vidx, adc = [], [], [], [], []
    for b, gi in enumerate(_orden(rng)):
        g = ia.GESTOS[gi]
        inten = gemelo2._intensidad(g, rng, 1)
        fatiga = min(1.0, cond.fatiga + 0.01 * b)
        r = gemelo2.sensor_por_etapas(de._fuentes(p, g, inten, fatiga, rng), ses, cond, amp, rng, desfase, etapas=senal)
        F.append(ia.caracteristicas_lote(de._ventanas(r[0])[:VENTANAS]))
        gesto += [gi] * VENTANAS
        amp_pose = 0.0 if g == "reposo" else min(1.0, inten)
        pose.append(de.POSE_REPOSO + (amp_pose * de.PROGRESO_VENTANA[:VENTANAS])[:, None] * (de.POSES_6[gi] - de.POSE_REPOSO))
        vidx.append(np.arange(VENTANAS))
        if senal:
            adc.append(r[2]["adc"][de.INICIO:de.INICIO + ia.PASO * VENTANAS:4])    # 250 Hz para dibujar
    vidx = np.concatenate(vidx)
    X = gemelo2.contexto(gemelo2.calibrar(np.concatenate(F), cal), vidx.astype(np.int8))
    d = {"X": X.astype(np.float32), "gesto": np.array(gesto, np.int8), "pose": np.concatenate(pose).astype(np.float32),
         "estable": de.PROGRESO_VENTANA[vidx] >= 0.9, "nivel": nivel + 1, "semilla": semilla}
    if senal:
        d["adc"] = np.concatenate(adc).astype(np.float32)
    return d


def choques(pred, pista):
    """pred: (G, T) gesto que hace cada mano -> ventana en la que choca cada una (T si termina la pista)."""
    G, T = pred.shape
    error = (pred != pista["gesto"][None]) & pista["estable"][None]
    seguidas, choque = np.zeros(G, int), np.full(G, T)
    for t in range(T):
        if pista["estable"][t]:
            seguidas = np.where(error[:, t], seguidas + 1, 0)
            nuevas = (seguidas >= CHOQUE) & (choque == T)
            choque[nuevas] = t + 1
    return choque
