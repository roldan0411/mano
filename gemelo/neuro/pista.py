"""
La pista de la neuroevolución: una persona del gemelo haciendo gestos
=====================================================================

Como el circuito de los carros, pero para una prótesis: una persona virtual
se pone los electrodos, se calibra (20 s) y hace 21 gestos al azar (cada uno
3 veces, con fuerza variable). Las manos solo "escuchan" los 10 electrodos
(120 entradas: 40 características calibradas × 3 ventanas) y tienen que hacer
lo que la persona quiere, ventana a ventana (cada 100 ms).

Una mano CHOCA cuando hace el gesto equivocado durante medio segundo seguido
(5 ventanas). Las primeras ventanas de cada gesto (transición, el músculo
todavía no llegó) no cuentan ni a favor ni en contra.
"""

from dataclasses import replace

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


def recorrido(semilla, senal=False, ajustes=None, crudo=False):
    """Una persona y su pista. ajustes (para las pruebas de robustez):
         nivel      nivel de dificultad fijo (0 a 4)
         ruido      multiplica el ruido de los electrodos
         fatiga     fatiga fija (0 a 1)
         giro_antes_deg  los electrodos quedaron girados este ángulo ANTES de calibrar
         giro_deg   los 10 electrodos giran juntos este ángulo DESPUÉS de calibrar
         axial_cm   y se corren esto a lo largo del brazo
         apagados   electrodos (índices) que se despegan después de calibrar
       crudo=True devuelve además las ventanas crudas de la pista y de la calibración."""
    a = ajustes or {}
    rng = np.random.default_rng(semilla)
    amp = gemelo2.Amplificador()
    nivel = int(rng.integers(0, len(sm.NIVELES)))
    nivel = a.get("nivel", nivel)
    cond = de.condiciones_persona(nivel, rng)
    if "ruido" in a:
        cond = replace(cond, ruido=cond.ruido * a["ruido"])
    if "fatiga" in a:
        cond = replace(cond, fatiga=float(a["fatiga"]))
    p = gemelo2.Persona(cond, rng)
    ses = p.colocar(rng)
    if a.get("giro_antes_deg"):                                   # se pusieron girados, pero calibró así
        ses = dict(ses, W=sm.acople(ses["angulos"] + a["giro_antes_deg"], ses["axial_cm"], p.dang))
    desfase = rng.normal(0, amp.desfase_mv_sd * 1e-3, sm.NUM_CANALES)
    F_cal, x_cal = [], []
    for g in ia.GESTOS:                                           # calibración de 20 s al ponerse los electrodos
        cuentas, _ = gemelo2.sensor_por_etapas(de._fuentes(p, g, 1.0, cond.fatiga, rng), ses, cond, amp, rng, desfase)
        x_cal.append(de._ventanas(cuentas))
        F_cal.append(ia.caracteristicas_lote(x_cal[-1]))
    cal = gemelo2.calibracion(np.concatenate(F_cal))
    if a.get("giro_deg") or a.get("axial_cm"):                     # se corren después de calibrar
        ses = dict(ses, W=sm.acople(ses["angulos"] + a.get("giro_deg", 0.0), ses["axial_cm"] + a.get("axial_cm", 0.0), p.dang))
    apagados = list(a.get("apagados", ()))
    F, gesto, pose, vidx, adc, x_pista = [], [], [], [], [], []
    for b, gi in enumerate(_orden(rng)):
        g = ia.GESTOS[gi]
        inten = gemelo2._intensidad(g, rng, 1)
        fatiga = min(1.0, cond.fatiga + 0.01 * b)
        r = gemelo2.sensor_por_etapas(de._fuentes(p, g, inten, fatiga, rng), ses, cond, amp, rng, desfase, etapas=senal)
        cuentas = r[0]
        if apagados:                                              # electrodo suelto: el ADC queda en la mitad, con su ruido
            cuentas = cuentas.copy()
            cuentas[:, apagados] = np.clip(np.round(2048 + rng.normal(0, 1.5, (len(cuentas), len(apagados)))), 0, 4095)
        x = de._ventanas(cuentas)[:VENTANAS]
        x_pista.append(x)
        F.append(ia.caracteristicas_lote(x))
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
    if crudo:
        d.update(x=np.concatenate(x_pista), x_cal=np.concatenate(x_cal), ventana_idx=vidx, cal=cal)
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
