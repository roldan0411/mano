"""
Episodios para el visor 3D: la mano intentando los gestos en tiempo real
========================================================================

Cada época o iteración de entrenamiento produce un "intento": una secuencia
temporal (10 cuadros por segundo) con la pose que la red le da a la mano,
la pose objetivo y la señal EMG que la provocó. El visor la reproduce a
velocidad real sobre la mano InMoov; la pose objetivo va encima en
transparente. Así se ve, intento tras intento, cómo la mano aprende.
"""

import numpy as np

from . import config, metricas  # noqa: F401
from .datos_emg import NOMBRES_DOF, POSES_6
import ia_mano as ia
import simulador as sm

DT = 0.1                      # una ventana nueva cada 100 ms
VENTANAS_POR_GESTO = 14       # 1,4 s por gesto: el arranque de la contracción y el sostenido


def secuencia_prueba(d1, persona=None, semilla=0):
    """Una persona de prueba (nunca vista) haciendo los 7 gestos seguidos."""
    rng = np.random.default_rng(semilla)
    if persona is None:
        persona = int(rng.choice(np.unique(d1["persona"][d1["split"] == 2])))
    idx = []
    for g in range(ia.NUM_GESTOS):
        reps = np.unique(d1["repeticion"][(d1["persona"] == persona) & (d1["gesto"] == g)])
        i = np.flatnonzero(d1["repeticion"] == reps[0])
        idx.append(i[np.argsort(d1["ventana_idx"][i])][:VENTANAS_POR_GESTO])
    idx = np.concatenate(idx)
    x = d1["x_a"][idx].astype(np.float32) - 2048
    senal = np.concatenate([x[0]] + [w[-ia.PASO:] for w in x[1:]])     # señal continua a 1 kHz
    return {"idx": idx, "persona": persona, "objetivo": d1["pose"][idx], "gesto": d1["gesto"][idx].astype(int),
            "emg": senal[::5]}                                         # 200 Hz para dibujar


def _base(seq, titulo, seccion, dof, objetivo, dt, gesto, pred_gesto=None, metr=None):
    return {"titulo": titulo, "seccion": seccion, "dt": dt,
            "dof": np.round(np.asarray(dof, float), 4).tolist(),
            "objetivo": np.round(np.asarray(objetivo, float), 4).tolist(),
            "gesto": [int(g) for g in gesto],
            "pred_gesto": None if pred_gesto is None else [int(g) for g in pred_gesto],
            "nombres": [ia.NOMBRES[g] for g in ia.GESTOS],
            "emg": None if seq is None else {"fs": 200, "c1": np.round(seq["emg"][:, 0], 1).tolist(),
                                             "c2": np.round(seq["emg"][:, 1], 1).tolist()},
            "metricas": metr or {}}


def episodio_pose(seq, dof_pred, titulo, seccion):
    """La red predice la pose continua (sonda del contrastivo o cabeza de pose)."""
    dof = np.c_[np.clip(dof_pred[:, :5], 0, 1), np.clip(dof_pred[:, 5], -1, 1)]
    r2 = metricas.r2(seq["objetivo"], dof)["r2"]
    return _base(seq, titulo, seccion, dof, seq["objetivo"], DT, seq["gesto"], metr={"R² pose": round(r2, 3)})


def _firmware(probs):
    """Lo mismo que protesis_tiempo_real.ino: umbral de confianza + mayoría de 3."""
    hist, ultima, salida = [-1, -1, -1], 0, []
    for p in probs:
        k = int(np.argmax(p))
        if p[k] >= ia.UMBRAL_CONFIANZA:
            ultima = k
        hist = hist[1:] + [ultima]
        if hist[0] >= 0 and (hist[0] == hist[1] or hist[0] == hist[2]):
            salida.append(hist[0])
        elif hist[1] >= 0 and hist[1] == hist[2]:
            salida.append(hist[1])
        else:
            salida.append(ultima)
    return np.array(salida)


def episodio_firmware(seq, probs, titulo, seccion, poses=None, params=None):
    """La red clasifica el gesto y la mano física virtual (servos con su velocidad real,
    tendones con holgura) va a la pose de ese gesto, como haría el ESP32."""
    poses = poses or ia.POSES_BASE
    pred = _firmware(probs)
    mano = sm.ManoFisica(params)
    mano.servo_dedos = np.array(poses["reposo"][0], float)
    logradas = []
    for g in pred:
        d, m = poses[ia.GESTOS[g]]
        for _ in range(10):                                     # 10 subpasos de 10 ms
            dd, mm = mano.paso(DT / 10, np.array(d, float), m)
        logradas.append(np.r_[dd, mm])
    logradas = np.array(logradas)
    acierto = float(np.mean(pred == seq["gesto"]))
    r2 = metricas.r2(seq["objetivo"], logradas)["r2"]
    return _base(seq, titulo, seccion, logradas, seq["objetivo"], DT, seq["gesto"], pred,
                 {"acierto": round(acierto, 3), "R² pose": round(r2, 3)})


def episodio_refuerzo(agente, titulo, seccion, rng, segundos=1.0, dt=0.05):
    """El agente intenta cada gesto CON su ruido de exploración actual (lo que
    está probando en esta iteración), sobre la mano física virtual."""
    mu = agente.poses()
    sigma = np.exp(agente.log_sigma.detach().cpu().numpy())
    mano = sm.ManoFisica(agente.mano.p)
    dof, obj, gesto = [], [], []
    for gi, g in enumerate(ia.GESTOS):
        d, m = mu[g]
        a = np.r_[d, m] + sigma * rng.normal(0, 1, 6)
        cmd_d, cmd_m = np.clip(a[:5], 0, 1), float(np.clip(a[5], -1, 1))
        for _ in range(int(segundos / dt)):
            dd, mm = mano.paso(dt, cmd_d, cmd_m)
            dof.append(np.r_[dd, mm])
            obj.append(sm.POSES_OBJETIVO[g][0] + [sm.POSES_OBJETIVO[g][1]])
            gesto.append(gi)
    dof, obj = np.array(dof), np.array(obj)
    fin = np.arange(int(segundos / dt) - 1, len(dof), int(segundos / dt))
    r2 = metricas.r2(obj[fin], dof[fin])["r2"]
    return _base(None, titulo, seccion, dof, obj, dt, gesto, metr={"R² pose": round(r2, 3),
                                                                   "exploración σ": round(float(sigma.mean()), 3)})


def episodio_formas(dof_real, dof_pred, gestos, titulo, seccion, cuadros=7):
    """Red de formas: de 21 puntos a 6 GDL. Cada pose se sostiene 0,7 s."""
    rep = lambda a: np.repeat(np.asarray(a), cuadros, axis=0)
    r2 = metricas.r2(dof_real, dof_pred)["r2"]
    return _base(None, titulo, seccion, rep(dof_pred), rep(dof_real), DT, rep(gestos), metr={"R² 6 GDL": round(r2, 3)})
