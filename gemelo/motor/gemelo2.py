"""
Gemelo v2: antebrazo con geometría, amplificador por etapas y calibración
=========================================================================

Cambios frente a simulador.py / datos_emg.py (la física muscular es la misma):

1. GEOMETRÍA. Cada músculo tiene un ángulo y una profundidad en el corte del
   antebrazo; cada par de electrodos, un ángulo y un corrimiento a lo largo
   del brazo. El acople músculo → electrodo sale de la distancia, así que
   "recolocar electrodos" es moverlos de verdad (y se puede dibujar).
2. AMPLIFICADOR POR ETAPAS, como el circuito real:
     electrodo (µV) → AD620 con G = 1 + 49,4 kΩ / RG → pasa-banda 16–482 Hz ×10
     → desplazamiento a 1,65 V → ADC de 12 bits.
   Incluye el desfase continuo entre electrodos (media celda), que el AD620
   amplifica ANTES del filtro y puede saturarlo.
3. CALIBRACIÓN POR SESIÓN. Al ponerse los electrodos, la persona hace cada
   gesto una vez (ronda 0). Las características se expresan relativas a esa
   calibración: log(1 + c) − media de la calibración.
4. Se mantienen las PERSONAS: cada persona tiene su anatomía, ganancias y patrón.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

from .. import config  # noqa: F401  (agrega la raíz del proyecto a sys.path)
from .. import datos_emg as de
import ia_mano as ia
import simulador as sm

MUSCULOS = de.sm.NOMBRES_MUSCULOS
#                 ángulo (°, 90 = dorso, 270 = palma)   profundidad (1 = bajo la piel)
GEOMETRIA = {"FDS": (270, 0.75), "FDP": (250, 0.45), "FPL": (300, 0.50), "FCR": (315, 0.85), "FCU": (225, 0.85),
             "EDC": (90, 0.80), "EIP": (110, 0.55), "EPL": (70, 0.50), "ECR": (45, 0.85), "ECU": (135, 0.85)}
ANG = np.array([GEOMETRIA[m][0] for m in MUSCULOS], float)
PROF = np.array([GEOMETRIA[m][1] for m in MUSCULOS], float)
ELECTRODOS_NOMINAL = np.array([270.0, 90.0])          # canal 1 sobre flexores, canal 2 sobre extensores
SIGMA_ANG, LAMBDA_PROF, PISO_CONDUCCION = 38.0, 0.35, 0.05
N_GESTOS = ia.NUM_GESTOS


@dataclass
class Amplificador:
    rg_ohm: float = 470.0           # resistencia de ganancia del AD620
    desfase_mv_sd: float = 12.0     # desfase continuo entre electrodos (desv. estándar por sesión)
    riel_ad620_v: float = 6.1       # salida máxima del AD620 con ±7,5 V
    ganancia_filtro: float = 10.0   # 100 kΩ / 10 kΩ

    @property
    def g_ad620(self):
        return 1 + 49.4e3 / self.rg_ohm


def _etapa_filtro_sos():
    """Pasa-banda de la segunda etapa (sin la ganancia del AD620), discretizado a 4 kHz."""
    R1, C1, Rf, Cf = 10e3, 1e-6, 100e3, 3.3e-9
    Rth, C2 = 51e3, 1e-6
    num = np.polymul([-Rf * C1, 0], [Rth * C2, 0])
    den = np.polymul(np.polymul([Rf * Cf, 1], [R1 * C1, 1]), [Rth * C2, 1])
    b, a = signal.bilinear(num, den, fs=sm.FS_INTERNA)
    return signal.tf2sos(b, a)


SOS_FILTRO = _etapa_filtro_sos()


def acople(ang_electrodos, axial_cm, persona_dang=None):
    """Matriz 2×10 de acople músculo → electrodo a partir de la geometría."""
    ang = ANG + (0 if persona_dang is None else persona_dang)
    d = (ang[None, :] - np.asarray(ang_electrodos)[:, None] + 180) % 360 - 180
    w = np.exp(-0.5 * (d / SIGMA_ANG) ** 2) * np.exp(-(1 - PROF[None, :]) / LAMBDA_PROF)
    w = w + PISO_CONDUCCION * np.exp(-(1 - PROF[None, :]) / LAMBDA_PROF)
    w = w / w.max(axis=1, keepdims=True)
    return w * np.exp(-np.abs(np.asarray(axial_cm))[:, None] / 3.0)


class Persona(de.Persona):
    """Persona del gemelo v1 + anatomía propia (los músculos no están en el mismo ángulo en todos)."""

    def __init__(self, cond, rng):
        super().__init__(cond, rng)
        self.dang = rng.normal(0, 8 * cond.variabilidad + 2, len(MUSCULOS))

    def colocar(self, rng):
        """Una sesión: dónde quedan los electrodos, su desfase continuo y la piel de hoy."""
        d = self.cond.desplazamiento
        ang = ELECTRODOS_NOMINAL + rng.normal(0, 30 * d + 3, 2)
        axial = rng.normal(0, 1.5 * d + 0.2, 2)
        return {"angulos": ang, "axial_cm": axial, "W": acople(ang, axial, self.dang),
                "piel": self.piel * np.exp(rng.normal(0, 0.1, 2))}


def sensor_por_etapas(s, ses, cond, amp, rng, desfase_v, etapas=False):
    """Fuentes musculares (N, 10) en V a 4 kHz → cuentas del ADC (n, 2) a 1 kHz.
    Con etapas=True devuelve además la señal en cada etapa (a 1 kHz)."""
    N = s.shape[0]
    t = np.arange(N) / sm.FS_INTERNA
    v = (s @ ses["W"].T) * ses["piel"]
    v += rng.normal(0, 3e-6 * cond.ruido, (N, 2))
    v += cond.interferencia_60hz * 1e-6 * np.sin(2 * np.pi * 60 * t + rng.uniform(0, 2 * np.pi))[:, None] * np.array([1.0, 0.8])
    ta = np.clip(t - de.INICIO / 1000, 0, None)
    v += (rng.uniform(0.05, 0.3) * 1e-3 * np.exp(-ta / 0.15) * np.sin(2 * np.pi * 4 * ta) * (ta > 0))[:, None] * np.array([1.0, 0.7])
    v_elec = v + desfase_v                                        # desfase continuo de los electrodos
    v_ad = np.clip(amp.g_ad620 * v_elec, -amp.riel_ad620_v, amp.riel_ad620_v)
    v_fil = np.empty_like(v_ad)
    for c in range(2):
        zi = signal.sosfilt_zi(SOS_FILTRO) * 0                    # arranca en reposo
        v_fil[:, c] = np.clip(signal.sosfilt(SOS_FILTRO, v_ad[:, c] - v_ad[0, c], zi=zi)[0], -7.5, 7.5)
    D = sm.DECIMACION
    vadc = np.clip(1.65 + v_fil[D - 1::D], 0, 3.3)
    cuentas = np.clip(np.round(vadc / 3.3 * 4095 + rng.normal(0, 1.5, vadc.shape)), 0, 4095).astype(np.int16)
    saturo = bool(np.any(np.abs(v_ad) >= amp.riel_ad620_v * 0.999))
    if not etapas:
        return cuentas, saturo
    return cuentas, saturo, {"electrodo_uv": (v_elec[D - 1::D] - desfase_v) * 1e6, "ad620_v": v_ad[D - 1::D],
                             "filtro_v": v_fil[D - 1::D], "adc": cuentas}


def calibracion(F_cal):
    """Media de log(1 + característica) en la ronda de calibración de una sesión."""
    return np.log1p(F_cal).mean(axis=0).astype(np.float32)


def calibrar(F, cal):
    return (np.log1p(F) - cal).astype(np.float32)


def contexto(Fc, ventana_idx, k=3):
    ctx = de.indices_contexto(ventana_idx, k)
    return Fc[ctx].reshape(len(Fc), -1)


def _intensidad(g, rng, ronda):
    if g == "reposo" or ronda == 0:
        return 1.0
    return rng.uniform(0.75, 1.25) if rng.random() < 0.75 else rng.uniform(0.35, 0.75)


def simular_persona(args):
    """Una persona, varias sesiones (cada una con su calibración). Corre en procesos aparte."""
    pid, nivel, sesiones, rondas, semilla, amp_dict = args
    rng = np.random.default_rng(semilla)
    amp = Amplificador(**amp_dict)
    cond = de.condiciones_persona(nivel, rng)
    p = Persona(cond, rng)
    out = {k: [] for k in ("x", "F", "gesto", "pose", "fuerza", "ventana_idx", "ronda", "sesion", "repeticion")}
    cals, sat, rep = [], 0, 0
    for s_i in range(sesiones):
        ses = p.colocar(rng)
        desfase = rng.normal(0, amp.desfase_mv_sd * 1e-3, 2)
        sid = pid * 100 + s_i
        F_ses, saturada = [], False
        for ronda in range(rondas):
            fatiga = min(1.0, cond.fatiga + 0.05 * ronda)
            for g in (ia.GESTOS if ronda == 0 else rng.permutation(ia.GESTOS)):
                gi = ia.GESTOS.index(g)
                inten = _intensidad(g, rng, ronda)
                cuentas, sat_g = sensor_por_etapas(de._fuentes(p, g, inten, fatiga, rng), ses, cond, amp, rng, desfase)
                saturada |= sat_g
                x = de._ventanas(cuentas)
                F = ia.caracteristicas_lote(x).astype(np.float32)
                amp_pose = 0.0 if g == "reposo" else min(1.0, inten)
                pose = de.POSE_REPOSO + (amp_pose * de.PROGRESO_VENTANA)[:, None] * (de.POSES_6[gi] - de.POSE_REPOSO)
                n = len(x)
                out["x"].append(x)
                out["F"].append(F)
                out["gesto"].append(np.full(n, gi, np.int8))
                out["pose"].append(pose.astype(np.float32))
                out["fuerza"].append(((0.0 if g == "reposo" else inten) * de.PROGRESO_VENTANA).astype(np.float32))
                out["ventana_idx"].append(np.arange(n, dtype=np.int8))
                out["ronda"].append(np.full(n, ronda, np.int8))
                out["sesion"].append(np.full(n, sid, np.int32))
                out["repeticion"].append(np.full(n, sid * 100 + rep, np.int32))
                if ronda == 0:
                    F_ses.append(F)
                rep += 1
        cals.append((sid, calibracion(np.concatenate(F_ses))))
        sat += saturada
    d = {k: np.concatenate(v) for k, v in out.items()}
    d["persona"] = np.full(len(d["gesto"]), pid, np.int32)
    d["nivel"] = np.full(len(d["gesto"]), nivel + 1, np.int8)
    return d, cals, sat


def unir_lote(resultados):
    """Junta varias personas en un lote y agrega las características calibradas y con contexto."""
    d = {k: np.concatenate([r[0][k] for r in resultados]) for k in resultados[0][0]}
    cal = {sid: c for r in resultados for sid, c in r[1]}
    cal_v = np.stack([cal[s] for s in d["sesion"]])
    d["Fc"] = calibrar(d["F"], cal_v)
    d["X"] = contexto(d["Fc"], d["ventana_idx"]).astype(np.float32)
    d["transicion"] = de.PROGRESO_VENTANA[d["ventana_idx"].astype(int)] < 0.9
    return d, cal, sum(r[2] for r in resultados)


# ------------------------------------------------------------------
# Demostración para la aplicación: una persona, su calibración y los 7 gestos,
# con la señal en cada etapa del amplificador y la geometría de los electrodos
# ------------------------------------------------------------------
VENTANAS_DEMO = 14


def demostracion(semilla, amp=None, nivel=None):
    rng = np.random.default_rng(semilla)
    amp = amp or Amplificador()
    nivel = int(rng.integers(0, len(sm.NIVELES))) if nivel is None else nivel
    cond = de.condiciones_persona(nivel, rng)
    p = Persona(cond, rng)
    ses = p.colocar(rng)
    desfase = rng.normal(0, amp.desfase_mv_sd * 1e-3, 2)
    F_cal = []
    for g in ia.GESTOS:                                           # calibración de 20 s al encender
        cuentas, _ = sensor_por_etapas(de._fuentes(p, g, 1.0, cond.fatiga, rng), ses, cond, amp, rng, desfase)
        F_cal.append(ia.caracteristicas_lote(de._ventanas(cuentas)))
    cal = calibracion(np.concatenate(F_cal))
    etapas = {k: [] for k in ("electrodo_uv", "ad620_v", "filtro_v", "adc")}
    F, gesto, pose, vidx, activ, saturo = [], [], [], [], [], False
    for gi, g in enumerate(ia.GESTOS):
        inten = 1.0
        s = de._fuentes(p, g, inten, cond.fatiga, rng)
        cuentas, sat, et = sensor_por_etapas(s, ses, cond, amp, rng, desfase, etapas=True)
        saturo |= sat
        a, b = de.INICIO, de.INICIO + ia.PASO * VENTANAS_DEMO
        for k in etapas:
            etapas[k].append(et[k][a:b])
        x = de._ventanas(cuentas)[:VENTANAS_DEMO]
        F.append(ia.caracteristicas_lote(x))
        gesto += [gi] * VENTANAS_DEMO
        amp_pose = 0.0 if g == "reposo" else 1.0
        pose.append(de.POSE_REPOSO + (amp_pose * de.PROGRESO_VENTANA[:VENTANAS_DEMO])[:, None] * (de.POSES_6[gi] - de.POSE_REPOSO))
        vidx.append(np.arange(VENTANAS_DEMO))
        # activación muscular media por ventana (10 músculos), para dibujar el antebrazo
        A = np.abs(s[de.INICIO * sm.DECIMACION:]) / (sm.A_MVC * (1 + 0.3 * cond.fatiga) * p.ganancia + 1e-12)
        D = ia.PASO * sm.DECIMACION
        activ.append(np.stack([A[k * D:(k + 2) * D].mean(0) for k in range(VENTANAS_DEMO)]))
    F = np.concatenate(F)
    Fc = calibrar(F, cal)
    X = contexto(Fc, np.concatenate(vidx).astype(np.int8))
    dec = 4                                                       # 250 Hz para dibujar
    senal = {k: np.round(np.concatenate(v)[::dec], 3 if k != "adc" else 0).tolist() for k, v in etapas.items()}
    act = np.concatenate(activ)
    act = np.clip(act / max(np.percentile(act, 99), 1e-9), 0, 1)
    return {"X": X.astype(np.float32), "F": F.astype(np.float32), "Fc": Fc, "cal": cal,
            "gesto": np.array(gesto), "pose": np.concatenate(pose).astype(np.float32),
            "vista": {"dt": ia.PASO / 1000, "fs_senal": 1000 / dec, "senal": senal,
                      "activacion": np.round(act, 3).tolist(), "musculos": MUSCULOS,
                      "geometria": {"angulos": ANG.tolist(), "profundidad": PROF.tolist(),
                                    "dang_persona": np.round(p.dang, 1).tolist(),
                                    "electrodos_nominal": ELECTRODOS_NOMINAL.tolist(),
                                    "electrodos": np.round(ses["angulos"], 1).tolist(),
                                    "axial_cm": np.round(ses["axial_cm"], 2).tolist(),
                                    "acople": np.round(ses["W"], 3).tolist()},
                      "persona": {"nivel": nivel + 1, **{k: round(float(v), 3) for k, v in asdict(cond).items()}},
                      "amplificador": {**asdict(amp), "g_ad620": round(amp.g_ad620, 1),
                                       "g_total": round(amp.g_ad620 * amp.ganancia_filtro, 0),
                                       "desfase_mv": np.round(desfase * 1e3, 2).tolist(), "saturo": saturo},
                      "calibracion": np.round(cal, 3).tolist(),
                      "caracteristicas": np.round(F, 1).tolist()}}
