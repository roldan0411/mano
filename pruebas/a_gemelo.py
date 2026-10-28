"""
Pruebas A · El gemelo digital es fiel (modelo de antebrazo, electrodos y adquisición)
===================================================================================
  A1 acople músculo-electrodo     A2 ganancia y saturación del AD620
  A3 respuesta en frecuencia      A4 espectro de la señal simulada
  A5 diafonía entre electrodos    A6 simulado frente a EMG real (UCI)
"""

from comun import (ACENTO, AZULES, COLORES_CANAL, GRIS, NARANJA, ROJO, VERDES, Prueba, figura, plt)

import glob
from pathlib import Path
from dataclasses import asdict

import numpy as np
from scipy import signal, stats

import ia_mano as ia
import simulador as sm
from gemelo import datos_emg as de
from gemelo.motor import gemelo2

FLEX, EXT = list(range(5)), list(range(5, 10))
MUS = sm.NOMBRES_MUSCULOS


def sesiones(n=12, gestos=("abierta", "puno", "pinza", "senalar", "flexion", "extension"), nivel=1, semilla=10, amp=None):
    """Señal de los electrodos (µV, 1 kHz) durante cada gesto, para n personas."""
    amp = amp or gemelo2.Amplificador()
    out = []
    for k in range(n):
        rng = np.random.default_rng(semilla + k)
        cond = de.condiciones_persona(nivel, rng)
        p = gemelo2.Persona(cond, rng)
        ses = p.colocar(rng)
        desfase = rng.normal(0, amp.desfase_mv_sd * 1e-3, sm.NUM_CANALES)
        for g in gestos:
            _, sat, et = gemelo2.sensor_por_etapas(de._fuentes(p, g, 1.0, cond.fatiga, rng), ses, cond, amp, rng, desfase, etapas=True)
            out.append({"persona": k, "gesto": g, "uv": et["electrodo_uv"], "ad620": et["ad620_v"], "adc": et["adc"], "sat": sat})
    return out


def a1():
    with Prueba("A1", "Acople músculo–electrodo", "A. Gemelo digital",
                "Verificar que cada uno de los 10 electrodos capte sobre todo los músculos que tiene debajo, según la geometría del antebrazo.",
                ["RF-01", "RF-02"],
                "Se calcula la matriz de acople W (10 electrodos × 10 músculos) en la posición nominal: "
                "w = exp(−Δθ²/2σ²)·exp(−(1−p)/λ) + piso, normalizada por electrodo. Para cada electrodo se identifica el músculo "
                "de mayor acople y se compara con el músculo superficial (p ≥ 0,75) angularmente más cercano. Se mide además cuánto "
                "cambia W al girar todos los electrodos 15°.",
                "Los 10 electrodos captan como máximo un músculo de su propio grupo (flexor o extensor) y ese músculo es el superficial más cercano.") as p:
        W = sm.ACOPLE_BASE
        sup = np.where(sm.PROF >= 0.75)[0]
        filas, ok = [], 0
        for c in range(10):
            m = int(np.argmax(W[c]))
            d = np.abs((sm.ANG[sup] - sm.ELECTRODOS_NOMINAL[c] + 180) % 360 - 180)
            cercano = int(sup[np.argmin(d)])
            grupo_ok = (m < 5) == (c < 5)
            ok += grupo_ok and m == cercano
            filas.append({"electrodo": ia.CANALES[c], "angulo": sm.ELECTRODOS_NOMINAL[c], "musculo_max": MUS[m], "acople_max": W[c, m],
                          "superficial_mas_cercano": MUS[cercano], "grupo_correcto": grupo_ok,
                          "propio_vs_opuesto": W[c, :5].sum() / W[c, 5:].sum() if c < 5 else W[c, 5:].sum() / W[c, :5].sum()})
        W15 = sm.acople(sm.ELECTRODOS_NOMINAL + 15)
        p.resultado = {"electrodos": filas, "aciertos": ok, "cambio_W_giro15_pct": float(np.abs(W15 - W).sum() / np.abs(W).sum() * 100),
                       "sigma_ang_deg": sm.SIGMA_ANG, "lambda_prof": sm.LAMBDA_PROF, "piso": sm.PISO_CONDUCCION}
        p.cumple = ok == 10
        p.conclusion = (f"{ok} de 10 electrodos captan como máximo el músculo superficial más cercano de su grupo; "
                        f"girar los electrodos 15° cambia el {p.resultado['cambio_W_giro15_pct']:.0f} % del acople.")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})
        im = ax[0].imshow(W, cmap="Blues", vmin=0, vmax=1)
        ax[0].set_xticks(range(10), MUS, rotation=45)
        ax[0].set_yticks(range(10), ia.CANALES)
        ax[0].grid(False)
        for c in range(10):
            for m in range(10):
                ax[0].text(m, c, f"{W[c, m]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if W[c, m] > 0.55 else "#333")
        ax[0].set_title("Matriz de acople W (electrodo × músculo)")
        plt.colorbar(im, ax=ax[0], fraction=0.04)
        a = plt.subplot(1, 2, 2, projection="polar")
        ax[1].remove()
        a.set_theta_zero_location("E")
        for k, m in enumerate(MUS):
            a.scatter(np.radians(sm.ANG[k]), sm.PROF[k], s=260, color="#e9a3a0" if k < 5 else "#f2c48a", edgecolor="#555", zorder=3)
            a.text(np.radians(sm.ANG[k]), sm.PROF[k], m, ha="center", va="center", fontsize=6.5, zorder=4)
        for c in range(10):
            a.scatter(np.radians(sm.ELECTRODOS_NOMINAL[c]), 1.12, s=70, color=COLORES_CANAL[c], edgecolor="#333", zorder=3)
            a.text(np.radians(sm.ELECTRODOS_NOMINAL[c]), 1.3, ia.CANALES[c], ha="center", va="center", fontsize=7.5)
        a.set_ylim(0, 1.38)
        a.set_yticklabels([])
        a.set_title("Corte del antebrazo: músculos (radio = profundidad)\ny electrodos (90° = dorso, 270° = palma)", fontsize=9.5)
        p.figuras.append(figura("A1", "acople", fig))


def a2():
    with Prueba("A2", "Ganancia y saturación del AD620", "A. Gemelo digital",
                "Verificar que la etapa del amplificador de instrumentación del gemelo cumpla G = 1 + 49,4 kΩ/RG y que modele la saturación por el desfase de los electrodos.",
                ["RF-03"],
                "Para 9 valores de RG entre 100 Ω y 10 kΩ se simula una sesión y se mide la ganancia como la pendiente entre la salida "
                "del AD620 y la señal del electrodo (regresión lineal en las muestras no saturadas). Se calcula además el desfase "
                "máximo que tolera cada ganancia antes de llegar al riel (6,1 V).",
                "Error de ganancia menor a 1 % para todos los RG; la saturación aparece cuando G·desfase supera el riel.") as p:
        filas = []
        rgs = np.logspace(2, 4, 9)
        for rg in rgs:
            amp = gemelo2.Amplificador(rg_ohm=float(rg), desfase_mv_sd=2.0)
            s = sesiones(1, ("puno",), semilla=3, amp=amp)[0]
            v = s["uv"] * 1e-6
            y = s["ad620"]
            libre = np.abs(y) < amp.riel_ad620_v * 0.98
            pend = np.polyfit(v[libre].ravel(), (y - np.median(y, 0))[libre].ravel(), 1)[0] if libre.sum() > 100 else np.nan
            filas.append({"rg_ohm": rg, "g_formula": amp.g_ad620, "g_medida": pend, "error_pct": abs(pend / amp.g_ad620 - 1) * 100,
                          "desfase_max_mv": amp.riel_ad620_v / amp.g_ad620 * 1e3})
        err = max(f["error_pct"] for f in filas)
        sat = sesiones(1, ("puno",), semilla=3, amp=gemelo2.Amplificador(rg_ohm=100.0, desfase_mv_sd=40.0))[0]["sat"]
        p.resultado = {"tabla": filas, "error_max_pct": err, "satura_con_rg100_desfase40mv": sat}
        p.cumple = bool(err < 1.0 and sat)
        p.conclusion = f"Error máximo de ganancia {err:.3f} %; con RG = 100 Ω y desfase de ±40 mV el AD620 satura, como predice G·desfase > 6,1 V."
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].loglog(rgs, [f["g_formula"] for f in filas], color=GRIS, lw=2, label="fórmula 1 + 49,4 kΩ/RG")
        ax[0].loglog(rgs, [f["g_medida"] for f in filas], "o", color=ACENTO, label="medida en el gemelo")
        ax[0].set_xlabel("RG (Ω)")
        ax[0].set_ylabel("ganancia G")
        ax[0].set_title("Ganancia del AD620")
        ax[0].legend()
        rg = np.logspace(2, 4, 100)
        g = 1 + 49.4e3 / rg
        ax[1].fill_between(rg, 6.1 / g * 1e3, 200, color=ROJO, alpha=0.15, label="satura")
        ax[1].semilogx(rg, 6.1 / g * 1e3, color=ROJO)
        ax[1].axhline(12, color=GRIS, ls="--", label="desfase típico (±12 mV, 1 σ)")
        ax[1].axvline(470, color=ACENTO, ls=":", label="RG del proyecto (470 Ω)")
        ax[1].set_ylim(0, 120)
        ax[1].set_xlabel("RG (Ω)")
        ax[1].set_ylabel("desfase máximo (mV)")
        ax[1].set_title("Desfase de electrodos que tolera sin saturar")
        ax[1].legend(fontsize=7.5)
        p.figuras.append(figura("A2", "ad620", fig))


def a3():
    with Prueba("A3", "Respuesta en frecuencia del filtro", "A. Gemelo digital",
                "Verificar que el pasa-banda de la segunda etapa (R1 = 10 kΩ, C1 = 1 µF, Rf = 100 kΩ, Cf = 3,3 nF) tenga los cortes de diseño.",
                ["RF-03"],
                "Se obtiene la respuesta en frecuencia del filtro analógico (función de transferencia) y la del filtro discretizado a 4 kHz "
                "(transformación bilineal) que usa el gemelo. Se buscan los puntos de −3 dB respecto de la ganancia máxima.",
                "Cortes a 16 y 482 Hz con ±5 %; ganancia en la banda de 20 dB (×10).") as p:
        R1, C1, Rf, Cf, Rth, C2 = 10e3, 1e-6, 100e3, 3.3e-9, 51e3, 1e-6
        num = np.polymul([-Rf * C1, 0], [Rth * C2, 0])
        den = np.polymul(np.polymul([Rf * Cf, 1], [R1 * C1, 1]), [Rth * C2, 1])
        f = np.logspace(0, 3.3, 3000)
        _, ha = signal.freqs(num, den, 2 * np.pi * f)
        _, hd = signal.sosfreqz(gemelo2.SOS_FILTRO, worN=f, fs=sm.FS_INTERNA)
        res = {}
        for nombre, h in (("analogico", ha), ("digital", hd)):
            db = 20 * np.log10(np.abs(h))
            tope = db.max()
            dentro = np.where(db >= tope - 3)[0]
            res[nombre] = {"ganancia_db": tope, "f_baja": f[dentro[0]], "f_alta": f[dentro[-1]]}
        fl, fh = 1 / (2 * np.pi * R1 * C1), 1 / (2 * np.pi * Rf * Cf)
        e_b = abs(res["digital"]["f_baja"] / fl - 1) * 100
        e_a = abs(res["digital"]["f_alta"] / fh - 1) * 100
        p.resultado = {**res, "f_baja_diseno": fl, "f_alta_diseno": fh, "error_baja_pct": e_b, "error_alta_pct": e_a}
        p.cumple = bool(e_b <= 5 and e_a <= 5 and abs(res["digital"]["ganancia_db"] - 20) < 0.5)
        p.conclusion = (f"Cortes del filtro del gemelo: {res['digital']['f_baja']:.1f} Hz y {res['digital']['f_alta']:.0f} Hz "
                        f"(diseño {fl:.1f} y {fh:.0f} Hz; errores {e_b:.1f} % y {e_a:.1f} %), ganancia {res['digital']['ganancia_db']:.1f} dB.")
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.semilogx(f, 20 * np.log10(np.abs(ha)), color=GRIS, lw=2.5, label="analógico")
        ax.semilogx(f, 20 * np.log10(np.abs(hd)), color=ACENTO, lw=1.3, ls="--", label="discretizado a 4 kHz (gemelo)")
        for x in (fl, fh):
            ax.axvline(x, color=NARANJA, ls=":")
        ax.axhline(res["digital"]["ganancia_db"] - 3, color=ROJO, lw=0.8, ls=":", label="−3 dB")
        ax.set_xlabel("frecuencia (Hz)")
        ax.set_ylabel("ganancia (dB)")
        ax.set_ylim(-10, 24)
        ax.set_title("Pasa-banda de la segunda etapa (16–482 Hz, ×10)")
        ax.legend()
        p.figuras.append(figura("A3", "bode", fig, 14))


def a4(datos):
    with Prueba("A4", "Espectro de la señal simulada", "A. Gemelo digital",
                "Verificar que el contenido en frecuencia del EMG simulado esté en el rango fisiológico del EMG de superficie.",
                ["RF-01"],
                "Con 12 personas y 6 gestos se toma la señal de cada electrodo en contracción sostenida (desde 300 ms después del "
                "inicio), se estima la densidad espectral (Welch, 256 ms) y se calculan la frecuencia mediana (MDF) y la media (MNF) "
                "del electrodo más activo de cada gesto.",
                "Frecuencia mediana entre 50 y 150 Hz (rango reportado para EMG de superficie del antebrazo).") as p:
        mdf, mnf, psds = [], [], []
        for s in datos:
            x = s["uv"][de.INICIO + 300:]
            c = int(np.argmax(np.std(x, 0)))
            fr, P = signal.welch(x[:, c] - x[:, c].mean(), fs=1000, nperseg=256)
            acum = np.cumsum(P) / P.sum()
            mdf.append(fr[np.searchsorted(acum, 0.5)])
            mnf.append((fr * P).sum() / P.sum())
            psds.append(P / P.max())
        mdf, mnf = np.array(mdf), np.array(mnf)
        p.resultado = {"mdf_mediana_hz": np.median(mdf), "mdf_p10": np.percentile(mdf, 10), "mdf_p90": np.percentile(mdf, 90),
                       "mnf_mediana_hz": np.median(mnf), "fraccion_en_rango": float(np.mean((mdf >= 50) & (mdf <= 150))), "n": len(mdf)}
        p.cumple = bool(50 <= np.median(mdf) <= 150 and p.resultado["fraccion_en_rango"] >= 0.9)
        p.conclusion = (f"Frecuencia mediana {np.median(mdf):.0f} Hz (p10–p90: {np.percentile(mdf, 10):.0f}–{np.percentile(mdf, 90):.0f} Hz); "
                        f"{p.resultado['fraccion_en_rango'] * 100:.0f} % de las señales dentro de 50–150 Hz.")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        P = np.array(psds)
        ax[0].fill_between(fr, np.percentile(P, 10, 0), np.percentile(P, 90, 0), color=ACENTO, alpha=0.2)
        ax[0].plot(fr, np.median(P, 0), color=ACENTO)
        ax[0].axvspan(50, 150, color=VERDES[1], alpha=0.12, label="rango esperado de la MDF")
        ax[0].set_xlim(0, 500)
        ax[0].set_xlabel("frecuencia (Hz)")
        ax[0].set_ylabel("potencia normalizada")
        ax[0].set_title("Densidad espectral del electrodo más activo")
        ax[0].legend()
        ax[1].hist(mdf, bins=20, color=ACENTO, alpha=0.8)
        ax[1].axvline(50, color=GRIS, ls="--")
        ax[1].axvline(150, color=GRIS, ls="--")
        ax[1].set_xlabel("frecuencia mediana (Hz)")
        ax[1].set_title("Frecuencia mediana por persona y gesto")
        p.figuras.append(figura("A4", "espectro", fig))


def a5(datos):
    with Prueba("A5", "Diafonía entre electrodos", "A. Gemelo digital",
                "Medir cuánto se parecen las señales de electrodos vecinos (diafonía) y verificar que los 10 canales aporten información distinta.",
                ["RF-02"],
                "Con las mismas sesiones de A4 se calcula la correlación de Pearson entre las envolventes (RMS cada 50 ms) de cada par "
                "de electrodos, promediada sobre personas y gestos.",
                "Correlación media entre vecinos menor a 0,9 (no son redundantes) y entre flexores y extensores opuestos menor a 0,5.") as p:
        Cs = []
        for s in datos:
            x = s["uv"][de.INICIO:]
            n = len(x) // 50
            env = np.sqrt((x[:n * 50].reshape(n, 50, 10) ** 2).mean(1))
            Cs.append(np.corrcoef(env.T))
        Cm = np.nanmean(Cs, 0)
        vecinos = [Cm[i, i + 1] for i in range(4)] + [Cm[i, i + 1] for i in range(5, 9)]
        opuestos = [Cm[c, c + 5] for c in range(5)]
        p.resultado = {"vecinos_media": np.mean(vecinos), "vecinos_max": np.max(vecinos), "opuestos_media": np.mean(opuestos),
                       "matriz": Cm}
        p.cumple = bool(np.mean(vecinos) < 0.9 and np.mean(opuestos) < 0.5)
        p.conclusion = (f"Correlación media entre vecinos {np.mean(vecinos):.2f} (máx. {np.max(vecinos):.2f}); "
                        f"entre flexor y extensor opuestos {np.mean(opuestos):.2f}.")
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        im = ax.imshow(Cm, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(10), ia.CANALES)
        ax.set_yticks(range(10), ia.CANALES)
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{Cm[i, j]:.2f}", ha="center", va="center", fontsize=6.3, color="white" if abs(Cm[i, j]) > 0.6 else "#333")
        ax.set_title("Correlación de las envolventes entre electrodos")
        plt.colorbar(im, fraction=0.045)
        p.figuras.append(figura("A5", "diafonia", fig, 11))


def _estadisticas(x, fs):
    """Estadísticas que no dependen de la escala, en ventanas de 200 ms a fs Hz."""
    n = int(0.2 * fs)
    k = len(x) // n
    v = x[:k * n].reshape(k, n) - x[:k * n].reshape(k, n).mean(1, keepdims=True)
    rms = np.sqrt((v ** 2).mean(1))
    mav = np.abs(v).mean(1)
    ok = mav > 0
    banda = signal.butter(4, [20, 90], btype="band", fs=fs, output="sos")     # misma banda de análisis para los dos
    fr, P = signal.welch(signal.sosfiltfilt(banda, x - x.mean()), fs=fs, nperseg=min(len(x), 128))
    acum = np.cumsum(P) / P.sum()
    return rms[ok] / mav[ok], stats.kurtosis(v[ok], axis=1), fr[np.searchsorted(acum, 0.5)]


def a6(datos):
    with Prueba("A6", "Señal simulada frente a EMG real", "A. Gemelo digital",
                "Comparar estadísticas de forma de la señal del gemelo con EMG real del dataset UCI «EMG data for gestures» (36 personas, pulsera Myo de 8 canales, CC BY 4.0).",
                ["RF-01"],
                "La Myo muestrea a ~200 Hz, así que el gemelo se filtra (pasa-bajos 90 Hz) y se lleva a 200 Hz antes de comparar. En "
                "contracción se calculan, por ventana de 200 ms del canal más activo: el cociente RMS/MAV (forma de la distribución de "
                "amplitud; 1,25 para una gaussiana), la curtosis y la frecuencia mediana en la banda de análisis 20–90 Hz (la misma "
                "para las dos señales: el EMG real viene sin pasa-altos y con artefactos de movimiento).",
                "Diferencia de medianas menor a 10 % en RMS/MAV y a 25 % en la frecuencia mediana (en la banda común).") as p:
        sim = {"rms_mav": [], "curtosis": [], "mdf": []}
        sos = signal.butter(4, 90, fs=1000, output="sos")
        for s in datos:
            x = s["uv"][de.INICIO + 300:]
            c = int(np.argmax(np.std(x, 0)))
            y = signal.sosfiltfilt(sos, x[:, c])[::5]
            r, k, m = _estadisticas(y, 200)
            sim["rms_mav"] += list(r)
            sim["curtosis"] += list(k)
            sim["mdf"].append(m)
        real = {"rms_mav": [], "curtosis": [], "mdf": []}
        archivos = sorted(glob.glob(str(Path.home() / "gemelo_mano" / "datos" / "externos" / "uci_emg" / "**" / "*.txt"), recursive=True))
        import pandas as pd
        fs_real = []
        for a in archivos[::3][:24]:
            d = pd.read_csv(a, sep="\t")
            t = d["time"].to_numpy(float)
            fs_real.append(1000 / np.median(np.diff(np.unique(t))))
            clase_v = d["class"].to_numpy()
            canales = np.stack([d[f"channel{i}"].to_numpy(float) for i in range(1, 9)], 1)
            cortes = np.flatnonzero(np.diff(clase_v) != 0) + 1                 # cada repetición por separado
            for i0, i1 in zip(np.r_[0, cortes], np.r_[cortes, len(clase_v)]):
                if clase_v[i0] not in (2, 3, 4, 7) or t[i1 - 1] - t[i0] < 1500:  # puño, flexión, extensión, palma
                    continue
                tt, xx = t[i0:i1], canales[i0:i1]
                tt, u = np.unique(tt, return_index=True)
                rej = np.arange(tt[0] + 300, tt[-1], 5.0)                     # desde 300 ms después del inicio, a 200 Hz
                X = np.stack([np.interp(rej, tt, xx[u, j]) for j in range(8)], 1)
                c = int(np.argmax(np.std(X, 0)))
                r, k, mm = _estadisticas(X[:, c], 200)
                real["rms_mav"] += list(r)
                real["curtosis"] += list(k)
                real["mdf"].append(mm)
        res = {}
        for k in sim:
            s, r = np.asarray(sim[k]), np.asarray(real[k])
            res[k] = {"sim_mediana": np.median(s), "sim_p25": np.percentile(s, 25), "sim_p75": np.percentile(s, 75),
                      "real_mediana": np.median(r), "real_p25": np.percentile(r, 25), "real_p75": np.percentile(r, 75),
                      "dif_pct": abs(np.median(s) / np.median(r) - 1) * 100, "ks": stats.ks_2samp(s, r).statistic}
        res["fs_real_hz"] = float(np.median(fs_real))
        p.resultado = res
        p.cumple = bool(res["rms_mav"]["dif_pct"] < 10 and res["mdf"]["dif_pct"] < 25)
        p.conclusion = (f"RMS/MAV: gemelo {res['rms_mav']['sim_mediana']:.3f} vs real {res['rms_mav']['real_mediana']:.3f} "
                        f"({res['rms_mav']['dif_pct']:.1f} %); frecuencia mediana en 20–90 Hz: {res['mdf']['sim_mediana']:.0f} vs "
                        f"{res['mdf']['real_mediana']:.0f} Hz ({res['mdf']['dif_pct']:.0f} %); curtosis {res['curtosis']['sim_mediana']:.2f} "
                        f"vs {res['curtosis']['real_mediana']:.2f}.")
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        for a, k, t in zip(ax, ("rms_mav", "curtosis", "mdf"), ("RMS / MAV", "curtosis (exceso)", "frecuencia mediana 20–90 Hz")):
            a.boxplot([sim[k], real[k]], showfliers=False, patch_artist=True,
                      boxprops={"facecolor": "#dbe7f7"}, medianprops={"color": ACENTO})
            a.set_xticks([1, 2], ["gemelo", "real (UCI)"])
            a.set_title(t)
        p.figuras.append(figura("A6", "real", fig))


def a_extra_etapas():
    """Figura para el documento técnico: la señal en cada etapa de la cadena, un gesto, 10 electrodos."""
    d = sesiones(1, ("puno",), semilla=5)[0]
    t = np.arange(len(d["uv"])) / 1000
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for a, (k, u, esc) in zip(ax, (("uv", "µV", None), ("ad620", "V", None), ("adc", "cuentas", None))):
        s = d[k].astype(float)
        s = s - np.median(s, 0)
        e = np.percentile(np.abs(s), 99.9)
        for c in range(10):
            a.plot(t, 9 - c + 0.48 * np.clip(s[:, c] / e, -1, 1), color=COLORES_CANAL[c], lw=0.5)
        a.set_yticks(range(10), ia.CANALES[::-1])
        a.set_ylabel(f"{u} (±{e:.3g})")
        a.grid(False)
    ax[0].set_title("Electrodos")
    ax[1].set_title("Salida del AD620 (G = 106)")
    ax[2].set_title("ADC de 12 bits (sin la continua)")
    ax[2].set_xlabel("tiempo (s) · el gesto «puño» empieza a los 0,5 s")
    figura("T", "etapas_puno", fig)


if __name__ == "__main__":
    a1()
    a2()
    a3()
    datos = sesiones()
    a4(datos)
    a5(datos)
    a6(datos)
    a_extra_etapas()
