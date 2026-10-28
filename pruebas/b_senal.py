"""
Pruebas B · Procesamiento de la señal
=====================================
  B1 características contra fórmulas analíticas y contra la implementación de referencia
  B2 efecto de la calibración de la sesión
  B3 largo de la ventana: acierto frente a retardo
"""

from comun import (ACENTO, GRIS, NARANJA, Prueba, entrenar_mlp, figura, intervalo_bootstrap, leer_lotes, plt,
                   separar_personas)

import gc

import numpy as np

import ia_mano as ia
from gemelo import datos_emg as de
from gemelo.motor import gemelo2


def b1():
    with Prueba("B1", "Características contra fórmulas analíticas", "B. Procesamiento de la señal",
                "Verificar que MAV, RMS, ZC y WL (las mismas fórmulas del ESP32) den los valores teóricos y que la versión vectorizada "
                "coincida con la de referencia canal por canal.",
                ["RF-04"],
                "Se generan senoidales de amplitud A y frecuencia f conocidas (más la continua del ADC, 2048 cuentas) en ventanas de "
                "200 muestras y se comparan con los valores teóricos: MAV = 2A/π, RMS = A/√2, ZC = 2·f·T, WL ≈ 4A·f·T. Luego se "
                "comparan 2000 ventanas de ruido aleatorio de 10 canales entre caracteristicas_lote (vectorizada) y caracteristicas_canal.",
                "Error relativo menor a 1 % en MAV y RMS; ZC con a lo sumo 1 cruce de diferencia (borde de la ventana); WL dentro de "
                "la cota de discretización 1 − cos(πf/fs); diferencia vectorizada/referencia menor a 1e-9. (El criterio inicial pedía "
                "ZC exacto y WL < 2 %, sin considerar el muestreo: se corrigió con la cota teórica.)") as p:
        filas, peor, zc_dif, wl_ok = [], {"MAV": 0, "RMS": 0, "ZC": 0, "WL": 0}, 0, True
        T = ia.VENTANA / ia.FS
        n = np.arange(ia.VENTANA)
        for A in (100, 400, 900):
            for f in (40, 80, 150):
                x = 2048 + A * np.sin(2 * np.pi * f * n / ia.FS + 0.37)
                v = np.repeat(x[:, None], 10, 1)[None]
                F = ia.caracteristicas_lote(v)[0, :4]
                teo = {"MAV": 2 * A / np.pi, "RMS": A / np.sqrt(2), "ZC": 2 * f * T, "WL": 4 * A * f * T}
                err = {k: abs(F[i] / teo[k] - 1) * 100 for i, k in enumerate(teo)}
                for k in peor:
                    peor[k] = max(peor[k], err[k])
                zc_dif = max(zc_dif, abs(F[2] - teo["ZC"]))
                wl_ok &= err["WL"] <= (1 - np.cos(np.pi * f / ia.FS)) * 100 + 0.5
                filas.append({"A": A, "f": f, **{k: F[i] for i, k in enumerate(teo)}, **{f"{k}_teo": v_ for k, v_ in teo.items()}})
        rng = np.random.default_rng(1)
        V = np.round(2048 + rng.normal(0, rng.uniform(5, 400, (2000, 1, 10)), (2000, 200, 10)))
        L = ia.caracteristicas_lote(V)
        R = np.array([[v for c in range(10) for v in ia.caracteristicas_canal(V[i, :, c])] for i in range(2000)])
        dif = float(np.max(np.abs(L - R) / np.maximum(np.abs(R), 1)))
        offset = float(np.max(np.abs(ia.caracteristicas_lote(V + 300) - L)))
        p.resultado = {"tabla": filas, "error_max_pct": peor, "zc_max_cruces_dif": zc_dif, "wl_dentro_cota": bool(wl_ok),
                       "cota_wl_150hz_pct": (1 - np.cos(np.pi * 150 / ia.FS)) * 100,
                       "dif_vectorizada_referencia": dif, "efecto_continua": offset,
                       "criterio_inicial": "ZC exacto y WL < 2 %: no cumplía por efecto del muestreo (no es un error de la implementación)"}
        p.cumple = bool(peor["MAV"] < 1 and peor["RMS"] < 1 and zc_dif <= 1 and wl_ok and dif < 1e-9 and offset < 1e-6)
        p.conclusion = (f"Errores máximos: MAV {peor['MAV']:.2f} %, RMS {peor['RMS']:.2f} %, ZC {zc_dif:.0f} cruce, WL {peor['WL']:.2f} % "
                        f"(cota de muestreo a 150 Hz: {(1 - np.cos(np.pi * 150 / ia.FS)) * 100:.1f} %); "
                        f"vectorizada = referencia (dif. {dif:.1e}); sumar una continua no cambia nada.")
        fig, ax = plt.subplots(1, 4, figsize=(11, 2.9))
        for a, k in zip(ax, ("MAV", "RMS", "ZC", "WL")):
            a.plot([r[f"{k}_teo"] for r in filas], [r[k] for r in filas], "o", color=ACENTO)
            lim = [0, max(r[f"{k}_teo"] for r in filas) * 1.05]
            a.plot(lim, lim, color=GRIS, lw=1)
            a.set_title(k)
            a.set_xlabel("teórico")
        ax[0].set_ylabel("calculado")
        p.figuras.append(figura("B1", "analitico", fig))


def _contexto(Fc, vidx):
    return gemelo2.contexto(Fc, vidx.astype(np.int8))


def b2(d):
    with Prueba("B2", "Efecto de la calibración de la sesión", "B. Procesamiento de la señal",
                "Medir cuánto mejora el reconocimiento en personas nuevas al expresar las características relativas a la calibración "
                "de 20 s de cada sesión.",
                ["RF-05", "RNF-01"],
                "Con los datos guardados por el motor se separa el 20 % de las personas para prueba (5 particiones distintas). Se "
                "entrena el mismo MLP (120-64-32-7) con dos entradas: (a) log(1 + F) con contexto, sin calibrar; (b) log(1 + F) − "
                "calibración de la sesión, con contexto. Se compara el acierto en las personas de prueba.",
                "La calibración mejora el acierto en personas nuevas en todas las particiones.") as p:
        logF = np.log1p(d["F"])
        cal = np.stack([d["cal"][int(s)] for s in d["sesion"]])
        Xs = _contexto(logF, d["ventana_idx"])
        Xc = d["X"]
        uso = d["ronda"] >= 1
        filas = []
        for semilla in range(5):
            tr, te = separar_personas(d["persona"], 0.2, semilla)
            tr, te = tr & uso, te & uso
            fila = {"semilla": semilla}
            for nombre, X in (("sin_calibrar", Xs), ("calibrado", Xc)):
                pr = entrenar_mlp(X[tr], d["gesto"][tr], X[te], epocas=6, semilla=semilla)
                fila[nombre] = float(np.mean(pr.argmax(1) == d["gesto"][te]))
            filas.append(fila)
            print("   ", fila, flush=True)
        a, b = np.array([f["sin_calibrar"] for f in filas]), np.array([f["calibrado"] for f in filas])
        p.resultado = {"particiones": filas, "sin_calibrar_media": a.mean(), "calibrado_media": b.mean(),
                       "mejora_media_pp": (b - a).mean() * 100, "mejora_min_pp": (b - a).min() * 100, "cal_shape": list(cal.shape)}
        p.cumple = bool(np.all(b > a))
        p.conclusion = (f"Acierto en personas nuevas: {a.mean() * 100:.1f} % sin calibrar → {b.mean() * 100:.1f} % calibrado "
                        f"(+{(b - a).mean() * 100:.1f} puntos; mejora en {np.sum(b > a)} de 5 particiones).")
        fig, ax = plt.subplots(figsize=(5.5, 3.3))
        for k in range(len(filas)):
            ax.plot([0, 1], [a[k] * 100, b[k] * 100], "o-", color=GRIS, alpha=0.6)
        ax.plot([0, 1], [a.mean() * 100, b.mean() * 100], "o-", color=ACENTO, lw=3, label="media")
        ax.set_xticks([0, 1], ["sin calibrar", "calibrado (20 s)"])
        ax.set_ylabel("acierto en personas nuevas (%)")
        ax.set_title("Efecto de la calibración de la sesión")
        ax.legend()
        p.figuras.append(figura("B2", "calibracion", fig, 10))


def b3():
    with Prueba("B3", "Largo de la ventana: acierto frente a retardo", "B. Procesamiento de la señal",
                "Elegir el largo de la ventana de análisis: una ventana más larga da características más estables, pero la decisión llega más tarde.",
                ["RF-04", "RNF-04"],
                "Con la señal cruda guardada (10 electrodos) se recalculan las características usando las últimas L muestras de cada "
                "ventana (L = 50, 100, 150 y 200 ms), con su calibración por sesión y contexto de 3 ventanas, y se entrena el mismo MLP "
                "(3 particiones por persona). El retardo que agrega la ventana es L/2 (retardo de grupo de un promedio móvil).",
                "Documentar la curva; la ventana de 200 ms debe quedar dentro del 1 % del mejor acierto.") as p:
        import h5py
        from comun import MOTOR
        archivos = sorted((MOTOR / "lotes").glob("*.h5"))
        archivos = archivos[:: max(1, len(archivos) // 14)][:14]
        largos = (50, 100, 150, 200)
        F = {L: [] for L in largos}
        meta = {k: [] for k in ("gesto", "persona", "sesion", "ronda", "ventana_idx")}
        for a in archivos:
            with h5py.File(a, "r") as f:
                x = f["x"][()]
                for k in meta:
                    meta[k].append(f[k][()])
            for L in largos:
                F[L].append(ia.caracteristicas_lote(x[:, -L:, :]).astype(np.float32))
            del x
            gc.collect()
        meta = {k: np.concatenate(v) for k, v in meta.items()}
        uso = meta["ronda"] >= 1
        filas = []
        for L in largos:
            Fl = np.log1p(np.concatenate(F[L]))
            cal = {}
            for s in np.unique(meta["sesion"]):
                m = (meta["sesion"] == s) & (meta["ronda"] == 0)
                cal[s] = Fl[m].mean(0)
            Fc = Fl - np.stack([cal[s] for s in meta["sesion"]])
            X = _contexto(Fc, meta["ventana_idx"])
            acc = []
            for semilla in range(3):
                tr, te = separar_personas(meta["persona"], 0.25, semilla)
                tr, te = tr & uso, te & uso
                pr = entrenar_mlp(X[tr], meta["gesto"][tr], X[te], epocas=8, semilla=semilla)
                acc.append(float(np.mean(pr.argmax(1) == meta["gesto"][te])))
            filas.append({"ventana_ms": L, "acierto": np.mean(acc), "sd": np.std(acc), "retardo_ms": L / 2})
            print("   ", filas[-1], flush=True)
        mejor = max(f["acierto"] for f in filas)
        p.resultado = {"tabla": filas, "n_ventanas": int(uso.sum())}
        p.cumple = bool(filas[-1]["acierto"] >= mejor - 0.01)
        p.conclusion = "; ".join(f"{f['ventana_ms']} ms: {f['acierto'] * 100:.1f} %" for f in filas) + \
            f". Con 200 ms la decisión se retrasa {100} ms por la ventana."
        fig, ax = plt.subplots(figsize=(6, 3.3))
        ax.errorbar([f["ventana_ms"] for f in filas], [f["acierto"] * 100 for f in filas], [f["sd"] * 100 for f in filas],
                    fmt="o-", color=ACENTO, capsize=3, label="acierto (personas nuevas)")
        ax.set_xlabel("largo de la ventana (ms)")
        ax.set_ylabel("acierto (%)")
        a2 = ax.twinx()
        a2.plot([f["ventana_ms"] for f in filas], [f["retardo_ms"] for f in filas], "s--", color=NARANJA, label="retardo (ms)")
        a2.set_ylabel("retardo que agrega (ms)")
        a2.grid(False)
        ax.set_title("Largo de ventana: acierto frente a retardo")
        p.figuras.append(figura("B3", "ventana", fig, 11))


if __name__ == "__main__":
    b1()
    d = leer_lotes(max_lotes=40)
    b2(d)
    del d
    gc.collect()
    b3()
