"""
Pruebas C · Modelos de IA
=========================
  C1 2 electrodos frente a 10              C2 ablación: cuántos y cuáles electrodos
  C3 generalización a personas nuevas      C4 robustez (desplazamiento, ruido, fatiga)
  C5 falla de un electrodo                 C9 convergencia de la competencia (2 y 10 electrodos)
"""

from comun import (ACENTO, AZULES, COLORES_CANAL, GESTOS_ES, GRIS, MOTOR, NARANJA, ROJO, VERDES, Prueba, campeona,
                   columnas_canales, entrenar_mlp, figura, intervalo_bootstrap, leer_examen, leer_lotes, plt, predecir,
                   separar_personas)

import gc
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, r2_score

import ia_mano as ia
from gemelo import metricas
from gemelo.neuro import pista

UNIFORME = np.full(7, 1 / 7)


def _metricas(y, probs, P=None, pose=None):
    m = {"acierto": float(np.mean(probs.argmax(1) == y)), "pseudo_r2": metricas.pseudo_r2_mcfadden(y, probs, UNIFORME)}
    if P is not None:
        m["r2_pose"] = float(r2_score(P, pose))
    return m


def c1(d):
    with Prueba("C1", "2 electrodos frente a 10", "C. Modelos de IA",
                "Cuantificar cuánto mejora el reconocimiento al pasar de 2 electrodos (uno sobre flexores y otro sobre extensores, "
                "como el diseño inicial) a 10.",
                ["RF-02", "RF-06", "RF-13", "RNF-01"],
                "Los 2 electrodos del diseño inicial estaban a 270° y 90°: son F3 y E3 del anillo actual, así que la configuración de 2 "
                "se obtiene con las mismas personas usando solo sus características. Se entrena el mismo MLP (gesto + pose) con 2 y con "
                "10 electrodos en 5 particiones por persona (80 % entrenamiento, 20 % prueba) y se compara con la prueba t pareada.",
                "Con 10 electrodos el acierto, el pseudo-R² y el R² de la pose son mayores en las 5 particiones (p < 0,05).") as p:
        cols2 = columnas_canales([ia.CANALES.index("F3"), ia.CANALES.index("E3")])
        uso = d["ronda"] >= 1
        filas = []
        for s in range(5):
            tr, te = separar_personas(d["persona"], 0.2, s)
            tr, te = tr & uso, te & uso
            fila = {"particion": s}
            for nombre, cols in (("2", cols2), ("10", np.arange(120))):
                pr, pose = entrenar_mlp(d["X"][tr][:, cols], d["gesto"][tr], d["X"][te][:, cols], Ptr=d["pose"][tr], epocas=6, semilla=s)
                m = _metricas(d["gesto"][te], pr, d["pose"][te], pose)
                fila.update({f"{k}_{nombre}": v for k, v in m.items()})
            filas.append(fila)
            print("   ", {k: round(v, 4) for k, v in fila.items()}, flush=True)
        res = {"particiones": filas}
        cumple = True
        for k in ("acierto", "pseudo_r2", "r2_pose"):
            a, b = np.array([f[f"{k}_2"] for f in filas]), np.array([f[f"{k}_10"] for f in filas])
            t = stats.ttest_rel(b, a)
            res[k] = {"2": a.mean(), "2_sd": a.std(), "10": b.mean(), "10_sd": b.std(), "p": t.pvalue}
            cumple &= bool(np.all(b > a) and t.pvalue < 0.05)
        p.resultado = res
        p.cumple = cumple
        p.conclusion = (f"Acierto {res['acierto']['2'] * 100:.1f} % → {res['acierto']['10'] * 100:.1f} %; pseudo-R² "
                        f"{res['pseudo_r2']['2']:.3f} → {res['pseudo_r2']['10']:.3f}; R² pose {res['r2_pose']['2']:.3f} → "
                        f"{res['r2_pose']['10']:.3f} (p = {res['acierto']['p']:.1e}).")
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
        for a, k, t in zip(ax, ("acierto", "pseudo_r2", "r2_pose"), ("acierto", "pseudo-R² (gesto)", "R² (pose)")):
            v2, v10 = [f[f"{k}_2"] for f in filas], [f[f"{k}_10"] for f in filas]
            a.bar([0, 1], [np.mean(v2), np.mean(v10)], color=[GRIS, ACENTO], width=0.6)
            a.errorbar([0, 1], [np.mean(v2), np.mean(v10)], [np.std(v2), np.std(v10)], fmt="none", color="#333", capsize=4)
            lo = min(min(v2), min(v10))
            a.set_ylim(max(0, lo - (1 - lo) * 0.6), 1)
            a.set_xticks([0, 1], ["2 electrodos", "10 electrodos"])
            a.set_title(t)
        p.figuras.append(figura("C1", "2_vs_10", fig))


def c2(d):
    with Prueba("C2", "Ablación: cuántos y cuáles electrodos", "C. Modelos de IA",
                "Determinar cuánto aporta cada electrodo y cuántos hacen falta, como guía para el hardware.",
                ["RF-02", "RNF-03"],
                "Selección hacia adelante: empezando sin electrodos, en cada paso se agrega el electrodo que más sube el acierto en "
                "personas nuevas (análisis discriminante lineal, rápido y estable). Además, para cada cantidad k se promedia el acierto "
                "de 12 subconjuntos al azar de k electrodos.",
                "Documentar la curva y el orden de importancia; con los 10 el acierto debe ser el máximo (o a menos de 0,5 puntos).") as p:
        uso = d["ronda"] >= 1
        tr, te = separar_personas(d["persona"], 0.25, 7)
        tr, te = tr & uso, te & uso
        rng = np.random.default_rng(0)
        idx_tr = rng.choice(np.flatnonzero(tr), min(120_000, tr.sum()), replace=False)
        Xtr, ytr, Xte, yte = d["X"][idx_tr], d["gesto"][idx_tr], d["X"][te], d["gesto"][te]

        def acc(canales):
            c = columnas_canales(canales)
            m = LinearDiscriminantAnalysis().fit(Xtr[:, c], ytr)
            return float(m.score(Xte[:, c], yte))

        elegidos, curva = [], []
        restantes = list(range(10))
        while restantes:
            mejor = max(restantes, key=lambda c: acc(elegidos + [c]))
            elegidos.append(mejor)
            restantes.remove(mejor)
            curva.append(acc(elegidos))
            print(f"    + {ia.CANALES[mejor]} → {curva[-1] * 100:.1f} %", flush=True)
        azar = []
        for k in range(1, 11):
            vals = [acc(list(rng.choice(10, k, replace=False))) for _ in range(12 if k < 10 else 1)]
            azar.append({"k": k, "media": np.mean(vals), "min": np.min(vals), "max": np.max(vals)})
        p.resultado = {"orden": [ia.CANALES[c] for c in elegidos], "curva_adelante": curva, "azar": azar,
                       "k_para_99pct_del_maximo": int(np.argmax(np.array(curva) >= 0.99 * max(curva)) + 1)}
        p.cumple = bool(curva[-1] >= max(curva) - 0.005)
        p.conclusion = (f"Orden de importancia: {', '.join(ia.CANALES[c] for c in elegidos)}. Con {p.resultado['k_para_99pct_del_maximo']} "
                        f"electrodos bien elegidos se llega al 99 % del máximo; con 2 al azar el acierto medio es {azar[1]['media'] * 100:.1f} % "
                        f"y con los 10, {curva[-1] * 100:.1f} % (LDA).")
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
        k = np.arange(1, 11)
        ax[0].plot(k, np.array(curva) * 100, "o-", color=ACENTO, label="mejor electrodo en cada paso")
        ax[0].fill_between(k, [a["min"] * 100 for a in azar], [a["max"] * 100 for a in azar], color=GRIS, alpha=0.2)
        ax[0].plot(k, [a["media"] * 100 for a in azar], "s--", color=GRIS, label="electrodos al azar (media y rango)")
        for i, c in enumerate(elegidos):
            ax[0].annotate(ia.CANALES[c], (k[i], curva[i] * 100), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax[0].set_xlabel("cantidad de electrodos")
        ax[0].set_ylabel("acierto en personas nuevas (%)")
        ax[0].set_title("¿Cuántos electrodos hacen falta?")
        ax[0].legend(fontsize=7.5)
        gan = np.diff(np.r_[1 / 7, curva]) * 100
        ax[1].bar(range(10), gan, color=[COLORES_CANAL[c] for c in elegidos])
        ax[1].set_xticks(range(10), [ia.CANALES[c] for c in elegidos])
        ax[1].set_ylabel("puntos de acierto que agrega")
        ax[1].set_title("Aporte de cada electrodo (en el orden elegido)")
        p.figuras.append(figura("C2", "ablacion", fig))


def c3():
    with Prueba("C3", "Generalización a personas nuevas", "C. Modelos de IA",
                "Medir el desempeño de la campeona de la competencia con las 20 personas del examen, que ninguna red vio nunca.",
                ["RF-06", "RF-07", "RF-13", "RNF-01"],
                "Se carga la campeona más reciente del motor y se evalúa en el examen fijo (20 personas de los 5 niveles de dificultad, "
                "cada una con su calibración). Se calculan acierto, recall por gesto, pseudo-R² de McFadden, R² de la pose, la matriz "
                "de confusión y el intervalo de confianza del 95 % del acierto por persona (bootstrap).",
                "Acierto ≥ 90 % y recall ≥ 85 % en cada gesto (RNF-01).") as p:
        modelo, ficha, carpeta = campeona()
        ex = leer_examen()
        probs, pose = predecir(modelo, ex["X"])
        y = ex["gesto"].astype(int)
        m = _metricas(y, probs, ex["pose"], pose)
        cm = confusion_matrix(y, probs.argmax(1), labels=range(7))
        recall = np.diag(cm) / cm.sum(1)
        por_persona = [float(np.mean(probs[ex["persona"] == q].argmax(1) == y[ex["persona"] == q])) for q in np.unique(ex["persona"])]
        estable = ~ex["transicion"].astype(bool)
        p.resultado = {**m, "campeona": ficha.get("campeona"), "generacion": ficha.get("generacion"), "recall": recall,
                       "recall_min": recall.min(), "ic95_acierto": intervalo_bootstrap(por_persona),
                       "acierto_por_persona_min": min(por_persona), "acierto_sin_transiciones": float(np.mean(probs[estable].argmax(1) == y[estable])),
                       "matriz": cm, "parametros": int(sum(np.prod(v.shape) for v in modelo.trainable_weights)), "n": len(y)}
        p.cumple = bool(m["acierto"] >= 0.90 and recall.min() >= 0.85)
        ic = p.resultado["ic95_acierto"]
        p.conclusion = (f"Campeona #{ficha.get('campeona')} (generación {ficha.get('generacion')}): acierto {m['acierto'] * 100:.2f} % "
                        f"(IC 95 %: {ic[0] * 100:.1f}–{ic[1] * 100:.1f} %), recall mínimo {recall.min() * 100:.1f} % "
                        f"({GESTOS_ES[int(np.argmin(recall))]}), pseudo-R² {m['pseudo_r2']:.3f}, R² pose {m['r2_pose']:.3f}.")
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [1.2, 1]})
        cmn = cm / cm.sum(1, keepdims=True)
        ax[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax[0].grid(False)
        for i in range(7):
            for j in range(7):
                if cm[i, j]:
                    ax[0].text(j, i, f"{cmn[i, j] * 100:.1f}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.6 else "#333")
        ax[0].set_xticks(range(7), GESTOS_ES, rotation=40)
        ax[0].set_yticks(range(7), GESTOS_ES)
        ax[0].set_xlabel("predicho")
        ax[0].set_ylabel("real")
        ax[0].set_title("Matriz de confusión (% por fila)")
        ax[1].barh(range(len(por_persona)), np.sort(por_persona)[::-1] * 100, color=ACENTO)
        ax[1].axvline(90, color=ROJO, ls="--", label="mínimo (90 %)")
        ax[1].set_xlim(min(80, min(por_persona) * 100 - 2), 100)
        ax[1].set_xlabel("acierto (%)")
        ax[1].set_ylabel("persona del examen")
        ax[1].set_yticks([])
        ax[1].set_title("Acierto por persona")
        ax[1].legend()
        p.figuras.append(figura("C3", "generalizacion", fig))


def _evaluar_sesion(args):
    semilla, ajustes = args
    d = pista.recorrido(semilla, ajustes=ajustes)
    return d["X"], d["gesto"], d["pose"]


def _barrido(modelo, condiciones, semillas, pool):
    """Para cada condición, acierto de la campeona en personas nuevas simuladas con esa perturbación."""
    out = []
    for nombre, ajustes in condiciones:
        res = list(pool.map(_evaluar_sesion, [(s, ajustes) for s in semillas]))
        X = np.concatenate([r[0] for r in res])
        y = np.concatenate([r[1] for r in res]).astype(int)
        pr, _ = predecir(modelo, X)
        accs = [float(np.mean(predecir(modelo, r[0])[0].argmax(1) == r[1])) for r in res]
        out.append({"condicion": nombre, **ajustes, "acierto": float(np.mean(pr.argmax(1) == y)), "sd_personas": float(np.std(accs))})
        print(f"    {nombre:28s} {out[-1]['acierto'] * 100:5.1f} %", flush=True)
    return out


def c4(pool):
    with Prueba("C4", "Robustez: desplazamiento, ruido y fatiga", "C. Modelos de IA",
                "Medir cuánto se degrada la campeona cuando los electrodos se corren después de calibrar, cuando hay más ruido o fatiga, "
                "y cuánto se recupera al volver a calibrar.",
                ["RF-05", "RNF-02"],
                "Se simulan 8 personas nuevas (nivel de dificultad 2) por condición, cada una con la pista de 21 gestos. Se barre: giro "
                "de los 10 electrodos después de calibrar (0–30°), corrimiento a lo largo del brazo (0–2 cm), ruido de los electrodos "
                "(×1 a ×5) y fatiga (0 a 0,9). Para el giro se repite con los electrodos ya girados al calibrar (recalibración).",
                "Acierto ≥ 85 % con giro de 10° sin recalibrar, ruido ×3 y fatiga 0,6; con recalibración ≥ 95 % para todos los giros.") as p:
        modelo, ficha, _ = campeona()
        sem = list(range(9000, 9008))
        base = {"nivel": 1}
        giros = (0, 5, 10, 15, 20, 30)
        r_giro = _barrido(modelo, [(f"giro {g}° sin recalibrar", {**base, "giro_deg": g}) for g in giros], sem, pool)
        r_recal = _barrido(modelo, [(f"giro {g}° recalibrado", {**base, "giro_antes_deg": g}) for g in giros], sem, pool)
        r_axial = _barrido(modelo, [(f"corrimiento {a} cm", {**base, "axial_cm": a}) for a in (0, 0.5, 1, 1.5, 2)], sem, pool)
        r_ruido = _barrido(modelo, [(f"ruido ×{r}", {**base, "ruido": r}) for r in (1, 2, 3, 5)], sem, pool)
        r_fat = _barrido(modelo, [(f"fatiga {f}", {**base, "fatiga": f}) for f in (0.0, 0.3, 0.6, 0.9)], sem, pool)
        v = lambda lista, k, val: next(r["acierto"] for r in lista if r.get(k) == val)
        p.resultado = {"campeona": ficha.get("campeona"), "giro": r_giro, "recalibrado": r_recal, "axial": r_axial, "ruido": r_ruido, "fatiga": r_fat}
        criterios = {"giro10": v(r_giro, "giro_deg", 10) >= 0.85, "ruido3": v(r_ruido, "ruido", 3) >= 0.85,
                     "fatiga06": v(r_fat, "fatiga", 0.6) >= 0.85, "recal": min(r["acierto"] for r in r_recal) >= 0.95}
        p.resultado["criterios"] = criterios
        p.cumple = all(criterios.values())
        p.conclusion = (f"Giro sin recalibrar: {v(r_giro, 'giro_deg', 10) * 100:.0f} % a 10° y {v(r_giro, 'giro_deg', 30) * 100:.0f} % a 30°; "
                        f"recalibrando: ≥ {min(r['acierto'] for r in r_recal) * 100:.0f} %. Ruido ×3: {v(r_ruido, 'ruido', 3) * 100:.0f} %; "
                        f"fatiga 0,6: {v(r_fat, 'fatiga', 0.6) * 100:.0f} %; corrimiento 2 cm: {v(r_axial, 'axial_cm', 2) * 100:.0f} %.")
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.1), sharey=True)
        ax[0].plot(giros, [r["acierto"] * 100 for r in r_giro], "o-", color=ROJO, label="sin recalibrar")
        ax[0].plot(giros, [r["acierto"] * 100 for r in r_recal], "s-", color=ACENTO, label="recalibrando")
        ax[0].set_xlabel("giro de los electrodos (°)")
        ax[0].legend(fontsize=7.5)
        for a, lista, k, t in ((ax[1], r_axial, "axial_cm", "corrimiento (cm)"), (ax[2], r_ruido, "ruido", "ruido (× nominal)"),
                               (ax[3], r_fat, "fatiga", "fatiga")):
            a.plot([r[k] for r in lista], [r["acierto"] * 100 for r in lista], "o-", color=ROJO if k == "axial_cm" else ACENTO)
            a.set_xlabel(t)
        for a in ax:
            a.axhline(85, color=GRIS, ls="--", lw=1)
        ax[0].set_ylabel("acierto en personas nuevas (%)")
        fig.suptitle("Robustez de la campeona", fontweight="bold", y=1.02)
        p.figuras.append(figura("C4", "robustez", fig))


def c5(pool, d):
    with Prueba("C5", "Falla de un electrodo", "C. Modelos de IA",
                "Medir qué pasa si un electrodo se despega durante el uso y probar una mitigación: entrenar con electrodos apagados al azar.",
                ["RNF-03"],
                "Para cada uno de los 10 electrodos se simulan 8 personas nuevas a las que ese electrodo se les despega después de "
                "calibrar (el ADC queda plano, solo ruido). Se evalúa la campeona y dos MLP iguales entrenados con los datos del motor: "
                "uno normal y otro con «electrodo despegado» simulado al azar en el 30 % de las ventanas de entrenamiento.",
                "Con cualquier electrodo despegado, acierto ≥ 80 % (RNF-03).") as p:
        modelo, ficha, _ = campeona()
        sem = list(range(9100, 9108))
        sesiones = {c: list(pool.map(_evaluar_sesion, [(s, {"nivel": 1, "apagados": [c]}) for s in sem])) for c in range(10)}
        uso = d["ronda"] >= 1
        rng = np.random.default_rng(3)
        idx = rng.choice(np.flatnonzero(uso), min(250_000, uso.sum()), replace=False)
        Xtr, ytr, ses_tr = d["X"][idx], d["gesto"][idx], d["sesion"][idx]
        cal_tr = np.stack([d["cal"][int(s)] for s in ses_tr])
        F_plano = ia.caracteristicas_lote(np.round(2048 + np.random.default_rng(0).normal(0, 1.5, (64, 200, 1))))[:, :4].mean(0)
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cal_t = torch.as_tensor(cal_tr, dtype=torch.float32, device=dev)
        lp = torch.as_tensor(np.log1p(F_plano), dtype=torch.float32, device=dev)

        def aumento(xb, ib):
            xb = xb.clone().float()
            n = len(ib)
            elegir = torch.rand(n, device=dev) < 0.3
            c = torch.randint(0, 10, (n,), device=dev)
            for k in range(4):
                val = lp[k] - cal_t[ib, c * 4 + k]
                for j in range(3):
                    col = j * 40 + c * 4 + k
                    xb[elegir, col[elegir]] = val[elegir]
            return xb

        X_eval = {c: np.concatenate([r[0] for r in sesiones[c]]) for c in range(10)}
        y_eval = {c: np.concatenate([r[1] for r in sesiones[c]]).astype(int) for c in range(10)}
        Xall = np.concatenate(list(X_eval.values()))
        pr_n = entrenar_mlp(Xtr, ytr, Xall, epocas=6, semilla=1)
        pr_a = entrenar_mlp(Xtr, ytr, Xall, epocas=6, semilla=1, aumento=aumento)
        filas, k0 = [], 0
        sanas = list(pool.map(_evaluar_sesion, [(s, {"nivel": 1}) for s in sem]))
        Xs, ys = np.concatenate([r[0] for r in sanas]), np.concatenate([r[1] for r in sanas]).astype(int)
        base = {"campeona": float(np.mean(predecir(modelo, Xs)[0].argmax(1) == ys)),
                "mlp_normal": float(np.mean(entrenar_mlp(Xtr, ytr, Xs, epocas=6, semilla=1).argmax(1) == ys)),
                "mlp_con_apagados": float(np.mean(entrenar_mlp(Xtr, ytr, Xs, epocas=6, semilla=1, aumento=aumento).argmax(1) == ys))}
        for c in range(10):
            n = len(y_eval[c])
            fila = {"electrodo": ia.CANALES[c],
                    "campeona": float(np.mean(predecir(modelo, X_eval[c])[0].argmax(1) == y_eval[c])),
                    "mlp_normal": float(np.mean(pr_n[k0:k0 + n].argmax(1) == y_eval[c])),
                    "mlp_con_apagados": float(np.mean(pr_a[k0:k0 + n].argmax(1) == y_eval[c]))}
            k0 += n
            filas.append(fila)
            print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()}, flush=True)
        peor = {k: min(f[k] for f in filas) for k in ("campeona", "mlp_normal", "mlp_con_apagados")}
        p.resultado = {"por_electrodo": filas, "sin_falla": base, "peor": peor, "campeona_id": ficha.get("campeona")}
        p.cumple = bool(peor["campeona"] >= 0.80)
        p.conclusion = (f"Peor caso con un electrodo despegado: campeona {peor['campeona'] * 100:.0f} %, MLP normal "
                        f"{peor['mlp_normal'] * 100:.0f} %, MLP entrenado con electrodos apagados {peor['mlp_con_apagados'] * 100:.0f} % "
                        f"(sin falla: {base['campeona'] * 100:.0f} / {base['mlp_normal'] * 100:.0f} / {base['mlp_con_apagados'] * 100:.0f} %).")
        fig, ax = plt.subplots(figsize=(10, 3.4))
        x = np.arange(10)
        for k, (clave, nombre, color) in enumerate((("campeona", "campeona actual", GRIS), ("mlp_normal", "MLP normal", AZULES[2]),
                                                    ("mlp_con_apagados", "MLP entrenado con electrodos apagados", VERDES[2]))):
            ax.bar(x + (k - 1) * 0.27, [f[clave] * 100 for f in filas], 0.27, color=color, label=nombre)
        ax.axhline(80, color=ROJO, ls="--", lw=1, label="mínimo (80 %)")
        ax.set_xticks(x, [f"{f['electrodo']}\ndespegado" for f in filas], fontsize=8)
        ax.set_ylabel("acierto (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Acierto con un electrodo despegado después de calibrar")
        ax.legend(fontsize=7.5, ncol=4, loc="lower center")
        p.figuras.append(figura("C5", "falla_electrodo", fig))


def c9():
    with Prueba("C9", "Convergencia de la competencia (PBT)", "C. Modelos de IA",
                "Describir cómo aprende la población de 10 redes del motor y comparar la convergencia con 2 y con 10 electrodos.",
                ["RF-07", "RNF-09"],
                "Se leen de la base SQLite del motor las generaciones (mejor y promedio del puntaje en el examen) y las campeonas "
                "(generación y acierto), con 10 electrodos y con el archivo de 2 electrodos. Se mide en qué generación cada uno "
                "supera el 95 % y el 99 % de acierto.",
                "Informativa: documentar la convergencia; con 10 electrodos la campeona supera a la mejor de 2 electrodos.") as p:
        from comun import RESULTADOS
        res = {}
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        for nombre, ruta, color in (("10 electrodos", MOTOR / "gemelo.db", ACENTO),
                                    ("2 electrodos", MOTOR.parent / "archivo_2canales" / "gemelo.db", GRIS)):
            if not ruta.exists():
                continue
            db = sqlite3.connect(ruta)
            g = np.array(db.execute("SELECT id, mejor, media FROM generaciones ORDER BY id").fetchall(), float)
            c = np.array(db.execute("SELECT gen, acierto, pseudo_r2, r2_pose FROM campeonas ORDER BY id").fetchall(), float)
            hasta = lambda u: int(c[np.argmax(c[:, 1] >= u), 0]) if (c[:, 1] >= u).any() else None
            res[nombre] = {"generaciones": len(g), "campeonas": len(c), "mejor_acierto": c[:, 1].max(), "pseudo_r2": c[-1, 2],
                           "r2_pose": c[-1, 3], "gen_95": hasta(0.95), "gen_99": hasta(0.99)}
            ax[0].plot(g[:, 0], g[:, 1] * 100, color=color, lw=1.4, label=f"{nombre}: mejor")
            ax[0].plot(g[:, 0], g[:, 2] * 100, color=color, lw=0.6, alpha=0.5)
            ax[1].step(c[:, 0], c[:, 1] * 100, where="post", color=color, marker="o", ms=3, label=nombre)
        ax[0].set_xscale("symlog", linthresh=10)
        ax[0].set_ylim(80, 100)
        ax[0].set_xlabel("generación")
        ax[0].set_ylabel("puntaje en el examen (%)")
        ax[0].set_title("Mejor (grueso) y promedio de las 10 redes")
        ax[0].legend(fontsize=7.5)
        ax[1].set_xscale("symlog", linthresh=10)
        ax[1].set_ylim(90, 100)
        ax[1].set_xlabel("generación")
        ax[1].set_ylabel("acierto de la campeona (%)")
        ax[1].set_title("Campeonas a lo largo del tiempo")
        ax[1].legend(fontsize=7.5)
        p.figuras.append(figura("C9", "convergencia", fig))
        p.resultado = res
        a10, a2 = res.get("10 electrodos", {}).get("mejor_acierto"), res.get("2 electrodos", {}).get("mejor_acierto")
        p.cumple = None
        p.conclusion = (f"10 electrodos: {res['10 electrodos']['generaciones']} generaciones, mejor acierto {a10 * 100:.2f} % "
                        f"(≥ 99 % desde la generación {res['10 electrodos']['gen_99']}). 2 electrodos: {res['2 electrodos']['generaciones']} "
                        f"generaciones, mejor {a2 * 100:.2f} %." if a2 else "")


if __name__ == "__main__":
    import sys
    pedidas = set(sys.argv[1:]) or {"C1", "C2", "C3", "C4", "C5", "C9"}
    with ProcessPoolExecutor(3) as pool:
        if "C3" in pedidas:
            c3()
        if "C9" in pedidas:
            c9()
        if "C4" in pedidas:
            c4(pool)
        d = leer_lotes(max_lotes=40) if pedidas & {"C1", "C2", "C5"} else None
        if "C1" in pedidas:
            c1(d)
        if "C2" in pedidas:
            c2(d)
        if "C5" in pedidas:
            c5(pool, d)
