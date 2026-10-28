"""
Pruebas C (cont.) · Neuroevolución, aprendizaje contrastivo y mano bebé
======================================================================
  C6 neuroevolución (NEAT) frente a gradiente
  C7 aprendizaje contrastivo con pocas etiquetas
  C8 mano bebé: reproducibilidad del aprendizaje autónomo
"""

from comun import ACENTO, GRIS, NARANJA, ROJO, VERDES, Prueba, figura, plt

import queue
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import ia_mano as ia
from gemelo.neuro import pista

UNIFORME = np.full(7, 1 / 7)


def c6(semillas=6, generaciones=150):
    with Prueba("C6", "Neuroevolución (NEAT) frente a gradiente", "C. Modelos de IA",
                "Medir cuánto aprende la neuroevolución desde cero y compararla con modelos entrenados con gradiente con los mismos datos.",
                ["RF-08", "RF-13"],
                f"Se corre NEAT ({generaciones} generaciones, 150 redes, 6 pistas por generación) con {semillas} semillas distintas, "
                "usando el banco de 40 personas de la aplicación. Con esas mismas 40 personas se entrenan una regresión logística y un "
                "MLP (64-32) con gradiente. Todos se evalúan en las mismas 8 personas del examen: acierto en las ventanas estables, "
                "pseudo-R² y segundos que aguanta sin chocar en la pista. Se comparan con la prueba de Wilcoxon (una cola).",
                "Informativa: documentar la curva de NEAT y la brecha con el gradiente (se espera que el gradiente gane en acierto).") as p:
        from gemelo import metricas
        from gemelo.neuro import neat, servicio
        n = servicio.Neuro(queue.Queue(), queue.Queue())
        n.preparar_pistas()
        n._renovar_pistas = lambda: None
        n.guardar = lambda: None
        curvas, finales = [], []
        for s in range(semillas):
            n.pob = neat.Poblacion(servicio.N_POBLACION, semilla=100 + s)
            t0, filas = time.time(), []
            for g in range(generaciones):
                filas.append(n.generacion()["historial"][-1])
            curvas.append([f["acierto"] for f in filas])
            mejor = max(filas, key=lambda f: f["acierto"])
            finales.append({"semilla": s, "acierto": mejor["acierto"], "pseudo_r2": mejor["pseudo_r2"], "r2_pose": mejor["r2_pose"],
                            "aguante": mejor["aguante_examen"], "ocultas": filas[-1]["ocultas"], "conexiones": filas[-1]["conexiones"],
                            "segundos": time.time() - t0})
            print(f"    NEAT semilla {s}: {mejor['acierto'] * 100:.1f} % · {time.time() - t0:.0f} s", flush=True)
        n.pool.shutdown()
        X = np.concatenate([q["X"][q["estable"]] for q in n.banco])
        y = np.concatenate([q["gesto"][q["estable"]] for q in n.banco]).astype(int)
        esc = StandardScaler().fit(X)
        base = {"logistica": [], "mlp": []}
        for s in range(semillas):
            for nombre, m in (("logistica", LogisticRegression(max_iter=2000, C=1.0, random_state=s)),
                              ("mlp", MLPClassifier((64, 32), max_iter=400, random_state=s, early_stopping=True))):
                m.fit(esc.transform(X), y)
                accs, probs_all, ys, ag = [], [], [], []
                for q in n.examen:
                    pr = m.predict_proba(esc.transform(q["X"]))
                    probs_all.append(pr[q["estable"]])
                    ys.append(q["gesto"][q["estable"]].astype(int))
                    ag.append(pista.choques(pr.argmax(1)[None], q)[0] * pista.DT)
                P, Y = np.concatenate(probs_all), np.concatenate(ys)
                base[nombre].append({"acierto": float(np.mean(P.argmax(1) == Y)), "pseudo_r2": metricas.pseudo_r2_mcfadden(Y, P, UNIFORME),
                                     "aguante": float(np.mean(ag))})
        a_neat = np.array([f["acierto"] for f in finales])
        a_mlp = np.array([b["acierto"] for b in base["mlp"]])
        w = stats.wilcoxon(a_mlp, a_neat, alternative="greater")
        largo = len(n.examen[0]["X"]) * pista.DT
        p.resultado = {"neat": finales, "logistica": base["logistica"], "mlp": base["mlp"], "wilcoxon_p": w.pvalue,
                       "neat_media": a_neat.mean(), "neat_sd": a_neat.std(), "mlp_media": a_mlp.mean(),
                       "logistica_media": np.mean([b["acierto"] for b in base["logistica"]]), "largo_pista_s": largo,
                       "generaciones": generaciones}
        p.cumple = None
        p.conclusion = (f"NEAT ({generaciones} generaciones): {a_neat.mean() * 100:.1f} ± {a_neat.std() * 100:.1f} % y aguanta "
                        f"{np.mean([f['aguante'] for f in finales]):.1f} s de {largo:.0f}; regresión logística "
                        f"{p.resultado['logistica_media'] * 100:.1f} %; MLP {a_mlp.mean() * 100:.1f} % (Wilcoxon p = {w.pvalue:.3f}). "
                        f"Las redes de NEAT terminan con ~{np.mean([f['conexiones'] for f in finales]):.0f} conexiones.")
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
        C = np.array(curvas) * 100
        g = np.arange(generaciones)
        ax[0].fill_between(g, C.min(0), C.max(0), color=ACENTO, alpha=0.15)
        ax[0].plot(g, np.median(C, 0), color=ACENTO, label="NEAT (mediana y rango de 6 semillas)")
        ax[0].axhline(a_mlp.mean() * 100, color=VERDES[2], ls="--", label="MLP con gradiente")
        ax[0].axhline(p.resultado["logistica_media"] * 100, color=GRIS, ls=":", label="regresión logística")
        ax[0].set_xlabel("generación")
        ax[0].set_ylabel("acierto en el examen (%)")
        ax[0].set_title("NEAT aprende desde cero")
        ax[0].legend(fontsize=7.5)
        nombres = ["NEAT", "Logística", "MLP"]
        ags = [[f["aguante"] for f in finales], [b["aguante"] for b in base["logistica"]], [b["aguante"] for b in base["mlp"]]]
        ax[1].bar(nombres, [np.mean(a) for a in ags], color=[ACENTO, GRIS, VERDES[2]])
        ax[1].errorbar(nombres, [np.mean(a) for a in ags], [np.std(a) for a in ags], fmt="none", color="#333", capsize=4)
        ax[1].axhline(largo, color=ROJO, ls="--", lw=1, label="pista completa")
        ax[1].set_ylabel("segundos sin chocar (examen)")
        ax[1].set_title("Aguante en la pista")
        ax[1].legend(fontsize=7.5)
        p.figuras.append(figura("C6", "neat_vs_gradiente", fig))


def c7(n_personas=70, epocas=6):
    with Prueba("C7", "Aprendizaje contrastivo con pocas etiquetas", "C. Modelos de IA",
                "Verificar que el codificador contrastivo (SimCLR, sin etiquetas) aprenda una representación de la señal cruda de los 10 "
                "electrodos que sirva cuando hay pocas etiquetas.",
                ["RF-06"],
                f"Se simulan {n_personas} personas con dos vistas físicas de cada ventana (el mismo músculo medido con los electrodos "
                f"recolocados). El codificador 1D-CNN se entrena {epocas} épocas con la pérdida NT-Xent sin usar ninguna etiqueta. "
                "Después se entrena una regresión logística con el 1 %, 5 %, 10 % y 100 % de las etiquetas de las personas de "
                "entrenamiento, sobre (a) el embedding contrastivo y (b) las 40 características clásicas, y se evalúa en personas nuevas "
                "(3 repeticiones).",
                "Con 1 % de etiquetas, el embedding contrastivo iguala o supera a las características clásicas.") as p:
        from gemelo import datos_emg as de
        from gemelo.contrastivo import EntrenadorContrastivo, embeddings
        rng = np.random.default_rng(2027)
        tareas = [(i, i % 5, 3, int(rng.integers(1 << 31))) for i in range(n_personas)]
        with ProcessPoolExecutor(3) as ex:
            partes = list(ex.map(de.simular_persona, tareas))
        d = {k: np.concatenate([q[k] for q in partes]) for k in partes[0]}
        d["transicion"] = d["progreso"] < 0.9
        d["split"] = de.dividir_por_persona(d["persona"], 2027)
        ent = EntrenadorContrastivo(d, dim=64, lote=512, n_monitor=1500, semilla=0)
        hist = ent.entrenar([("simclr", epocas, False)])
        tr = np.flatnonzero((d["split"] == 0) & ~d["transicion"])
        te = np.flatnonzero((d["split"] == 2) & ~d["transicion"])
        E = embeddings(ent.cod, d["x_a"], normalizar=False)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        F = np.log1p(ia.caracteristicas_lote(d["x_a"]))
        filas = []
        for frac in (0.01, 0.05, 0.10, 1.0):
            res = {"fraccion": frac}
            for nombre, Z in (("contrastivo", E), ("caracteristicas", F)):
                accs = []
                for r in range(3):
                    sel = np.random.default_rng(r).choice(tr, max(70, int(len(tr) * frac)), replace=False)
                    esc = StandardScaler().fit(Z[sel])
                    m = LogisticRegression(max_iter=2000).fit(esc.transform(Z[sel]), d["gesto"][sel])
                    accs.append(float(m.score(esc.transform(Z[te]), d["gesto"][te])))
                res[nombre], res[f"{nombre}_sd"] = np.mean(accs), np.std(accs)
            filas.append(res)
            print("   ", {k: round(v, 3) for k, v in res.items()}, flush=True)
        p.resultado = {"tabla": filas, "historial": [{k: h[k] for k in ("epoca", "perdida", "knn", "sonda_logistica", "r2_pose")}
                                                     for h in hist], "n_ventanas": len(d["gesto"])}
        p.cumple = bool(filas[0]["contrastivo"] >= filas[0]["caracteristicas"])
        p.conclusion = ("; ".join(f"{f['fraccion'] * 100:g} % de etiquetas: contrastivo {f['contrastivo'] * 100:.1f} % vs "
                                  f"características {f['caracteristicas'] * 100:.1f} %" for f in filas) + ".")
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3))
        fr = [f["fraccion"] * 100 for f in filas]
        ax[0].errorbar(fr, [f["contrastivo"] * 100 for f in filas], [f["contrastivo_sd"] * 100 for f in filas], fmt="o-", color=ACENTO,
                       capsize=3, label="embedding contrastivo (sin etiquetas)")
        ax[0].errorbar(fr, [f["caracteristicas"] * 100 for f in filas], [f["caracteristicas_sd"] * 100 for f in filas], fmt="s--",
                       color=GRIS, capsize=3, label="40 características clásicas")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("etiquetas usadas (%)")
        ax[0].set_ylabel("acierto en personas nuevas (%)")
        ax[0].set_title("¿Sirve aprender sin etiquetas?")
        ax[0].legend(fontsize=7.5)
        h = p.resultado["historial"]
        ax[1].plot([x["epoca"] for x in h], [x["knn"] * 100 for x in h], "o-", color=ACENTO, label="kNN sobre el embedding")
        ax[1].set_xlabel("época (sin etiquetas)")
        ax[1].set_ylabel("acierto kNN (%)")
        a2 = ax[1].twinx()
        a2.plot([x["epoca"] for x in h][1:], [x["perdida"] for x in h][1:], "s--", color=NARANJA, label="pérdida NT-Xent")
        a2.set_ylabel("pérdida")
        a2.grid(False)
        ax[1].set_title("Entrenamiento contrastivo")
        p.figuras.append(figura("C7", "contrastivo", fig))


def c8(semillas=(1, 2, 3), intentos=15000):
    with Prueba("C8", "Mano bebé: reproducibilidad del aprendizaje autónomo", "C. Modelos de IA",
                "Verificar que la mano bebé aprenda sola, desde cero y sin metas predeterminadas, de forma reproducible.",
                ["RF-09", "RF-13"],
                f"Se hace nacer la mano bebé {len(semillas)} veces con semillas distintas y se la deja aprender {intentos:,} intentos. "
                "Se registran el número de habilidades descubiertas, la finura, la cobertura de lo que su cuerpo puede hacer y el R² "
                "del control (pose lograda frente a la propuesta, en 300 poses de examen) y de su modelo del cuerpo.".replace(",", "."),
                "R² del control ≥ 0,90 en todas las semillas, el repertorio crece en todas y el coeficiente de variación del número de "
                "habilidades es menor a 25 %.") as p:
        from gemelo.libre import bebe, cuerpo, internet
        red, mapa = internet.cargar(), cuerpo.Mapa()
        filas, curvas = [], []
        for s in semillas:
            b = bebe.Bebe(semilla=s)
            t0 = time.time()
            for _ in range(intentos):
                b.intento(mapa, red)
            r = b.resumen(mapa)
            filas.append({"semilla": s, "habilidades": r["n_habilidades"], "finura": r["nivel"], "r2_control": r["r2_control"],
                          "r2_cuerpo": r["r2_cuerpo"], "cobertura": r["cobertura"], "parecidos_internet": r["parecidos"],
                          "primera_habilidad": b.habilidades[0]["intento"] if b.habilidades else None,
                          "segundos": time.time() - t0})
            curvas.append(np.array(b.curva))
            print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in filas[-1].items()}, flush=True)
        h = np.array([f["habilidades"] for f in filas])
        cv = h.std() / h.mean()
        p.resultado = {"semillas": filas, "cv_habilidades": cv, "intentos": intentos}
        p.cumple = bool(min(f["r2_control"] for f in filas) >= 0.9 and cv < 0.25 and all(f["habilidades"] > 0 for f in filas))
        p.conclusion = (f"Habilidades descubiertas: {h.mean():.0f} ± {h.std():.0f} (CV {cv * 100:.0f} %); primera habilidad en el intento "
                        f"{np.mean([f['primera_habilidad'] for f in filas]):.0f} en promedio; R² del control "
                        f"{min(f['r2_control'] for f in filas):.3f}–{max(f['r2_control'] for f in filas):.3f}; cobertura "
                        f"{np.mean([f['cobertura'] for f in filas]) * 100:.0f} %.")
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3))
        for k, c in enumerate(curvas):
            ax[0].plot(c[:, 0], c[:, 4], color=ACENTO, alpha=0.5 + 0.2 * k, label=f"semilla {semillas[k]}")
            ax[1].plot(c[:, 0], c[:, 1], color=ACENTO, alpha=0.5 + 0.2 * k)
            ax[1].plot(c[:, 0], c[:, 3], color=VERDES[2], ls="--", alpha=0.5 + 0.2 * k)
        ax[0].set_xlabel("intentos")
        ax[0].set_ylabel("habilidades descubiertas")
        ax[0].set_title("Repertorio (sin tope)")
        ax[0].legend(fontsize=7.5)
        ax[1].set_xlabel("intentos")
        ax[1].set_ylim(0, 1.02)
        ax[1].set_title("R² del control (azul) y cobertura (verde)")
        p.figuras.append(figura("C8", "bebe", fig))


if __name__ == "__main__":
    import sys
    pedidas = set(sys.argv[1:]) or {"C6", "C7", "C8"}
    if "C8" in pedidas:
        c8()
    if "C7" in pedidas:
        c7()
    if "C6" in pedidas:
        c6()
