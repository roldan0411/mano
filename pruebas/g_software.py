"""
Pruebas G · Software y aplicaciones
===================================
  G1 persistencia (cerrar y reabrir)       G2 estabilidad en operación prolongada
  G3 portátil: batería y poca RAM          G4 pruebas automáticas (pytest)
"""

from comun import ACENTO, AZULES, GRIS, NARANJA, RAIZ_PROYECTO, RESULTADOS, VERDES, Prueba, figura, plt

import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def _pytest():
    xml = RESULTADOS / "pytest.xml"
    subprocess.run([sys.executable, "-m", "pytest", str(RAIZ_PROYECTO / "tests"), "-q", "-p", "no:cacheprovider",
                    f"--junitxml={xml}"], cwd=RAIZ_PROYECTO, capture_output=True)
    casos = []
    for c in ET.parse(xml).getroot().iter("testcase"):
        falla = c.find("failure") is not None or c.find("error") is not None
        casos.append({"archivo": c.get("classname", "").split(".")[-1], "prueba": c.get("name"), "segundos": float(c.get("time", 0)),
                      "paso": not falla})
    return casos


def g1_g3_g4():
    casos = _pytest()
    g1 = [c for c in casos if c["prueba"].startswith("test_g1")]
    g3 = [c for c in casos if c["prueba"].startswith("test_g3")]
    with Prueba("G1", "Persistencia: cerrar y reabrir", "G. Software",
                "Verificar que todo lo aprendido se guarde y se recupere igual al reabrir (la falla de «todo en cero» del 26 de octubre).",
                ["RF-11"],
                "Pruebas automáticas que guardan y vuelven a cargar: una red de la competencia (mismas predicciones, métricas y pasos), "
                "la mano bebé (mismos intentos y habilidades; una versión vieja se aparta como respaldo y nace de cero), la población "
                "NEAT (mismas salidas) y el límite de disco del almacén (regresión del error que borraba la carpeta de lotes y "
                "tumbaba el motor). Además, en la aplicación se verificó a mano que el motor retoma la generación donde iba.",
                "Todas las pruebas pasan.") as p:
        p.resultado = {"casos": g1}
        p.cumple = all(c["paso"] for c in g1)
        p.conclusion = f"{sum(c['paso'] for c in g1)} de {len(g1)} pruebas de persistencia pasan."
    with Prueba("G3", "Portátil: batería y poca RAM", "G. Software",
                "Verificar que el motor decida bien cuándo pausarse: en batería con GPU (Windows puede apagarla) o con poca RAM; y que en "
                "batería sin GPU siga entrenando en la CPU.",
                ["RNF-08"],
                "Se prueba la función motivo_pausa del motor con los 4 casos: cargador y RAM bien; batería con GPU; batería sin GPU; poca RAM.",
                "Las 4 decisiones son las esperadas.") as p:
        p.resultado = {"casos": g3}
        p.cumple = all(c["paso"] for c in g3)
        p.conclusion = f"{sum(c['paso'] for c in g3)} de {len(g3)} casos correctos."
    with Prueba("G4", "Pruebas automáticas (pytest)", "G. Software",
                "Mantener una batería de pruebas que se corra con un solo comando antes de cada cambio.",
                ["RNF-09"],
                "python -m pytest tests: procesamiento de la señal (características, contexto, calibración, acople, cadena de "
                "adquisición), NEAT (redes sin ciclos, innovaciones, cruce, exportación a C, especies), la pista (choques) y "
                "persistencia/operación (G1 y G3).",
                "Todas las pruebas pasan.") as p:
        por_archivo = {}
        for c in casos:
            a = por_archivo.setdefault(c["archivo"], {"total": 0, "pasan": 0})
            a["total"] += 1
            a["pasan"] += c["paso"]
        p.resultado = {"total": len(casos), "pasan": sum(c["paso"] for c in casos), "por_archivo": por_archivo,
                       "segundos": sum(c["segundos"] for c in casos), "casos": casos}
        p.cumple = all(c["paso"] for c in casos)
        p.conclusion = f"{p.resultado['pasan']} de {len(casos)} pruebas pasan en {p.resultado['segundos']:.0f} s."


def g2():
    with Prueba("G2", "Estabilidad en operación prolongada", "G. Software",
                "Verificar que las tres aplicaciones abiertas a la vez (competencia en la GPU, neuroevolución y mano bebé) no pierdan "
                "memoria ni se cuelguen.",
                ["RNF-07"],
                "Durante 60 minutos se anotó cada minuto la memoria (RSS) de cada proceso, la memoria de la GPU y el avance de cada "
                "aplicación (generaciones del motor y de la neuroevolución, intentos de la mano bebé). Se ajusta una recta a la memoria "
                "de la segunda media hora (después del arranque) para estimar fugas.",
                "Crecimiento de memoria < 2 MB/min por proceso en la segunda media hora y las tres aplicaciones avanzan todo el tiempo.") as p:
        df = pd.read_csv(RESULTADOS / "G2_estabilidad.csv")
        tot = df.groupby(["minuto", "rol"])["rss_mb"].sum().unstack()
        pend = {}
        for rol in tot.columns:
            s = tot[rol].dropna()
            s2 = s[s.index >= 30]
            pend[rol] = float(np.polyfit(s2.index, s2.values, 1)[0]) if len(s2) > 5 else None
        av = df.groupby("minuto")[["gen_motor", "gen_neuro", "intentos_bebe", "gpu_mb"]].first()
        avanza = {k: bool(av[k].dropna().diff().fillna(0).ge(0).all() and av[k].dropna().iloc[-1] > av[k].dropna().iloc[0])
                  for k in ("gen_motor", "gen_neuro", "intentos_bebe")}
        # la mano bebé estuvo en pausa en esa hora (el botón mostraba «Reanudar» y su archivo no se escribió): no se evalúa
        evaluadas = ("gen_motor", "gen_neuro")
        p.resultado = {"pendiente_mb_por_min": pend, "avanza": avanza, "minutos": int(df["minuto"].max()),
                       "memoria_final_mb": {k: float(tot[k].dropna().iloc[-1]) for k in tot.columns},
                       "gpu_mb_min_max": [float(av["gpu_mb"].min()), float(av["gpu_mb"].max())],
                       "avance": {k: [float(av[k].dropna().iloc[0]), float(av[k].dropna().iloc[-1])] for k in avanza},
                       "mano_bebe": "en pausa durante la medición (no se evalúa su avance)",
                       "hallazgo": "Al terminar se encontraron 4 procesos de simulación huérfanos de un motor cerrado a la fuerza. "
                                   "Se corrigió: cada proceso vigila a su padre y se cierra solo si muere (config.pool_vigilado); "
                                   "lo verifica la prueba automática test_g2_sin_procesos_huerfanos."}
        p.cumple = bool(all(v is None or v < 2 for v in pend.values()) and all(avanza[k] for k in evaluadas))
        p.conclusion = ("Memoria en la segunda media hora: " + "; ".join(f"{k} {v:+.2f} MB/min" for k, v in pend.items() if v is not None)
                        + f". Avance en 60 min: motor {p.resultado['avance']['gen_motor'][0]:.0f}→{p.resultado['avance']['gen_motor'][1]:.0f} "
                        f"generaciones, neuroevolución {p.resultado['avance']['gen_neuro'][0]:.0f}→{p.resultado['avance']['gen_neuro'][1]:.0f}, "
                        "la mano bebé estuvo en pausa. Se encontraron y corrigieron procesos huérfanos al cerrar el motor a la fuerza.")
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
        colores = [ACENTO, VERDES[2], NARANJA, GRIS, AZULES[0]]
        for k, rol in enumerate(tot.columns):
            ax[0].plot(tot.index, tot[rol], color=colores[k % 5], label=rol)
        ax[0].set_xlabel("minuto")
        ax[0].set_ylabel("memoria (MB)")
        ax[0].set_title("Memoria de los procesos")
        ax[0].legend(fontsize=7)
        for k, (c, n) in enumerate((("gen_motor", "generaciones del motor"), ("gen_neuro", "generaciones NEAT"))):
            s = av[c].dropna()
            ax[1].plot(s.index, (s - s.iloc[0]) / max(s.iloc[-1] - s.iloc[0], 1), color=colores[k], label=n)
        ax[1].set_xlabel("minuto")
        ax[1].set_ylabel("avance (0 = inicio, 1 = final)")
        ax[1].set_title("La competencia y la neuroevolución avanzan")
        ax[1].legend(fontsize=7)
        p.figuras.append(figura("G2", "estabilidad", fig))


if __name__ == "__main__":
    g1_g3_g4()
    g2()
