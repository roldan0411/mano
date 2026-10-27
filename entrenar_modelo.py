"""
============================================================
 PRÓTESIS MIOELÉCTRICA CON RECONOCIMIENTO DE GESTOS POR IA
 Programa 2 de 3 (parte B) — Entrenamiento con datos REALES
------------------------------------------------------------
 Roldán Rodríguez — Ing. CCIA
============================================================

Qué hace:
  1. Lee datos_emg.csv (generado por capturar_datos.py con tu brazo).
  2. Corta la señal en ventanas de 200 ms y extrae 8 características
     (MAV, RMS, ZC y WL de cada canal), igual que el ESP32.
  3. Compara modelos con validación cruzada agrupada por ronda:
     KNN, SVM, Random Forest y la RED NEURONAL 8-32-16-7 (Keras).
     Si le pasas la red base del gemelo digital (--base), también
     compara "red desde cero" contra "red con base del gemelo".
  4. Guarda la comparación, la matriz de confusión y la curva de
     entrenamiento de la red.
  5. Guarda la red final en modelo_red.h5 y la exporta a C:
       modelo_red.h  -> red neuronal (modelo principal)
       modelo_rf.h   -> Random Forest (respaldo)

Del gemelo al modelo físico (transferencia de aprendizaje):
  python entrenar_modelo.py --base gemelo_salida/red_base_gemelo.h5
  La red arranca con lo que aprendió en el gemelo. Primero se
  recalcula su capa de normalización con tus datos reales (alinea
  las escalas del simulador y de tu circuito) y luego se ajustan
  los pesos con una tasa de aprendizaje baja.

Instalar:
  pip install tensorflow numpy pandas scikit-learn matplotlib h5py

Usar:
  python entrenar_modelo.py
  python entrenar_modelo.py --base gemelo_salida/red_base_gemelo.h5
  python entrenar_modelo.py --datos datos_emg.csv
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import ia_mano as ia

GESTOS = ia.GESTOS


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
def construir_dataset(df):
    X, y, grupos = [], [], []
    for (sesion, rep, gesto), bloque in df.groupby(["sesion", "repeticion", "gesto"]):
        if gesto not in GESTOS:
            continue
        bloque = bloque.sort_values("muestra")
        senal = bloque[ia.CANALES].to_numpy()               # columnas F1..F5, E1..E5
        v = ia.ventanear(senal)
        if len(v) == 0:
            continue
        X.append(ia.caracteristicas_lote(v))
        y += [GESTOS.index(gesto)] * len(v)
        grupos += [int(sesion) * 1000 + int(rep)] * len(v)
    return np.vstack(X), np.array(y), np.array(grupos)


# ------------------------------------------------------------------
# Red neuronal (Keras): desde cero o desde la base del gemelo
# ------------------------------------------------------------------
class RedKeras:
    """Envoltura con fit/predict para compararla igual que los otros modelos."""

    def __init__(self, base=None, epocas=300):
        self.base, self.epocas = base, epocas
        self.historia = None

    def fit(self, X, y):
        import keras
        X = X.astype(np.float32)
        if self.base:
            self.modelo = ia.cargar_red(self.base)
            ia.readaptar_normalizacion(self.modelo, X)   # escalas reales
            ia.compilar(self.modelo, tasa=3e-4)           # ajuste fino suave
        else:
            self.modelo = ia.crear_red(X_adaptar=X)
        parar = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                              restore_best_weights=True)
        self.historia = self.modelo.fit(X, y, epochs=self.epocas, batch_size=64,
                                        validation_split=0.15, shuffle=True,
                                        callbacks=[parar], verbose=0).history
        return self

    def predict(self, X):
        return self.modelo(X.astype(np.float32), training=False).numpy().argmax(1)


# ------------------------------------------------------------------
# Exportar el RANDOM FOREST a C (respaldo)
# ------------------------------------------------------------------
def exportar_random_forest(rf, ruta):
    lineas = [
        "/*",
        f" * modelo_rf.h — Random Forest de {len(rf.estimators_)} árboles",
        " * Exportado automáticamente por entrenar_modelo.py. NO editar a mano.",
        " * Roldán Rodríguez — Ing. CCIA",
        " */",
        "#pragma once",
        "#include <math.h>",
        "",
        f"#define VENTANA {ia.VENTANA}",
        f"#define PASO {ia.PASO}",
        f"#define UMBRAL_ZC {ia.UMBRAL_ZC}",
        f"#define NUM_CARACTERISTICAS {rf.n_features_in_}",
        f"#define NUM_CLASES {len(rf.classes_)}",
        "",
        "static const char* const NOMBRES_GESTOS[NUM_CLASES] = {",
        "  " + ", ".join(f'"{GESTOS[c]}"' for c in rf.classes_),
        "};",
        "",
    ]
    for t, arbol in enumerate(rf.estimators_):
        tr = arbol.tree_
        hojas = {}

        def nodo(i, sangria):
            pad = "  " * sangria
            if tr.children_left[i] == -1:  # es una hoja
                hojas[i] = len(hojas)
                return [f"{pad}return {hojas[i]};"]
            f, thr = int(tr.feature[i]), float(tr.threshold[i])
            return ([f"{pad}if ((double)x[{f}] <= {thr!r}) {{"]
                    + nodo(tr.children_left[i], sangria + 1)
                    + [f"{pad}}} else {{"]
                    + nodo(tr.children_right[i], sangria + 1)
                    + [f"{pad}}}"])

        cuerpo = nodo(0, 1)
        filas = []
        for i, _ in sorted(hojas.items(), key=lambda kv: kv[1]):
            v = tr.value[i][0]
            v = v / v.sum()
            filas.append("  {" + ", ".join(f"{p:.6f}f" for p in v) + "}")
        lineas.append(f"static const float HOJAS_{t}[{len(hojas)}][NUM_CLASES] = {{")
        lineas.append(",\n".join(filas))
        lineas.append("};")
        lineas.append(f"static int arbol_{t}(const float *x) {{")
        lineas += cuerpo
        lineas.append("}")
        lineas.append("")

    n = len(rf.estimators_)
    lineas.append("/* Devuelve el gesto y su confianza (fracción promedio de votos, 0 a 1) */")
    lineas.append("static int predecir_gesto(const float *x, float *confianza) {")
    lineas.append("  float p[NUM_CLASES] = {0};")
    for t in range(n):
        lineas.append(f"  {{ int h = arbol_{t}(x); "
                      f"for (int k = 0; k < NUM_CLASES; k++) p[k] += HOJAS_{t}[h][k]; }}")
    lineas.append("  int mejor = 0;")
    lineas.append("  for (int k = 1; k < NUM_CLASES; k++) if (p[k] > p[mejor]) mejor = k;")
    lineas.append(f"  if (confianza) *confianza = p[mejor] / {n}.0f;")
    lineas.append("  return mejor;")
    lineas.append("}")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


def crear_rf():
    return RandomForestClassifier(n_estimators=25, max_depth=10, min_samples_leaf=2, random_state=42)


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="datos_emg.csv")
    ap.add_argument("--base", default=None,
                    help="red .h5 del gemelo digital para arrancar desde ella")
    args = ap.parse_args()

    import keras
    keras.utils.set_random_seed(42)

    df = pd.read_csv(args.datos)
    X, y, grupos = construir_dataset(df)
    n_grupos = len(np.unique(grupos))
    print(f"Dataset: {len(X)} ventanas, {n_grupos} rondas, {len(np.unique(y))} gestos")
    if n_grupos < 4:
        raise SystemExit("Muy pocos datos: graba al menos 10 rondas "
                         "(recomendado 30) con capturar_datos.py.")
    if args.base:
        print(f"Red base del gemelo: {args.base}")
        ia.cargar_red(args.base)  # verifica que sirva antes de empezar

    modelos = {
        "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "Random Forest": crear_rf,
        "Red neuronal": lambda: RedKeras(),
    }
    if args.base:
        modelos["Red + base gemelo"] = lambda: RedKeras(base=args.base)

    # 1) Comparación con validación cruzada agrupada (hasta 5 pliegues)
    gkf = GroupKFold(n_splits=min(5, n_grupos))
    resultados = {}
    for nombre, fabrica in modelos.items():
        acc = []
        for tr, te in gkf.split(X, y, grupos):
            modelo = fabrica().fit(X[tr], y[tr])
            acc.append(accuracy_score(y[te], modelo.predict(X[te])))
        resultados[nombre] = (np.mean(acc), np.std(acc))
        print(f"  {nombre:18s} precisión = {np.mean(acc)*100:5.1f} % ± {np.std(acc)*100:4.1f}")

    plt.figure(figsize=(8, 4))
    nombres = list(resultados)
    medias = [resultados[n][0] * 100 for n in nombres]
    desv = [resultados[n][1] * 100 for n in nombres]
    colores = ["#B4B2A9", "#B4B2A9", "#1D9E75", "#534AB7", "#D85A30"][:len(nombres)]
    plt.bar(nombres, medias, yerr=desv, capsize=6, color=colores)
    for i, m in enumerate(medias):
        plt.text(i, m + 2, f"{m:.1f} %", ha="center")
    plt.ylim(0, 105)
    plt.ylabel("Precisión (%)")
    plt.title("Comparación de modelos (validación cruzada agrupada)")
    plt.tight_layout()
    plt.savefig("comparacion_modelos.png", dpi=150)
    plt.close()

    usar_base = bool(args.base) and resultados["Red + base gemelo"][0] >= resultados["Red neuronal"][0]
    fabrica_red = (lambda: RedKeras(base=args.base)) if usar_base else (lambda: RedKeras())
    if args.base:
        print("\nSe usa la red " + ("CON base del gemelo." if usar_base else
                                    "desde cero (con tus datos rindió mejor)."))

    # 2) Evaluación de la red elegida con rondas nunca vistas
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, grupos))
    red = fabrica_red().fit(X[tr], y[tr])
    pred = red.predict(X[te])
    print("\nRed neuronal en rondas de prueba:")
    print(classification_report(y[te], pred, labels=range(len(GESTOS)),
                                target_names=GESTOS, zero_division=0))

    cm = confusion_matrix(y[te], pred, labels=range(len(GESTOS)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=GESTOS).plot(
        ax=ax, cmap="Purples", colorbar=False, xticks_rotation=45)
    plt.title("Matriz de confusión — Red neuronal")
    plt.tight_layout()
    plt.savefig("matriz_confusion.png", dpi=150)
    plt.close()

    h = red.historia
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(h["loss"], color="#534AB7", label="pérdida de entrenamiento")
    ax1.plot(h["val_loss"], color="#534AB7", ls="--", label="pérdida de validación")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida (entropía cruzada)", color="#534AB7")
    ax1.legend(loc="center right")
    ax2 = ax1.twinx()
    ax2.plot(np.array(h["val_accuracy"]) * 100, color="#1D9E75")
    ax2.set_ylabel("Precisión de validación (%)", color="#1D9E75")
    ax2.set_ylim(0, 105)
    fig.suptitle("Entrenamiento de la red neuronal" + (" (desde la base del gemelo)" if usar_base else ""))
    fig.tight_layout()
    fig.savefig("curva_entrenamiento.png", dpi=150)
    plt.close(fig)

    rf = crear_rf().fit(X[tr], y[tr])
    imp = sorted(zip(ia.NOMBRES_CARAC, rf.feature_importances_), key=lambda t: -t[1])
    with open("reporte_modelo.txt", "w", encoding="utf-8") as f:
        f.write("Comparación de modelos (precisión media ± desviación)\n")
        for n, (m, s) in resultados.items():
            f.write(f"  {n}: {m*100:.1f} % ± {s*100:.1f}\n")
        f.write(f"\nRed neuronal: 8-32-16-{len(GESTOS)}, {ia.contar_parametros(red.modelo)} parámetros, "
                f"{len(h['loss'])} épocas" + (", con base del gemelo\n" if usar_base else "\n"))
        f.write("Reporte de la red neuronal (rondas de prueba)\n")
        f.write(classification_report(y[te], pred, labels=range(len(GESTOS)),
                                      target_names=GESTOS, zero_division=0))
        f.write("\nImportancia de características (Random Forest)\n")
        for n, v in imp:
            f.write(f"  {n}: {v:.3f}\n")

    # 3) Modelos finales con TODOS los datos y exportación al ESP32
    red_final = fabrica_red().fit(X, y)
    rf_final = crear_rf().fit(X, y)
    ia.guardar_red(red_final.modelo, "modelo_red.h5",
                   {"origen": "datos reales" + (" + base del gemelo" if usar_base else ""),
                    "ventanas": int(len(X))})
    carpeta = "protesis_tiempo_real" if os.path.isdir("protesis_tiempo_real") else "."
    ia.exportar_red_c(red_final.modelo, os.path.join(carpeta, "modelo_red.h"))
    exportar_random_forest(rf_final, os.path.join(carpeta, "modelo_rf.h"))
    print(f"\nRed guardada en modelo_red.h5 ({ia.contar_parametros(red_final.modelo)} parámetros)")
    print(f"Modelos exportados a {carpeta}: modelo_red.h y modelo_rf.h")
    print("Gráficas: comparacion_modelos.png, matriz_confusion.png, curva_entrenamiento.png")
    print("Reporte: reporte_modelo.txt")
    print("Siguiente paso: abre protesis_tiempo_real.ino y súbelo al ESP32.")


if __name__ == "__main__":
    main()
