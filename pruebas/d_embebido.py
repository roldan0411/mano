"""
Pruebas D · Embebido (ESP32), sin la placa
==========================================
  D1 el firmware compilado en el PC da lo mismo que Python
  D2 recursos: memoria y tiempo de inferencia
  D3 presupuesto de muestreo de los 10 electrodos
  D4 latencia de reconocimiento (con el filtro de confianza y mayoría del firmware)
"""

from comun import (ACENTO, GESTOS_ES, GRIS, MOTOR, NARANJA, RAIZ_PROYECTO, RESULTADOS, ROJO, VERDES, Prueba, campeona,
                   figura, plt, predecir)

import json
import re
import shutil
import subprocess
import sys

import numpy as np

import ia_mano as ia
from gemelo.neuro import neat, pista

CARPETA = RESULTADOS / "compilacion"
POR_VENTANA = 40 + 120 + 2 + 7


def campeon_neat():
    carpetas = sorted((MOTOR / "neuro" / "campeones").glob("c*"))
    c = carpetas[-1]
    ficha = json.loads((c / "ficha.json").read_text(encoding="utf-8"))
    g = neat.Genoma(0)
    for i, a, b, w, act in ficha["genoma"]:
        g.conexiones[int(i)] = [int(a), int(b), float(w), bool(act)]
        for n in (a, b):
            if n >= neat.PRIMER_OCULTO:
                g.ocultas.add(int(n))
    return g, ficha, c


def compilar():
    CARPETA.mkdir(parents=True, exist_ok=True)
    _, ficha_r, c_red = campeona()
    _, ficha_n, c_neat = campeon_neat()
    shutil.copy(c_red / "modelo_red_calibrado.h", CARPETA / "modelo_red_calibrado.h")
    shutil.copy(c_neat / "red_neat.h", CARPETA / "red_neat.h")
    exe = CARPETA / "firmware_pc.exe"
    r = subprocess.run([sys.executable, "-m", "ziglang", "cc", "-O2", "-std=c99", "-o", str(exe),
                        str(RAIZ_PROYECTO / "pruebas" / "firmware_pc.c"), f"-I{CARPETA}", f"-I{RAIZ_PROYECTO}", "-lm"],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return exe, ficha_r, ficha_n


def correr_firmware(exe, d):
    n = CARPETA / "datos"
    n.mkdir(exist_ok=True)
    d["x_cal"].astype(np.uint16).tofile(n / "cal.bin")
    d["x"].astype(np.uint16).tofile(n / "pista.bin")
    d["ventana_idx"].astype(np.int8).tofile(n / "vidx.bin")
    subprocess.run([str(exe), str(n), str(len(d["x_cal"])), str(len(d["x"]))], check=True)
    cal = np.fromfile(n / "cal_out.bin", np.float32)
    out = np.fromfile(n / "salida.bin", np.float32).reshape(-1, POR_VENTANA)
    tiempos = np.loadtxt(n / "tiempos.txt")
    return cal, out, tiempos


def d1_d2():
    exe, ficha_r, ficha_n = compilar()
    modelo, _, _ = campeona()
    g_neat, _, _ = campeon_neat()
    filas, todo = [], []
    for s in range(5):
        d = pista.recorrido(9500 + s, crudo=True)
        cal_c, out, tiempos = correr_firmware(exe, d)
        F_py = ia.caracteristicas_lote(d["x"])
        probs, _ = predecir(modelo, d["X"])
        z_py = g_neat.activar(d["X"].astype(np.float64))
        F_c, X_c, conf_c, g_c, z_c = out[:, :40], out[:, 40:160], out[:, 160], out[:, 161].astype(int), out[:, 162:]
        fila = {"persona": s, "ventanas": len(d["x"]),
                "dif_rel_caracteristicas": float(np.max(np.abs(F_c - F_py) / np.maximum(np.abs(F_py), 1))),
                "dif_calibracion": float(np.max(np.abs(cal_c - d["cal"]))),
                "dif_entradas": float(np.max(np.abs(X_c - d["X"]))),
                "mismo_gesto_red": float(np.mean(g_c == probs.argmax(1))),
                "dif_confianza": float(np.max(np.abs(conf_c - probs.max(1)))),
                "mismo_gesto_neat": float(np.mean(z_c.argmax(1) == z_py.argmax(1))),
                "dif_neat": float(np.max(np.abs(z_c - z_py)))}
        filas.append(fila)
        todo.append(tiempos)
        print("   ", {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in fila.items()}, flush=True)
    with Prueba("D1", "El firmware da lo mismo que Python", "D. Embebido (ESP32)",
                "Verificar que el procesamiento del firmware (características de los 10 electrodos, calibración, contexto) y las redes "
                "exportadas a C den exactamente lo mismo que el gemelo en Python.",
                ["RF-12", "RNF-05"],
                "Se compila en el PC (compilador zig cc, C99, -O2) un programa que incluye los mismos archivos del firmware: "
                "procesamiento_emg.h, modelo_red_calibrado.h de la campeona y red_neat.h del campeón NEAT. Se le pasan las ventanas "
                "crudas (cuentas del ADC) de la calibración y de la pista de 5 personas simuladas y se compara, ventana por ventana, "
                "con Python.",
                "Mismo gesto en el 100 % de las ventanas (red y NEAT); características con diferencia relativa < 1e-4; entradas, "
                "confianza y puntajes con diferencia < 1e-3.") as p:
        peor = {k: (min if k.startswith("mismo") else max)(f[k] for f in filas) for k in filas[0] if k not in ("persona", "ventanas")}
        p.resultado = {"personas": filas, "peor": peor, "campeona": ficha_r.get("campeona"), "campeon_neat": ficha_n.get("campeon"),
                       "compilador": "zig cc (ziglang " + subprocess.run([sys.executable, "-m", "ziglang", "version"], capture_output=True,
                                                                          text=True).stdout.strip() + "), C99, -O2"}
        p.cumple = bool(peor["mismo_gesto_red"] == 1 and peor["mismo_gesto_neat"] == 1 and peor["dif_rel_caracteristicas"] < 1e-4
                        and peor["dif_entradas"] < 1e-3 and peor["dif_confianza"] < 1e-3 and peor["dif_neat"] < 1e-3)
        p.conclusion = (f"{sum(f['ventanas'] for f in filas)} ventanas: mismo gesto en el {peor['mismo_gesto_red'] * 100:.0f} % (red) y "
                        f"{peor['mismo_gesto_neat'] * 100:.0f} % (NEAT); diferencias máximas: características {peor['dif_rel_caracteristicas']:.1e}, "
                        f"entradas {peor['dif_entradas']:.1e}, confianza {peor['dif_confianza']:.1e}, NEAT {peor['dif_neat']:.1e}.")
    t = np.mean(todo, 0)
    with Prueba("D2", "Recursos del ESP32: memoria y tiempo", "D. Embebido (ESP32)",
                "Estimar si los modelos caben y corren a tiempo en el ESP32 (240 MHz, 520 KB de SRAM, 4 MB de flash).",
                ["RF-12", "RNF-06"],
                "Se cuentan los parámetros y multiplicaciones-acumulaciones (MAC) de la campeona y del campeón NEAT, la memoria de "
                "los buffers del firmware, y se mide el tiempo por ventana en el PC. Para el ESP32 se estima con un costo conservador "
                "de 20 ciclos por operación de punto flotante (incluye lecturas desde la flash). La medición en la placa queda fuera "
                "del alcance de este documento.",
                "Estimación < 10 ms por ventana (cada 100 ms llega una) y modelo + buffers < 10 % de la memoria.") as p:
        params = int(sum(np.prod(w.shape) for w in modelo.trainable_weights))
        texto = (CARPETA / "modelo_red_calibrado.h").read_text(encoding="utf-8")
        capas = [tuple(map(int, m)) for m in re.findall(r"static const float RED_W\d+\[(\d+)\]\[(\d+)\]", texto)]
        mac_red = sum(a * b for a, b in capas)
        mac_neat = int(sum(1 for c in g_neat.conexiones.values() if c[3]))
        ops_car = 10 * 200 * 12
        ciclos = 20
        est = {k: v * ciclos / 240e6 * 1e3 for k, v in (("caracteristicas_ms", ops_car), ("red_ms", mac_red), ("neat_ms", mac_neat))}
        ram = 10 * 200 * 2 + (40 * 4 + 40 * 8 + 3 * 40 * 4) + 4 * (120 + max(b for _, b in capas) * 2)
        flash = 4 * (sum(a * b + b for a, b in capas) + 2 * 120)
        p.resultado = {"capas": capas, "parametros_keras": params, "mac_red": mac_red, "mac_neat": mac_neat, "ops_caracteristicas": ops_car,
                       "pc_us": {"caracteristicas": t[0], "red": t[1], "neat": t[2]}, "esp32_estimado": est,
                       "esp32_total_ms": sum(est.values()), "ram_bytes": ram, "flash_modelo_bytes": flash,
                       "ram_pct_520k": ram / 520e3 * 100, "flash_pct_4mb": flash / 4e6 * 100}
        p.cumple = bool(sum(est.values()) < 10 and ram / 520e3 < 0.1 and flash / 4e6 < 0.1)
        p.conclusion = (f"Red {'-'.join([str(capas[0][0])] + [str(b) for _, b in capas])}: {mac_red} MAC y {flash / 1024:.0f} KB de flash; "
                        f"NEAT: {mac_neat} MAC. Buffers: {ram / 1024:.1f} KB de RAM. Estimado en el ESP32: "
                        f"{sum(est.values()):.1f} ms por ventana (características {est['caracteristicas_ms']:.1f} ms, red {est['red_ms']:.1f} ms); "
                        f"medido en el PC: {t[0] + t[1]:.1f} µs.")


def d3():
    with Prueba("D3", "Presupuesto de muestreo de los 10 electrodos", "D. Embebido (ESP32)",
                "Verificar que el ESP32 alcance a leer los 10 electrodos a 1 kHz por el multiplexor y a procesar cada ventana a tiempo.",
                ["RF-12", "RNF-06"],
                "Cálculo del tiempo de cada milisegundo: 10 lecturas, cada una con 4 escrituras digitales para elegir el canal del "
                "CD74HC4067, 3 µs de asentamiento y la conversión del ADC. Como el tiempo de analogRead depende de la versión del núcleo "
                "de Arduino, se toma un rango de 10 a 40 µs. Cada 100 ms se suma el procesamiento estimado en D2.",
                "Ocupación de la CPU menor a 50 % en el peor caso.") as p:
        d2 = json.loads((RESULTADOS / "D2.json").read_text(encoding="utf-8"))["resultado"]
        filas = []
        for adc in (10, 20, 40):
            lect = 10 * (adc + 3 + 4 * 0.2)
            proc = d2["esp32_total_ms"] * 1000 / 100
            filas.append({"analogread_us": adc, "lectura_us_por_ms": lect, "procesamiento_us_por_ms": proc,
                          "ocupacion_pct": (lect + proc) / 1000 * 100})
        p.resultado = {"tabla": filas}
        p.cumple = bool(max(f["ocupacion_pct"] for f in filas) < 50)
        p.conclusion = "Ocupación de la CPU: " + "; ".join(f"{f['ocupacion_pct']:.0f} % con analogRead de {f['analogread_us']} µs" for f in filas) + \
            ". Si la medición en la placa da más, conviene el modo continuo del ADC (DMA) o un ADC externo."
        fig, ax = plt.subplots(figsize=(6, 2.8))
        x = np.arange(len(filas))
        ax.bar(x, [f["lectura_us_por_ms"] / 10 for f in filas], color=ACENTO, label="leer 10 electrodos")
        ax.bar(x, [f["procesamiento_us_por_ms"] / 10 for f in filas], bottom=[f["lectura_us_por_ms"] / 10 for f in filas], color=NARANJA,
               label="características + red")
        ax.axhline(50, color=ROJO, ls="--", lw=1, label="límite (50 %)")
        ax.set_xticks(x, [f"analogRead {f['analogread_us']} µs" for f in filas])
        ax.set_ylabel("CPU ocupada (%)")
        ax.set_title("Presupuesto de cada milisegundo")
        ax.legend(fontsize=7.5)
        p.figuras.append(figura("D3", "muestreo", fig, 10))


def _filtrar(probs, umbral=ia.UMBRAL_CONFIANZA):
    """Lo mismo que el firmware: si duda no cambia, y gana 2 de las 3 últimas predicciones."""
    ultima, hist, out = 0, [-1, -1, -1], []
    for pr in probs:
        k = int(np.argmax(pr))
        if pr[k] >= umbral:
            ultima = k
        hist = hist[1:] + [ultima]
        if hist[0] >= 0 and (hist[0] == hist[1] or hist[0] == hist[2]):
            out.append(hist[0])
        elif hist[1] >= 0 and hist[1] == hist[2]:
            out.append(hist[1])
        else:
            out.append(ultima)
    return np.array(out)


def d4():
    with Prueba("D4", "Latencia de reconocimiento", "D. Embebido (ESP32)",
                "Medir cuánto tarda la mano en mostrar el gesto desde que la persona empieza a hacerlo.",
                ["RNF-04"],
                "Con 12 personas simuladas (21 gestos cada una) se pasa la campeona por el mismo filtro del firmware (umbral de "
                "confianza 0,60 y mayoría de 3) y, para cada gesto, se mide desde el inicio de la contracción hasta la primera "
                "ventana en que la salida filtrada es el gesto correcto. Cada ventana de 200 ms se decide al terminar, cada 100 ms. "
                "No incluye el movimiento de los servos.",
                "Mediana menor a 300 ms (RNF-04).") as p:
        modelo, ficha, _ = campeona()
        lat, lat_sin = [], []
        for s in range(12):
            d = pista.recorrido(9600 + s)
            probs, _ = predecir(modelo, d["X"])
            fil = _filtrar(probs)
            crudo = probs.argmax(1)
            g = d["gesto"]
            inicios = np.r_[0, np.flatnonzero(np.diff(g) != 0) + 1]
            for i0, i1 in zip(inicios, np.r_[inicios[1:], len(g)]):
                for serie, destino in ((fil, lat), (crudo, lat_sin)):
                    ok = np.flatnonzero(serie[i0:i1] == g[i0])
                    destino.append(200 + 100 * ok[0] if len(ok) else np.nan)
        lat, lat_sin = np.array(lat, float), np.array(lat_sin, float)
        p.resultado = {"mediana_ms": np.nanmedian(lat), "p90_ms": np.nanpercentile(lat, 90), "nunca_pct": np.mean(np.isnan(lat)) * 100,
                       "sin_filtro_mediana_ms": np.nanmedian(lat_sin), "sin_filtro_p90_ms": np.nanpercentile(lat_sin, 90),
                       "n_gestos": len(lat), "campeona": ficha.get("campeona")}
        p.cumple = bool(np.nanmedian(lat) < 300)
        p.conclusion = (f"Con el filtro del firmware: mediana {np.nanmedian(lat):.0f} ms, p90 {np.nanpercentile(lat, 90):.0f} ms; sin el "
                        f"filtro de mayoría: mediana {np.nanmedian(lat_sin):.0f} ms. La ventana de 200 ms pone el piso: la primera decisión "
                        f"posible llega a los 200 ms y el filtro de mayoría agrega 100 ms.")
        fig, ax = plt.subplots(figsize=(6.5, 3))
        bins = np.arange(150, 1050, 100)
        ax.hist(lat_sin[~np.isnan(lat_sin)], bins, color=GRIS, alpha=0.6, label="sin filtro de mayoría")
        ax.hist(lat[~np.isnan(lat)], bins, color=ACENTO, alpha=0.7, label="como el firmware")
        ax.axvline(300, color=ROJO, ls="--", lw=1, label="objetivo (300 ms)")
        ax.set_xlabel("latencia desde el inicio del gesto (ms)")
        ax.set_ylabel("gestos")
        ax.set_title("Latencia de reconocimiento")
        ax.legend(fontsize=7.5)
        p.figuras.append(figura("D4", "latencia", fig, 11))


if __name__ == "__main__":
    d1_d2()
    d3()
    d4()
