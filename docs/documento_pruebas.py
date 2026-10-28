"""Documentación de pruebas técnicas (plan, trazabilidad, fichas, resultados y hallazgos)."""

from datos_documentos import ORDEN, R, REQUERIMIENTOS, RESULTADOS, es, estado, n, pc, trazabilidad
from word import Documento

GRUPOS = {"A": "Gemelo digital", "B": "Procesamiento de la señal", "C": "Modelos de IA", "D": "Embebido (ESP32), sin la placa",
          "G": "Software y aplicaciones"}

ANALISIS = {
    "A1": "El acople reproduce la anatomía: los electrodos palmares captan flexores y los dorsales extensores, con FCU, FDS, FCR, ECR, "
          "EDC y ECU como los más captados. Los músculos profundos (FDP, FPL, EIP, EPL) llegan atenuados, como en la realidad. Un giro de "
          "15° cambia una cuarta parte del acople: por eso la calibración de la sesión es necesaria.",
    "A2": "La ganancia simulada coincide con la fórmula del fabricante. La gráfica de saturación muestra el margen del diseño: con RG = 470 Ω "
          "(G = 106) el AD620 tolera unos 58 mV de desfase continuo, varias veces el desfase típico de electrodos de gel.",
    "A3": "Los cortes medidos están dentro del 5 % del diseño. La pequeña diferencia en el corte bajo se debe al segundo polo pasa-altos "
          "(Rth·C2 = 3,1 Hz) y en el corte alto a la discretización bilineal a 4 kHz.",
    "A4": "**Hallazgo.** La mediana está en el rango esperado, pero un tercio de las señales supera los 150 Hz: corresponden a los músculos de la "
          "muñeca, simulados con banda de 45–300 Hz. **Acción recomendada:** bajar esas bandas (por ejemplo a 30–200 Hz) y repetir A4 y A6. "
          "Para el reconocimiento el efecto es menor, porque las características son relativas a la calibración.",
    "A5": "Los electrodos vecinos se parecen bastante (correlación de envolventes 0,86), esperable en electrodos monopolares tan cercanos; "
          "flexores y extensores opuestos son casi independientes. La redundancia entre vecinos es la que permite tolerar un electrodo suelto "
          "si se entrena para eso (C5).",
    "A6": "**Hallazgo.** La forma de la señal coincide con el EMG real (RMS/MAV a 0,2 %, curtosis parecida), pero el espectro del gemelo está "
          "corrido hacia arriba: en la misma banda de análisis su frecuencia mediana es mayor que la de la pulsera Myo. Confirma A4 desde "
          "datos reales. Limitación: la Myo muestrea a ~200 Hz y tiene su propio filtrado, así que la comparación se limita a 20–90 Hz.",
    "B1": "La implementación es correcta. El criterio inicial (ZC exacto, WL < 2 %) no consideraba el muestreo: en el borde de una ventana se "
          "puede ganar o perder un cruce, y con pocas muestras por ciclo la variación total discreta es menor que la continua. Se corrigió el "
          "criterio con la cota teórica, y así se registra en la ficha.",
    "B2": "La calibración de 20 s mejora el acierto en personas nuevas en todas las particiones. Es la razón por la que el firmware calibra al encender.",
    "B3": "Entre 100 y 200 ms el acierto es prácticamente el mismo; 50 ms pierde ~1 punto. Una ventana de 150 ms bajaría la latencia "
          "25 ms sin perder acierto: es una opción para mejorar D4.",
    "C1": "Pasar de 2 a 10 electrodos mejora las tres métricas en todas las particiones y de forma significativa. El mayor salto está en la pose "
          "(R²), porque con 10 electrodos se distinguen mejor los músculos de cada dedo.",
    "C2": "Unos pocos electrodos bien ubicados capturan casi toda la información (con LDA), pero en un brazalete que puede girar «bien "
          "ubicado» cambia en cada sesión: los 10 dan cobertura y redundancia. El orden de importancia sirve para decidir dónde reforzar el contacto.",
    "C3": "La campeona supera ampliamente el requerimiento con personas que nunca vio, incluidas las de nivel 5 (más ruido, interferencia, "
          "fatiga y desplazamiento). Los pocos errores se reparten entre puño, pinza y abierta (recall de 99,2 a 99,5 %); sin las ventanas de "
          "transición entre gestos el acierto apenas sube, a 99,8 %.",
    "C4": "La red tolera bien el ruido, la fatiga y el corrimiento a lo largo del brazo. Lo que más la afecta es que los electrodos giren "
          "después de calibrar (77 % a 30°); recalibrar lo resuelve por completo. Para el brazalete: calibrar al ponerlo y evitar que gire.",
    "C5": "**Hallazgo principal.** Si un electrodo se despega, la campeona cae hasta 27 %: nunca vio esa situación. Un MLP igual, entrenado "
          "simulando electrodos despegados al azar, se mantiene en 98 % con cualquier electrodo suelto y sin perder acierto cuando todos "
          "funcionan. **Acción recomendada:** incorporar ese aumento al entrenamiento del motor y de la neuroevolución (clave para electrodos secos).",
    "C6": "NEAT aprende de verdad desde cero —de redes casi vacías a ~90 % en 150 generaciones, con redes de menos de 100 conexiones—, pero "
          "sigue por debajo del gradiente con los mismos datos (diferencia significativa). Es valiosa por su red mínima y su "
          "interpretabilidad; para el uso diario, la red de la competencia es la recomendada.",
    "C7": "**Hallazgo.** Con 10 electrodos, las 40 características clásicas ya separan muy bien los gestos y el codificador contrastivo "
          "(6 épocas, 70 personas, sin etiquetas) no las supera ni con pocas etiquetas. **Acción recomendada:** más épocas y más datos, o "
          "combinar el embedding con las características (como en las cabezas multitarea del notebook) antes de descartarlo.",
    "C8": "La mano bebé aprende sola en las tres semillas, con un repertorio de tamaño parecido y control casi perfecto de su cuerpo. La "
          "variación entre semillas es la esperada en un aprendizaje por curiosidad.",
    "C9": "Con 10 electrodos la competencia supera el 99 % de acierto en pocas generaciones; con 2 electrodos la mejor marca fue 97,6 % "
          "después de 810 generaciones. Las caídas periódicas del promedio corresponden a la red nueva al azar que entra cada 12 generaciones.",
    "D1": "El firmware reproduce exactamente al gemelo: el mismo gesto en todas las ventanas, con diferencias del orden del redondeo de punto "
          "flotante. La calibración y el contexto se verificaron por separado. Diferencia de diseño documentada: en la pista de prueba el "
          "contexto se reinicia en cada gesto, como en Python; el firmware real lo mantiene continuo.",
    "D2": "Los modelos son pequeños para el ESP32: decenas de KB de flash y pocos KB de RAM, y una ventana se procesa en una fracción de los "
          "100 ms disponibles. Es una estimación conservadora; la medición en la placa queda fuera del alcance de este documento.",
    "D3": "Aun en el peor caso supuesto, el ESP32 queda con más de la mitad de la CPU libre. Si en la placa analogRead resulta más lento, el "
          "modo continuo del ADC con DMA o un ADC externo resuelven el presupuesto.",
    "D4": "**Hallazgo.** La mediana es exactamente 300 ms: el límite exigido. La ventana de 200 ms pone el piso y el filtro de mayoría agrega "
          "100 ms. Farrell y Weir (2007) ubican el retardo óptimo de una prótesis mioeléctrica entre 100 y 125 ms. **Acción "
          "recomendada:** ventana de 150 ms (B3) o mayoría de 2 decisiones; volver a medir.",
    "G1": "Todo lo aprendido se recupera igual al reabrir. Incluye la regresión del error del 26 de octubre (el límite de disco borraba la "
          "carpeta de lotes y el motor se caía).",
    "G2": "Sin fugas de memoria en una hora con las tres aplicaciones abiertas, y la competencia y la neuroevolución avanzaron todo el tiempo. "
          "La mano bebé estuvo en pausa por decisión del usuario. **Hallazgo corregido:** quedaban procesos de simulación huérfanos al "
          "cerrar el motor a la fuerza; ahora cada proceso vigila a su padre (prueba automática test_g2_sin_procesos_huerfanos).",
    "G3": "El motor se pausa en batería solo si está usando la GPU (Windows puede apagarla) y con poca RAM; en batería sin GPU sigue en la CPU.",
    "G4": "La batería completa corre en segundos y debe correrse antes de cada cambio.",
}

FUERA = [
    ["E1", "Ruido de entrada referido a la entrada", "Entradas de cada canal en corto; medir µV rms en la salida / ganancia.", "< 5 µV rms"],
    ["E2", "Rechazo de modo común y de 60 Hz", "Señal común de 1 V, 60 Hz en ambas entradas; medir salida.", "CMRR según hoja del AD620"],
    ["E3", "Servos: posición y tiempo de respuesta", "Comandar 0–100 % y medir ángulo y tiempo con video a 240 fps.", "Error < 5°; < 300 ms"],
    ["E4", "Consumo y autonomía", "Medir corriente con la mano en uso cíclico de gestos.", "≥ 4 h de uso"],
    ["F1", "Dataset propio con personas", "Protocolo fijo de gestos con consentimiento informado.", "≥ 10 personas"],
    ["F2", "Prueba TAC (Target Achievement Control)", "Simon et al. (2011): alcanzar posturas objetivo en tiempo real.", "Tasa de logro y tiempo"],
    ["F3", "Prueba funcional", "Agarrar y mover objetos de distinta forma y peso.", "Tasa de éxito"],
    ["F4", "Usabilidad", "Cuestionario SUS después de usar la prótesis.", "SUS ≥ 68"],
]


def construir(ruta):
    D = Documento("Documentación de pruebas técnicas", "Plan, especificación y resultados de las pruebas del gemelo digital, los modelos y el firmware",
                  "DP-GDM-01")
    D.tabla(["Versión", "Fecha", "Cambios"], [["1.0", "28/10/2026", "Primera ejecución completa: 26 pruebas y 24 pruebas automáticas."]],
            "Control de versiones.", anchos=[2, 3, 11.5])
    D.indice()

    D.h1("Introducción")
    D.h2("Propósito")
    D.p("Este documento planifica, especifica y registra las pruebas técnicas del gemelo digital de la prótesis mioeléctrica, de sus modelos de "
        "inteligencia artificial y de su firmware, y relaciona cada prueba con los requerimientos del **Documento técnico (DT-GDM-01)**. Sigue la "
        "estructura de la norma ISO/IEC/IEEE 29119-3: plan, especificación de cada prueba, resultados y hallazgos.")
    D.h2("Alcance")
    D.p("Se ejecutaron 26 pruebas en 5 grupos y una batería de 24 pruebas automáticas. **Quedan fuera del alcance**, por decisión del proyecto, "
        "las pruebas del hardware físico y las pruebas con personas reales; sus protocolos se dejan especificados en la sección 7.")
    D.tabla(["Grupo", "Qué verifica", "Pruebas"],
            [[g, d, ", ".join(k for k in ORDEN if k.startswith(g))] for g, d in GRUPOS.items()], "Grupos de pruebas.", anchos=[1.6, 7, 7.9])

    D.h1("Plan de pruebas")
    D.h2("Enfoque")
    D.vinetas([
        "**Pruebas de verificación del modelo** (A): cada modelo físico del gemelo contra su fórmula, su diseño o datos reales.",
        "**Pruebas de componentes** (B, D): el procesamiento contra referencias analíticas y el firmware contra Python, bit a bit.",
        "**Pruebas de desempeño** (C): siempre con **personas que el modelo no vio** (separación por persona), con varias particiones o "
        "semillas y prueba estadística cuando se comparan alternativas. Toda prueba de reconocimiento reporta R².",
        "**Pruebas de sistema** (G): persistencia, estabilidad y operación en la portátil, más la batería automática.",
        "Cada prueba guarda un registro (JSON) con su resultado y sus figuras en pruebas/resultados/: este documento se genera a partir de ellos.",
    ])
    D.h2("Criterios generales")
    D.vinetas([
        "**Cumple / No cumple**: según el criterio de aceptación de cada ficha, definido antes de ejecutar.",
        "**Informativa**: pruebas que caracterizan el sistema sin un umbral (por ejemplo, curvas de convergencia).",
        "Un **No cumple** no se oculta ni se corrige cambiando el criterio: se documenta como hallazgo con su acción. La única corrección de "
        "criterio (B1) se registra con su justificación.",
    ])
    D.h2("Entorno")
    D.tabla(["Elemento", "Detalle"], [
        ["Equipo", "Portátil con Intel Core 5 210H, 15,6 GB de RAM, GPU NVIDIA GeForce RTX 4050 Laptop; Windows 11"],
        ["Software", "Python 3.12.10, PyTorch 2.14.1 (CUDA 12.6), Keras 3.15.1, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, PySide6 6.12.0"],
        ["Compilador C", "zig cc (paquete ziglang 0.16.0), C99, -O2, para compilar el firmware en el PC"],
        ["Datos", "Lotes del motor (personas virtuales de 5 niveles), examen fijo de 20 personas, pistas de neuroevolución, dataset UCI «EMG data for gestures»"],
        ["Modelos evaluados", f"Campeona #{R['C3']['resultado']['campeona']} de la competencia; campeón NEAT #{R['D1']['resultado']['campeon_neat']}"],
        ["Ejecución", f"Del {min(r['fecha'] for r in R.values())} al {max(r['fecha'] for r in R.values())}, con las aplicaciones "
                                      "abiertas (la CPU y la GPU compartidas)"],
    ], "Entorno de ejecución.", anchos=[3.5, 13])

    D.h1("Matriz de trazabilidad")
    traz = trazabilidad()
    D.tabla(["Requerimiento", "Descripción", "Pruebas"],
            [[k, v, ", ".join(traz.get(k, [])) or "—"] for k, v in REQUERIMIENTOS.items()], "Cada requerimiento y las pruebas que lo verifican.",
            anchos=[2.2, 10.3, 4], tam=8.5)

    D.h1("Resumen de resultados")
    filas = []
    for k in ORDEN:
        if k in R:
            filas.append([k, R[k]["nombre"], estado(k), es(R[k]["conclusion"])])
    D.tabla(["ID", "Prueba", "Estado", "Resultado"], filas, "Resultado de cada prueba.", anchos=[1.1, 3.8, 1.9, 9.7], tam=8)
    cumple = sum(1 for k in R if R[k]["cumple"])
    no = [k for k in R if R[k]["cumple"] is False]
    info = [k for k in R if R[k]["cumple"] is None]
    D.p(f"**{cumple} cumplen, {len(no)} no cumplen ({', '.join(no)}) y {len(info)} son informativas ({', '.join(info)}).**")

    D.h1("Fichas de prueba")
    for g, nombre in GRUPOS.items():
        D.h2(f"Grupo {g}: {nombre}")
        for k in ORDEN:
            if not k.startswith(g) or k not in R:
                continue
            r = R[k]
            D.h3(f"{k} · {r['nombre']}")
            D.ficha([("Objetivo", r["objetivo"]), ("Requerimientos", ", ".join(r["requerimientos"])),
                     ("Procedimiento", r["procedimiento"]), ("Criterio de aceptación", r["criterio"]),
                     ("Resultado", es(r["conclusion"])), ("Estado", estado(k)), ("Ejecución", f"{r['fecha']} · {es(r['duracion_s'])} s")])
            for f in r["figuras"]:
                D.fig(RESULTADOS / f["archivo"], f"{k} · {r['nombre']}.", f.get("ancho_cm", 16))
            if k in ANALISIS:
                D.p(ANALISIS[k])

    D.h1("Hallazgos y acciones")
    D.tabla(["Prueba", "Hallazgo", "Acción", "Estado"], [
        ["A4, A6", "El espectro del gemelo está corrido hacia frecuencias altas.", "Bajar las bandas de los músculos de la muñeca y revalidar.", "Pendiente"],
        ["C5", "Un electrodo despegado hace caer la red al 27–80 %.", "Entrenar con electrodos apagados al azar (probado: 98 %).", "Pendiente"],
        ["C7", "El contrastivo no supera a las características con 10 electrodos.", "Más épocas y datos o combinarlo con las características.", "Pendiente"],
        ["D4", "Latencia mediana de 300 ms, en el límite.", "Ventana de 150 ms o mayoría de 2.", "Pendiente"],
        ["B1", "El criterio inicial no consideraba el muestreo.", "Criterio corregido con la cota teórica.", "Corregido"],
        ["G2", "Procesos huérfanos al cerrar el motor a la fuerza.", "Cada proceso vigila a su padre; prueba automática.", "Corregido"],
        ["—", "Rótulos de radio y cúbito intercambiados en el corte del antebrazo.", "Corregido en la interfaz.", "Corregido"],
        ["—", "El límite de disco borraba la carpeta de lotes y tumbaba el motor.", "Corregido; prueba de regresión en G1.", "Corregido"],
    ], "Hallazgos de las pruebas y su estado.", anchos=[1.8, 6, 6.2, 2.5], tam=8.5)

    D.h1("Pruebas fuera de alcance")
    D.p("Por decisión del proyecto, este documento no ejecuta pruebas con el hardware físico ni con personas reales. Se dejan especificadas "
        "para la siguiente etapa:")
    D.tabla(["ID", "Prueba", "Procedimiento", "Criterio propuesto"], FUERA, "Pruebas especificadas, no ejecutadas.", anchos=[1.2, 4.5, 6.8, 4], tam=8.5)

    D.h1("Cómo reproducir las pruebas")
    D.tabla(["Comando (desde la carpeta del proyecto)", "Pruebas"], [
        ["python pruebas/a_gemelo.py", "A1–A6"], ["python pruebas/b_senal.py", "B1–B3"],
        ["python pruebas/c_modelos.py", "C1–C5, C9"], ["python pruebas/c_extra.py", "C6–C8"],
        ["python pruebas/d_embebido.py", "D1–D4"], ["python pruebas/monitor_estabilidad.py 60", "datos de G2 (60 min)"],
        ["python pruebas/g_software.py", "G1–G4"], ["python -m pytest tests", "pruebas automáticas"],
        ["python docs/generar_documentos.py", "este documento y el técnico"],
    ], "Comandos (con el entorno %USERPROFILE%\\gemelo_mano\\venv).", anchos=[9, 7.5])
    t = R["G4"]["resultado"]
    D.p(f"La batería automática tiene {t['total']} pruebas en {len(t['por_archivo'])} archivos: "
        + "; ".join(f"{a} ({v['pasan']}/{v['total']})" for a, v in t["por_archivo"].items()) + ".")
    return D.guardar(ruta)
