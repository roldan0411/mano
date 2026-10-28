"""Documento técnico del sistema (requerimientos, arquitectura, modelos matemáticos, software y resultados)."""

from datos_documentos import CAPTURAS, R, REQUERIMIENTOS, es, estado, fig, ficha_campeona, ficha_neat, n, pc, trazabilidad
from word import Documento

import ia_mano as ia
import simulador as sm
from gemelo.libre import bebe
from gemelo.neuro import neat, servicio as neuro_srv
from gemelo.motor import poblacion


def construir(ruta):
    D = Documento("Documento técnico", "Gemelo digital, modelos de IA y firmware de la prótesis con 10 electrodos", "DT-GDM-01")
    D.tabla(["Versión", "Fecha", "Cambios"], [["1.0", "28/10/2026", "Primera versión: 10 electrodos, competencia, neuroevolución, mano bebé, firmware y pruebas."]],
            "Control de versiones.", anchos=[2, 3, 11.5])
    D.indice()
    camp, nt = ficha_campeona(), ficha_neat()
    c3, c1 = R["C3"]["resultado"], R["C1"]["resultado"]

    # 1 -------------------------------------------------------------------------
    D.h1("Introducción")
    D.h2("Propósito")
    D.p("Este documento describe técnicamente el gemelo digital de la prótesis mioeléctrica de mano: qué debe hacer el sistema "
        "(requerimientos), cómo está construido (arquitectura de hardware y software), los modelos matemáticos con los que se simula "
        "el antebrazo, los electrodos y la electrónica, los modelos de inteligencia artificial que reconocen los gestos y el firmware "
        "que lleva todo al ESP32. La verificación de cada requerimiento está en el documento complementario **Documentación de pruebas "
        "técnicas (DP-GDM-01)**.")
    D.h2("Alcance")
    D.p("La prótesis es una mano impresa (InMoov i2) con 5 servos de dedos y un servo de muñeca, controlada por un ESP32 que lee **10 "
        "electrodos de superficie** del antebrazo: 5 sobre los músculos flexores (F1–F5) y 5 sobre los extensores (E1–E5), cada uno "
        "medido contra un electrodo de referencia común. Reconoce 7 gestos: reposo, mano abierta, puño, pinza, señalar, flexión y "
        "extensión de muñeca.")
    D.p("El gemelo digital simula todo el sistema —personas, músculos, electrodos, amplificadores, filtro, conversor y mano— para que "
        "las redes aprendan sin persona ni hardware. Incluye tres formas de aprendizaje: una **competencia** de 10 redes entrenadas con "
        "gradiente (Population-Based Training), una **neuroevolución** (NEAT) en la que 150 redes aprenden desde cero, y una **mano bebé** "
        "que descubre sola lo que su cuerpo puede hacer. No forman parte de este documento la construcción del hardware ni las pruebas "
        "con personas reales.")
    D.h2("Definiciones y siglas")
    D.tabla(["Término", "Significado"], [
        ["EMG / sEMG", "Electromiografía / electromiografía de superficie: la señal eléctrica de los músculos medida sobre la piel."],
        ["Electrodo monopolar", "Electrodo cuya señal se mide contra un electrodo de referencia común (aquí, en el codo o sobre el cúbito)."],
        ["AD620", "Amplificador de instrumentación de Analog Devices; su ganancia la fija una resistencia RG."],
        ["ADC", "Conversor analógico-digital; el del ESP32 es de 12 bits (0 a 4095 cuentas)."],
        ["MAV, RMS, ZC, WL", "Valor absoluto medio, valor cuadrático medio, cruces por cero y longitud de onda: las 4 características por electrodo."],
        ["Calibración de la sesión", "Al ponerse los electrodos, la persona hace cada gesto una vez; las características se expresan relativas a eso."],
        ["Contexto", "Las 3 últimas ventanas juntas: 40 × 3 = 120 entradas de la red."],
        ["GDL", "Grados de libertad de la pose: 5 dedos y la muñeca."],
        ["R², pseudo-R²", "Coeficiente de determinación (pose) y pseudo-R² de McFadden (gestos); el proyecto los reporta siempre."],
        ["PBT", "Population-Based Training: población de redes que entrenan y se copian/mutan según su desempeño."],
        ["NEAT", "NeuroEvolution of Augmenting Topologies: evolución de redes que crecen, sin gradiente."],
        ["Gemelo digital", "Réplica en software del sistema físico, con sus mismos modelos y su misma cadena de procesamiento."],
    ], "Definiciones y siglas.", anchos=[4, 12.5])
    D.h2("Referencias")
    D.vinetas([
        "ISO/IEC/IEEE 29148:2018 — Ingeniería de requerimientos. ISO/IEC/IEEE 29119-3:2021 — Documentación de pruebas.",
        "Hermens H. J. et al. (2000). Development of recommendations for SEMG sensors and sensor placement procedures (SENIAM). "
        "*Journal of Electromyography and Kinesiology*, 10(5), 361–374.",
        "Hudgins B., Parker P., Scott R. N. (1993). A new strategy for multifunction myoelectric control. *IEEE Trans. Biomed. Eng.*, 40(1), 82–94.",
        "De Luca C. J. (1997). The use of surface electromyography in biomechanics. *Journal of Applied Biomechanics*, 13(2), 135–163.",
        "Analog Devices. *AD620: Low Cost Low Power Instrumentation Amplifier* (hoja de datos).",
        "McFadden D. (1974). Conditional logit analysis of qualitative choice behavior. En P. Zarembka (ed.), *Frontiers in Econometrics*.",
        "Jaderberg M. et al. (2017). Population Based Training of Neural Networks. arXiv:1711.09846.",
        "Stanley K. O., Miikkulainen R. (2002). Evolving Neural Networks through Augmenting Topologies. *Evolutionary Computation*, 10(2), 99–127.",
        "Chen T. et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML*. "
        "Khosla P. et al. (2020). Supervised Contrastive Learning. *NeurIPS*.",
        "Rolf M., Steil J. J., Gienger M. (2010). Goal babbling permits direct learning of inverse kinematics. *IEEE Trans. Auton. Ment. Dev.*, 2(3), 216–229.",
        "Baranes A., Oudeyer P.-Y. (2013). Active learning of inverse models with intrinsically motivated goal exploration in robots. "
        "*Robotics and Autonomous Systems*, 61(1), 49–73.",
        "Kapitanov A. et al. (2024). HaGRID – HAnd Gesture Recognition Image Dataset. *WACV* (licencia CC BY-SA 4.0).",
        "Lobov S. et al. (2018). Latent Factors Limiting the Performance of sEMG-Interfaces. *Sensors*, 18(4), 1122 (dataset UCI, CC BY 4.0).",
        "Farrell T. R., Weir R. F. (2007). The optimal controller delay for myoelectric prostheses. *IEEE Trans. Neural Syst. Rehabil. Eng.*, 15(1), 111–118.",
        "Langevin G. InMoov, robot humanoide de código abierto (inmoov.fr), mano i2.",
    ])

    # 2 -------------------------------------------------------------------------
    D.h1("Descripción general del sistema")
    D.p("El sistema tiene dos mitades que hacen exactamente la misma cuenta: la prótesis física, con el ESP32, y el gemelo digital en "
        "el PC. El gemelo genera personas virtuales con su señal de los 10 electrodos, las redes aprenden con esos datos y la mejor se "
        "exporta como código C al firmware.")
    D.fig(fig("T_arquitectura.png"), "Arquitectura: la cadena física de la prótesis (arriba) y su réplica en el gemelo digital (abajo). "
          "Las flechas punteadas unen cada etapa física con su modelo.", 16.5)
    D.h2("Componentes")
    D.tabla(["Componente", "Descripción"], [
        ["Mano", "InMoov i2 impresa en 3D: 5 servos de dedos (pulgar, índice, medio, anular, meñique) y un servo MG996R para la muñeca."],
        ["Electrodos", "10 de medición (F1–F5 flexores, E1–E5 extensores) en un anillo en el tercio proximal del antebrazo + 1 de referencia."],
        ["Electrónica por electrodo", "AD620 (RG = 470 Ω, G ≈ 106) y pasa-banda activo de 16–482 Hz con ganancia 10, desplazado a 1,65 V."],
        ["Adquisición", "Multiplexor analógico CD74HC4067 (16 canales) hacia un pin del ADC1 del ESP32; 1 kHz por electrodo."],
        ["Firmware", "protesis_tiempo_real.ino + procesamiento_emg.h: calibración, características, contexto, red, filtro y servos."],
        ["Gemelo digital", "Python 3.12: simulador del antebrazo y la electrónica, extracción de características idéntica a la del ESP32."],
        ["Competencia (motor)", "10 redes entrenando sin parar en la GPU (PyTorch + Keras 3); guarda campeonas con su código C."],
        ["Neuroevolución", "150 redes NEAT que aprenden desde cero a seguir la señal de una persona; exporta campeones a C."],
        ["Mano bebé", "Aprendizaje autónomo con 10 músculos y biomecánica de tendones; sin metas predeterminadas."],
        ["Aplicaciones", "Ventanas de escritorio nativas (PySide6 + OpenGL) para ver todo en vivo; procesos separados por cola."],
    ], "Componentes del sistema.", anchos=[3.8, 12.7])
    D.h2("Flujo de los datos")
    D.vinetas([
        "**Generar**: procesos de simulación crean personas (5 niveles de dificultad), sus sesiones y calibraciones, y guardan cada lote en disco.",
        "**Aprender**: la competencia entrena 10 redes con todo lo acumulado; la neuroevolución hace competir 150 redes en pistas de personas nuevas.",
        "**Evaluar**: un examen fijo de 20 personas que ninguna red vio decide la campeona; siempre se reportan acierto, pseudo-R² y R² de la pose.",
        "**Exportar**: la campeona se guarda como modelo Keras (.keras, .h5), código C (modelo_red_calibrado.h) y fotos de sus 7 gestos.",
        "**Ejecutar**: el firmware calibra al encender y aplica la misma cadena: características → calibración → contexto → red → filtro → servos.",
    ], numeradas=True)

    # 3 -------------------------------------------------------------------------
    D.h1("Requerimientos")
    D.p("Cada requerimiento tiene un código que se usa en el documento de pruebas para verificarlo (matriz de trazabilidad). Los "
        "requerimientos funcionales (RF) dicen qué hace el sistema; los no funcionales (RNF), con qué calidad.")
    traz = trazabilidad()
    D.tabla(["Código", "Requerimiento", "Verificación"],
            [[k, v, ", ".join(traz.get(k, [])) or "—"] for k, v in REQUERIMIENTOS.items() if k.startswith("RF")],
            "Requerimientos funcionales.", anchos=[1.6, 11.4, 3.5])
    D.tabla(["Código", "Requerimiento", "Verificación"],
            [[k, v, ", ".join(traz.get(k, [])) or "—"] for k, v in REQUERIMIENTOS.items() if k.startswith("RNF")],
            "Requerimientos no funcionales.", anchos=[1.6, 11.4, 3.5])
    D.h2("Restricciones de diseño")
    D.vinetas([
        "Aplicación de escritorio nativa para Windows (no web), por decisión del proyecto.",
        "Entrenamiento local en una portátil con GPU NVIDIA RTX 4050; TensorFlow no usa la GPU en Windows nativo, por eso Keras 3 corre sobre PyTorch con CUDA 12.6.",
        "La red que va al ESP32 debe ser pequeña (densa, sin convoluciones) y usar solo operaciones simples en punto flotante.",
        "El ESP32 tiene 8 canales de ADC1 (los de ADC2 no funcionan con WiFi): los 10 electrodos se leen por un multiplexor.",
        "Todo modelo reporta un R² (pseudo-R² de McFadden en clasificación).",
    ])

    # 4 -------------------------------------------------------------------------
    D.h1("Electrodos y montaje")
    D.p("Se eligió el montaje con el que se construyó y entrenó todo el sistema: **10 electrodos monopolares y uno de referencia** (11 en "
        "total). Cada electrodo se mide contra la referencia común con su propio AD620. Los 10 forman un anillo alrededor de la parte "
        "más gruesa del antebrazo, separados unos 30° (≈ 2,5 cm en un adulto).")
    D.fig(fig("A1_acople.png"), "Izquierda: matriz de acople músculo → electrodo. Derecha: corte del antebrazo con los 10 músculos "
          "(radio = profundidad) y los 10 electrodos.", 16)
    D.tabla(["Electrodo", "Ángulo", "Ubicación", "Músculo que más capta"],
            [[r["electrodo"], f"{r['angulo']:.0f}°", ("cara palmar" if r["electrodo"].startswith("F") else "cara dorsal"), r["musculo_max"]]
             for r in R["A1"]["resultado"]["electrodos"]],
            "Posición nominal de cada electrodo (90° = dorso, 270° = palma).", anchos=[2.5, 2, 4, 5], centrar=[0, 1])
    D.h2("Colocación")
    D.vinetas([
        "Anillo en el tercio del antebrazo más cercano al codo (donde es más grueso).",
        "F3 en el centro de la cara palmar y E3 en el centro del dorso; F1–F2 hacia el lado del meñique y F4–F5 hacia el pulgar; "
        "E1–E2 hacia el pulgar y E4–E5 hacia el meñique.",
        "Referencia sobre hueso: olécranon (punta del codo) o el borde subcutáneo del cúbito, que existe también en una persona amputada transradial.",
        "Calibrar (≈ 28 s) cada vez que se colocan los electrodos: el firmware lo hace al encender.",
    ])
    D.h2("Electrodos recomendados")
    D.p("Para las pruebas de banco se usan electrodos desechables Ag/AgCl de gel sólido **Kendall/Covidien H124SG** (Ø 24 mm, sensor de "
        "80 mm²): 11 por sesión. Para el uso diario, la recomendación es un **brazalete** con electrodos secos reutilizables de acero "
        "inoxidable (≈ 10 mm) y la referencia en la misma banda sobre el borde del cúbito; las pruebas C4 y C5 muestran que el brazalete "
        "funciona si se recalibra al ponerlo y si la red se entrena con electrodos que pierden contacto (sección 8).")

    # 5 -------------------------------------------------------------------------
    D.h1("Modelos matemáticos del gemelo digital")
    D.h2("Señal de cada músculo")
    D.p("Cada uno de los 10 músculos genera una señal estocástica: ruido blanco filtrado en su banda (Butterworth de 2.º orden), "
        "normalizado, modulado por la activación del gesto y por oscilaciones lentas, a 4 kHz.")
    D.ec(r"s_m(t) = A_{\mathrm{MVC}}\,(1+0{,}3\,\phi)\,g_m\,a_m(t)\,\left[1+0{,}08\sin(2\pi\,0{,}7\,t+\theta_{1})+0{,}03\sin(2\pi\,10\,t+\theta_{2})\right]\,n_m(t)",
         [(r"A_{\mathrm{MVC}}=0{,}4\ \mathrm{mV}", "amplitud de un músculo superficial al 100 %"), (r"\phi", "fatiga (0 a 1)"),
          (r"g_m", "ganancia del músculo en esa persona (lognormal)"), (r"a_m(t)", "activación"), (r"n_m(t)", "ruido filtrado en la banda del músculo")])
    D.p("La activación pasa del reposo al gesto con una constante de tiempo de 60 ms:")
    D.ec(r"a_m(t) = a_{1} + (a_{0}-a_{1})\,e^{-t/\tau},\qquad \tau = 60\ \mathrm{ms}")
    D.tabla(["Músculo", "Nombre", "Banda (Hz)", "Ángulo", "Profundidad"],
            [[k, v[0], f"{v[1][0]}–{v[1][1]}", f"{v[2]}°", n(v[3], 2)] for k, v in sm.MUSCULOS.items()],
            "Los 10 músculos del modelo: banda de frecuencia y posición en el corte del antebrazo.", anchos=[1.6, 6, 2.4, 2, 2.4], centrar=[2, 3, 4])
    D.tabla(["Gesto"] + list(sm.MUSCULOS), [[g] + [n(v, 2) for v in sm.ACTIVACION[g]] for g in ia.GESTOS],
            "Activación de cada músculo en cada gesto (antes de la variación de cada persona).", tam=8)
    D.h2("Electrodos y acople")
    D.p("El acople entre el músculo m y el electrodo c depende de la distancia angular, de la profundidad del músculo y del corrimiento "
        "del electrodo a lo largo del brazo:")
    D.ec(r"w_{cm} = \frac{e^{-\Delta\theta_{cm}^{2}/2\sigma^{2}}\,e^{-(1-p_m)/\lambda} + \kappa\,e^{-(1-p_m)/\lambda}}{\mathrm{máx}_{m}\left(\cdot\right)}\;e^{-|z_c|/\lambda_z}",
         [(r"\sigma=38^{\circ}", "ancho angular"), (r"\lambda=0{,}35", "atenuación por profundidad"), (r"\kappa=0{,}05", "conducción de volumen"),
          (r"\lambda_z=3\ \mathrm{cm}", "atenuación axial")])
    D.p("La tensión en cada electrodo suma los músculos, el ruido, la interferencia de 60 Hz y el artefacto de movimiento:")
    D.ec(r"v_c(t) = \pi_c\sum_{m=1}^{10} w_{cm}\,s_m(t) + \eta_c(t) + \rho_c\,I_{60}\sin(2\pi\,60\,t+\varphi) + \alpha_c\,A\,e^{-t/0{,}15}\sin(2\pi\,4\,t)",
         [(r"\pi_c", "estado de la piel de esa sesión"), (r"\eta_c", "ruido de los electrodos (3 µV × nivel)"), (r"I_{60}", "interferencia (µV)")])
    D.p("En cada sesión los 10 electrodos giran juntos alrededor del brazo (cómo se los puso la persona) y cada uno queda además un poco "
        "corrido de su lugar; el corrimiento crece con el nivel de dificultad d:")
    D.ec(r"\theta_c = \theta_c^{0} + \mathcal{N}(0,\,30d+3) + \mathcal{N}(0,\,4d+1),\qquad z_c = \mathcal{N}(0,\,1{,}5d+0{,}2)+\mathcal{N}(0,\,0{,}3d+0{,}05)")
    D.h2("Cadena de adquisición")
    D.p("Primera etapa: el AD620 amplifica la señal y también el desfase continuo entre cada electrodo y la referencia (media celda); "
        "si el producto supera el riel, satura.")
    D.ec(r"G = 1 + \frac{49{,}4\ \mathrm{k\Omega}}{R_G},\qquad v_{\mathrm{AD}} = \mathrm{recortar}\!\left(G\,(v_c+\delta_c),\ \pm 6{,}1\ \mathrm{V}\right)")
    D.p("Segunda etapa: pasa-banda activo (R1 = 10 kΩ, C1 = 1 µF, Rf = 100 kΩ, Cf = 3,3 nF, acople de salida Rth = 51 kΩ, C2 = 1 µF), "
        "discretizado a 4 kHz con la transformación bilineal:")
    D.ec(r"H(s) = -\frac{R_f C_1 s\;\cdot\;R_{th} C_2 s}{(R_f C_f s+1)(R_1 C_1 s+1)(R_{th} C_2 s+1)},\qquad f_L=\frac{1}{2\pi R_1 C_1}=15{,}9\ \mathrm{Hz},\quad f_H=\frac{1}{2\pi R_f C_f}=482\ \mathrm{Hz}")
    D.p("Tercera etapa: desplazamiento a 1,65 V, diezmado a 1 kHz y ADC de 12 bits con su ruido de cuantización:")
    D.ec(r"N = \mathrm{redondear}\!\left(\frac{1{,}65+v_f}{3{,}3}\cdot 4095 + \mathcal{N}(0,\,1{,}5)\right),\qquad 0 \leq N \leq 4095")
    D.fig(fig("A3_bode.png"), "Respuesta en frecuencia del pasa-banda: analógica y discretizada a 4 kHz (prueba A3).", 13)
    D.fig(fig("A2_ad620.png"), "Ganancia del AD620 frente a RG y desfase máximo de los electrodos antes de saturar (prueba A2).", 15)
    D.fig(fig("T_etapas_puno.png"), "La señal de los 10 electrodos en cada etapa durante el gesto «puño» (flexores en azul, extensores en verde).", 15)
    D.h2("Personas, sesiones y dificultad")
    D.p("Cada persona virtual tiene sus propias ganancias musculares, su propio patrón de cada gesto y su anatomía (los músculos no "
        "están en el mismo ángulo en todos). Las condiciones vienen de 5 niveles de dificultad, con ±15 % de variación por persona:")
    D.tabla(["Nivel", "Ruido (×)", "60 Hz (µV)", "Desplazamiento", "Fatiga", "Variabilidad"],
            [[k + 1, n(c.ruido), n(c.interferencia_60hz), n(c.desplazamiento, 2), n(c.fatiga, 2), n(c.variabilidad, 2)] for k, c in enumerate(sm.NIVELES)],
            "Niveles de dificultad de las personas virtuales.", centrar=[0, 1, 2, 3, 4, 5])
    D.h2("Características de cada ventana")
    D.p(f"Cada {ia.PASO} ms se toma una ventana de {ia.VENTANA} muestras de cada electrodo (cuentas del ADC, sin la media) y se calculan "
        "las 4 características clásicas de Hudgins et al. (1993); con 10 electrodos son 40 valores:")
    D.ec(r"\mathrm{MAV} = \frac{1}{N}\sum_{i=1}^{N}\left|x_i-\bar{x}\right|,\qquad \mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(x_i-\bar{x}\right)^{2}}")
    D.ec(r"\mathrm{ZC} = \sum_{i=1}^{N-1}\mathbb{1}\left[x_i x_{i+1}<0\ \wedge\ |x_{i+1}-x_i|\geq\varepsilon\right],\qquad \mathrm{WL} = \sum_{i=1}^{N-1}\left|x_{i+1}-x_i\right|",
         [(r"N=200", "muestras"), (r"\varepsilon=20", "umbral de cruce en cuentas")])
    D.h2("Calibración de la sesión y contexto")
    D.p("Al ponerse los electrodos la persona hace cada gesto una vez. Con esas ventanas C se calcula, para cada una de las 40 "
        "características, su referencia; luego todas se expresan relativas a ella y se juntan las 3 últimas ventanas:")
    D.ec(r"c_k = \frac{1}{|C|}\sum_{w\in C}\ln\left(1+F_k^{(w)}\right),\qquad \tilde{F}_k = \ln\left(1+F_k\right) - c_k")
    D.ec(r"\mathbf{x}_t = \left[\tilde{\mathbf{F}}_{t-2},\ \tilde{\mathbf{F}}_{t-1},\ \tilde{\mathbf{F}}_{t}\right]\in\mathbb{R}^{120}")
    D.p(f"La calibración absorbe la piel de ese día, la ganancia de cada electrodo y dónde quedaron: en personas nuevas sube el acierto "
        f"de {pc(R['B2']['resultado']['sin_calibrar_media'])} a {pc(R['B2']['resultado']['calibrado_media'])} (prueba B2) y, si los electrodos "
        "giran, recalibrar recupera prácticamente todo el acierto (prueba C4).")
    D.h2("Red de reconocimiento y competencia (PBT)")
    D.p("Cada red es densa: normalización de las 120 entradas, una o dos capas ocultas ReLU y dos salidas, el gesto (7, softmax) y la "
        "pose (6 GDL, lineal). Se entrena con PyTorch en la GPU minimizando:")
    D.ec(r"\mathcal{L} = -\sum_{k=1}^{7} y_k \ln p_k + \lambda_{\mathrm{pose}}\left\lVert \mathbf{P}-\hat{\mathbf{P}}\right\rVert^{2}")
    D.p("Diez redes con distinto genoma (tamaño, tasa de aprendizaje, olvido, L2, peso de la pose, ruido de aumento) entrenan por turnos "
        "con los datos acumulados. Después de cada generación rinden el examen fijo y se ordenan por el puntaje:")
    D.ec(r"S = 0{,}6\,\mathrm{acierto} + 0{,}2\,\mathrm{máx}\!\left(R^{2}_{\mathrm{McF}},0\right) + 0{,}2\,\mathrm{máx}\!\left(R^{2}_{\mathrm{pose}},0\right)")
    D.p("Las 3 peores copian los pesos de una de las 3 mejores y mutan su genoma; cada 12 generaciones entra una red nueva al azar. Si la "
        "mejor supera a la campeona por más de 0,003, se corona y se exporta.")
    D.tabla(["Gen", "Opciones"], [[k, ", ".join(map(str, v))] for k, v in poblacion.OPCIONES.items()] +
            [["lr", "10^U(−3,4; −2,3); muta ×0,7 a ×1,4"]], "Espacio de genomas de la competencia.", anchos=[3, 13.5])
    D.fig(fig("T_red_competencia.png"), "Arquitectura de la red campeona que se exporta al ESP32.", 15)
    D.h2("Métricas: R² y pseudo-R²")
    D.ec(r"R^{2} = 1-\frac{\sum_i\left(y_i-\hat{y}_i\right)^{2}}{\sum_i\left(y_i-\bar{y}\right)^{2}},\qquad R^{2}_{\mathrm{McF}} = 1-\frac{\ln\hat{L}_{\mathrm{modelo}}}{\ln\hat{L}_{\mathrm{nulo}}}",
         [(r"\hat{L}_{\mathrm{nulo}}", "verosimilitud de adivinar uniformemente entre los 7 gestos")])
    D.h2("Aprendizaje contrastivo")
    D.p("Un codificador 1D-CNN sobre la señal cruda de los 10 electrodos aprende sin etiquetas acercando dos vistas físicas de la misma "
        "contracción (los electrodos recolocados) y alejando las demás (NT-Xent, SimCLR); con etiquetas se usa la variante SupCon:")
    D.ec(r"\ell_{i} = -\ln\frac{\exp\left(\mathrm{sim}(z_i,z_j)/\tau\right)}{\sum_{k\neq i}\exp\left(\mathrm{sim}(z_i,z_k)/\tau\right)},\qquad \tau=0{,}1")
    D.h2("Neuroevolución (NEAT)")
    D.p(f"Las redes nacen casi vacías: cada salida conectada al sesgo y a {neat.ENTRADAS_INICIALES} de las 120 entradas al azar. La evolución "
        f"muta pesos (cada peso con probabilidad {n(neat.POR_PESO, 1)}), agrega conexiones ({n(neat.P_CONEXION, 2)}) y neuronas "
        f"({n(neat.P_NEURONA, 2)}) partiendo una conexión en dos. Las redes se agrupan en especies con la distancia de compatibilidad:")
    D.ec(r"\delta = c_1\frac{E+D}{N} + c_3\,\bar{W},\qquad c_1=1,\ c_3=0{,}4",
         [(r"E+D", "genes no compartidos"), (r"N", "genes del genoma más grande"), (r"\bar{W}", "diferencia media de pesos de los genes compartidos")])
    D.p("La pista es una persona del gemelo haciendo 21 gestos al azar (≈ 29 s). Una mano «choca» si hace el gesto equivocado 5 ventanas "
        "seguidas (0,5 s); las ventanas de transición no cuentan. El fitness de cada red, en 6 personas distintas por generación, es:")
    D.ec(r"f = \bar{T}_{\mathrm{sin\ chocar}} + 30\cdot\mathrm{acierto}")
    D.p("Cada especie recibe hijos en proporción a su fitness medio; la mejor de cada especie con 5 o más miembros pasa intacta, las "
        "demás se cruzan (75 %) o se copian y mutan; una especie que no mejora en 20 generaciones se extingue.")
    D.fig(fig("T_red_neat.png"), "Red del campeón NEAT: solo las entradas que usa, las neuronas que creció la evolución y sus conexiones "
          "(azul positivas, naranja negativas).", 15)
    D.h2("Mano bebé: aprendizaje autónomo")
    D.p("La mano bebé controla 10 músculos (no gestos). Una biomecánica que ella no conoce traduce las activaciones a pose; los tendones "
        "acoplan los dedos como en una mano real:")
    D.ec(r"q_w = \tanh\left(1{,}7\,(a_{\mathrm{ECR}}+a_{\mathrm{ECU}}-a_{\mathrm{FCR}}-a_{\mathrm{FCU}}) + 0{,}25\,a_{\mathrm{EDC}} - 0{,}25\,a_{\mathrm{FDS}}\right)")
    D.ec(r"f_d = g_d\left[1{,}1\left(W^{\mathrm{FDS}}_d a_{\mathrm{FDS}} + W^{\mathrm{FDP}}_d a_{\mathrm{FDP}}\right) - W^{\mathrm{EDC}}_d a_{\mathrm{EDC}} - W^{\mathrm{EIP}}_d a_{\mathrm{EIP}}\right] + 0{,}3\,q_w")
    D.ec(r"q_d = 0{,}25 + k_d\tanh\left(1{,}4\,f_d\right),\qquad k_d = 0{,}75\ \text{si}\ f_d\geq 0,\quad k_d=0{,}25\ \text{si}\ f_d<0")
    D.fig(fig("T_tendones.png"), "Reparto de los tendones entre los dedos y acoples: un solo músculo mueve varios dedos; señalar exige coordinar dos.", 15)
    D.p("Su cerebro son un modelo inverso (pose → músculos) y uno directo (músculos → pose), redes que aprenden en línea. Ella elige "
        "sola cómo aprender entre 4 estrategias (inventar una pose, ir más allá de algo que sabe, practicar lo nuevo, balbucear) según lo "
        "que más le enseña cada una (curiosidad):")
    D.ec(r"V_k \leftarrow V_k + 0{,}03\,(r-V_k),\qquad \pi_k = 0{,}85\,\frac{e^{V_k/0{,}08}}{\sum_j e^{V_j/0{,}08}} + \frac{0{,}15}{4}")
    D.ec(r"r = 0{,}3\,\frac{\mathrm{mín}(\nu/\rho,\,2)}{2} + \mathbb{1}[\mathrm{habilidad\ nueva}] + 0{,}5\,\mathbb{1}[\mathrm{algo\ nuevo}] + \mathrm{progreso}",
         [(r"\nu", "distancia a lo que ya sabe"), (r"\rho", "radio de novedad (empieza en 0,35 y baja ×0,85 con cada nivel de finura)")])
    D.p("Una pose es una **habilidad** cuando la repite 3 veces seguidas con error menor a 0,45ρ. El repertorio no tiene tope: si en "
        "1.500 intentos no encuentra nada nuevo, sube de finura. Internet es un espejo: cada habilidad se compara con 19.800 manos reales "
        "de HaGRID v2 (21 puntos convertidos a flexión de cada dedo) para anotar si alguna persona hace algo parecido.")
    D.h2("Firmware: de la señal al servo")
    D.p("El firmware (protesis_tiempo_real.ino) lee los 10 electrodos por el multiplexor en cada milisegundo y cada 100 ms ejecuta "
        "procesamiento_emg.h —la misma cuenta que Python, verificada en la prueba D1—, la red y un filtro de dos pasos:")
    D.vinetas([
        "**Confianza**: si la probabilidad del gesto ganador es menor a 0,60, mantiene el gesto anterior.",
        "**Mayoría**: muestra el gesto que ganó 2 de las 3 últimas decisiones (evita temblores).",
        "Los dedos van a la pose del gesto; la muñeca se mueve 2° cada 20 ms (100 °/s) para no pedir picos de corriente.",
    ])
    D.p("La primera decisión posible llega cuando termina la primera ventana (200 ms después del inicio del gesto) y la mayoría agrega "
        "una ventana: la latencia mínima es t = 200 + 100k ms con k ≥ 1 (prueba D4).")

    # 6 -------------------------------------------------------------------------
    D.h1("Software")
    D.h2("Procesos y aplicaciones")
    D.p("Cada aplicación es una ventana nativa de Qt; el trabajo pesado corre en procesos aparte que se comunican por colas, así la "
        "interfaz nunca se congela. Los procesos de simulación vigilan a su padre y se cierran solos si este muere.")
    D.tabla(["Aplicación", "Abre", "Qué hace"], [
        ["Gemelo de la mano", "Gemelo de la mano.lnk (app_gemelo.py)", "Competencia de 10 redes en la GPU, red en vivo, gemelo y electrodos, campeonas y datos."],
        ["Neuroevolución", "Neuroevolución de la mano.lnk (--neuro)", "150 redes NEAT: 30 manos en 3D, la pista, la red de la mejor, curvas y especies."],
        ["Mano bebé", "Mano bebé (entrenamiento libre).lnk (--libre)", "Aprendizaje autónomo: decisiones, repertorio, espejo de internet y curvas."],
    ], "Aplicaciones de escritorio.", anchos=[3.2, 5.5, 7.8])
    D.fig(CAPTURAS / "pestana_0.png", "Competencia de manos: las 10 manos InMoov siguiendo a una persona nueva, ranking del examen y "
          "puntaje por generación.", 16)
    D.fig(CAPTURAS / "pestana_1.png", "Red neuronal de la mano líder con sus activaciones en vivo (120 entradas por electrodo y ventana).", 16)
    D.fig(CAPTURAS / "pestana_2.png", "Gemelo y electrodos: corte del antebrazo con los 10 electrodos, la señal en cada etapa y las 40 "
          "características calibradas.", 16)
    D.fig(CAPTURAS / "pestana_3.png", "Campeonas: fotos de los 7 gestos de cada campeona.", 16)
    D.fig(CAPTURAS / "pestana_4.png", "Datos guardados por el motor.", 16)
    D.fig(CAPTURAS / "neuroevolucion.png", "Neuroevolución: 30 de las 150 manos (color = especie, gris = chocó), la pista con los 10 "
          "electrodos y la red de la mejor.", 16)
    D.fig(CAPTURAS / "mano_bebe.png", "Mano bebé: lo que decide hacer, su repertorio sin tope, el espejo de internet y sus curvas de R².", 16)
    D.h2("Persistencia")
    D.tabla(["Carpeta (en %USERPROFILE%\\gemelo_mano\\)", "Contenido"], [
        ["motor\\gemelo.db", "Base SQLite: lotes, generaciones, candidatas, campeonas y eventos."],
        ["motor\\lotes\\", "Cada lote: señal cruda de los 10 electrodos, características, calibración por sesión, gesto y pose (límite 3 GB)."],
        ["motor\\examen.h5", "Examen fijo: 20 personas que ninguna red ve."],
        ["motor\\poblacion\\", "Las 10 redes de la competencia (se guardan en cada generación)."],
        ["motor\\campeonas\\", "Cada campeona: modelo .keras/.h5, modelo_red_calibrado.h, ficha y fotos."],
        ["motor\\neuro\\", "Población NEAT, pistas y campeones con red_neat.h."],
        ["motor\\bebe\\", "Estado de la mano bebé (se guarda cada 20 s) y fotos de sus habilidades."],
        ["archivo_2canales\\", "Datos y modelos del diseño anterior de 2 electrodos (línea base de las pruebas)."],
    ], "Dónde se guarda cada cosa.", anchos=[5.5, 11])
    D.h2("Estructura del repositorio")
    D.tabla(["Ruta", "Contenido"], [
        ["app_gemelo.py", "Punto de entrada de las tres aplicaciones."],
        ["simulador.py · ia_mano.py", "Física del antebrazo y electrodos · características, red y exportación a C (compartido con el ESP32)."],
        ["gemelo/motor/", "Gemelo v2 (cadena por etapas), almacén, población PBT, fotos."],
        ["gemelo/neuro/", "NEAT, la pista y el servicio de neuroevolución."],
        ["gemelo/libre/", "Mano bebé: cuerpo, cerebro y espejo de internet."],
        ["gemelo/escritorio/", "Ventanas Qt y escenas OpenGL."],
        ["protesis_tiempo_real.ino · procesamiento_emg.h", "Firmware del ESP32 y procesamiento en C puro."],
        ["pruebas/ · tests/", "Pruebas técnicas (registros y figuras) y pruebas automáticas (pytest)."],
        ["docs/", "Este documento y el de pruebas, y el código que los genera."],
    ], "Estructura del repositorio.", anchos=[6, 10.5])

    # 7 -------------------------------------------------------------------------
    D.h1("Resultados principales")
    D.tabla(["Indicador", "Valor", "Prueba"], [
        ["Acierto de la campeona en 20 personas nuevas", f"{pc(c3['acierto'], 2)} (IC 95 %: {pc(c3['ic95_acierto'][0])}–{pc(c3['ic95_acierto'][1])})", "C3"],
        ["Pseudo-R² de McFadden (gestos) · R² de la pose", f"{n(c3['pseudo_r2'], 3)} · {n(c3['r2_pose'], 3)}", "C3"],
        ["Recall mínimo por gesto", pc(c3["recall_min"]), "C3"],
        ["2 → 10 electrodos (acierto · R² de la pose)", f"{pc(c1['acierto']['2'])} → {pc(c1['acierto']['10'])} · {n(c1['r2_pose']['2'], 3)} → {n(c1['r2_pose']['10'], 3)}", "C1"],
        ["Efecto de la calibración de 20 s", f"{pc(R['B2']['resultado']['sin_calibrar_media'])} → {pc(R['B2']['resultado']['calibrado_media'])}", "B2"],
        ["NEAT desde cero (150 generaciones)", f"{pc(R['C6']['resultado']['neat_media'])} ± {pc(R['C6']['resultado']['neat_sd'])}", "C6"],
        ["Mano bebé: habilidades en 15.000 intentos", es(R["C8"]["conclusion"].split(";")[0].replace("Habilidades descubiertas: ", "")), "C8"],
        ["Firmware frente a Python", f"mismo gesto en el {pc(R['D1']['resultado']['peor']['mismo_gesto_red'], 0)} de las ventanas", "D1"],
        ["Tiempo estimado por ventana en el ESP32", f"{n(R['D2']['resultado']['esp32_total_ms'], 1)} ms de 100 ms", "D2"],
        ["Latencia de reconocimiento (mediana)", f"{R['D4']['resultado']['mediana_ms']:.0f} ms", "D4"],
    ], "Resultados principales (detalle en DP-GDM-01).", anchos=[8, 6, 2.5])
    no = [k for k in R if R[k]["cumple"] is False]
    D.p(f"De {len(R)} pruebas, {sum(1 for k in R if R[k]['cumple'])} cumplen, {len(no)} no cumplen ({', '.join(no)}) y "
        f"{sum(1 for k in R if R[k]['cumple'] is None)} son informativas. Las que no cumplen son hallazgos que guían el trabajo siguiente.")

    # 8 -------------------------------------------------------------------------
    D.h1("Limitaciones y trabajo futuro")
    D.vinetas([
        "**Espectro del gemelo** (A4, A6): la señal simulada tiene más contenido en alta frecuencia que el EMG real (frecuencia mediana 55 Hz "
        "frente a 36 Hz en la misma banda). Bajar las bandas de los músculos de la muñeca y volver a validar.",
        "**Electrodos que se despegan** (C5): la campeona cae al 27–80 % si un electrodo pierde contacto; entrenando con electrodos "
        "apagados al azar se mantiene en 98 %. Incorporar ese aumento al motor y a la neuroevolución.",
        "**Latencia** (D4): la mediana es 300 ms, justo en el límite; con el filtro de mayoría de 2 decisiones o ventanas de 150 ms "
        "(prueba B3: mismo acierto) bajaría a ~200–250 ms.",
        "**Aprendizaje contrastivo** (C7): con 10 electrodos las características clásicas ya son muy informativas; el codificador "
        "contrastivo necesita más datos y épocas para aportar.",
        "**Brazalete**: modelar electrodos secos (más impedancia, pérdidas de contacto) y la referencia en la banda.",
        "**Fuera de alcance de este documento**: pruebas del hardware (ruido, CMRR, servos, batería) y con personas reales (dataset "
        "propio, prueba TAC, prueba funcional y usabilidad). Sus protocolos están en DP-GDM-01.",
    ])
    return D.guardar(ruta)
