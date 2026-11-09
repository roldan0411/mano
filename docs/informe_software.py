"""
Informe de desarrollo del software: competencia de manos, neuroevolución y mano bebé
====================================================================================
Cronología tomada de las solicitudes del equipo (con su fecha y hora) y del historial de git; cifras
tomadas de los registros de las pruebas y del estado guardado por cada programa.

  python docs/informe_software.py     → docs/Informe_software_IS-GDM-01.docx y .pdf
"""

import ast
import pickle
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from datos_documentos import CAPTURAS, MOTOR, R, RAIZ, es, estado, fig, n, pc  # noqa: E402
from word import Documento  # noqa: E402

from gemelo.libre import bebe as modulo_bebe  # noqa: E402,F401  (para leer su estado guardado)
from gemelo.neuro import neat  # noqa: E402

CRONOLOGIA = [
    ("22/10 15:10–15:34", "Analizar el proyecto; adoptar aprendizaje contrastivo, datos sintéticos de 50.000 a 100.000 muestras y entrenamiento local en la GPU.",
     "Gemelo digital base, datos sintéticos y notebook de entrenamiento (antecedente de este informe)."),
    ("22/10 17:22–17:42", "Ver el entrenamiento en vivo, con la mano InMoov, y no en cámara rápida.", "Tablero en vivo y mano InMoov 3D."),
    ("23/10 17:58", "Muchas manos que compitan hasta llegar a la mejor, entrenando sin parar en un programa de escritorio; guardar datos y fotos de las mejores; subir el R².",
     "Análisis y propuesta de una población de redes con examen fijo."),
    ("23/10 18:07", "Correr en la GPU, ver la «Competencia de manos» y la red neuronal mientras entrena, ver los electrodos moverse; conservar las personas; aprobada la calibración de 20 s; 8 a 10 manos.",
     "Diseño aprobado: 10 manos, gemelo v2 con calibración por sesión."),
    ("23/10 18:19–18:33", "Debe ser aplicación de escritorio, no web, y quedar en el escritorio.", "Interfaz web descartada; aplicación Qt + OpenGL con acceso directo."),
    ("23/10 18:58", "Nueva ventana de entrenamiento libre: que aprenda poses no predeterminadas, como un bebé, alimentándose de la red.", "Primera versión de la mano bebé."),
    ("23/10 19:06", "Repite los mismos gestos: debe partir de cero con conocimiento de internet.", "Segunda versión: imita manos reales de HaGRID v2."),
    ("26/10 15:23", "Al abrir los programas todo aparece en cero: deben retomar la última red guardada o la de mayor conocimiento. La mano bebé debe entrenar sin límite.",
     "Causa encontrada y corregida (sección 7); los programas retoman lo guardado."),
    ("26/10 15:46–16:04", "La mano bebé debe aprender sola, sin escenarios predeterminados, y empezar desde cero.", "Tercera versión: aprendizaje autónomo; la versión anterior queda como respaldo."),
    ("26/10 16:19–16:22", "Se comparte un video de carros que aprenden a manejar: hacer neuroevolución para la mano.", "Programa de neuroevolución NEAT con 150 redes."),
    ("27/10 14:29–14:36", "Todo debe usar 10 electrodos (5 flexores y 5 extensores); documentar; pruebas sin hardware ni personas.", "Competencia y neuroevolución migradas a 10 electrodos y reiniciadas."),
    ("27/10 15:14–28/10 12:24", "Continuar las pruebas y terminar la documentación.", "26 pruebas técnicas y 24 pruebas automáticas."),
    ("09/11 16:09", "Elaborar este informe del software con las solicitudes y sus fechas.", "El presente documento."),
]

HITOS = [
    ["22/10", "—", "—", "Antecedente: gemelo base y datos sintéticos."],
    ["23/10", "Primera versión con 10 manos en la GPU (2 electrodos).", "—", "Versiones 1 y 2 (imitación de HaGRID v2)."],
    ["26/10", "Retoma lo guardado; entrena en la CPU sin GPU; corrige el error de disco.", "Primera versión: 150 redes, 30 visibles.", "Versión 3: autónoma, desde cero."],
    ["27/10", "Migrada a 10 electrodos; reinicia desde cero.", "Migrada a 120 entradas; reinicia desde cero.", "Sin cambios: no usa electrodos."],
    ["28/10", "Procesos sin huérfanos; pruebas C1–C5 y C9.", "Prueba C6; campeón exportado a C verificado.", "Prueba C8; 299 habilidades observadas."],
]

ERRORES = [
    ["23/10", "Competencia", "Falta de RAM con los procesos de simulación.", "Cada proceso reservaba hilos de cálculo de más.", "Un hilo por proceso: de 850 a 110 MB cada uno."],
    ["23/10", "Competencia", "Las manos 3D se veían oscuras.", "La iluminación por defecto no daba de frente.", "Sombreador propio con luz frontal."],
    ["26/10", "Competencia", "Todo aparecía en cero al abrir.", "Sin cargador el motor se pausaba, y la ventana solo recibía datos al terminar una generación.",
     "Publica lo guardado al abrir y en pausa; sin GPU entrena en la CPU."],
    ["26/10", "Competencia", "El motor se caía tras la generación 465.", "Al llegar al límite de 3 GB intentaba borrar lotes ya borrados, es decir, la carpeta.", "Solo considera los lotes existentes; prueba de regresión."],
    ["26/10", "Neuroevolución", "Se estancaba cerca de 70 % de acierto.", "Mutaba todos los pesos a la vez.", "Cada peso muta con probabilidad 0,2 (sección 4.2)."],
    ["26/10", "Neuroevolución", "Riesgo de dos ventanas sobre la misma población.", "No había exclusión entre procesos.", "Candado: una sola neuroevolución a la vez."],
    ["28/10", "Competencia y neuroevolución", "Procesos de simulación huérfanos.", "Cerrar el motor a la fuerza no los terminaba.", "Cada proceso vigila a su padre; prueba automática."],
    ["28/10", "Competencia", "Radio y cúbito intercambiados en el corte del antebrazo.", "Rótulos del dibujo (la simulación estaba bien).", "Rótulos corregidos."],
]

AJUSTE = [["A", "Todos", "2", "3", "63,2 %"], ["B", "Todos", "2", "6", "71,2 %"], ["C", "Todos", "4", "6", "60,4 %"], ["D", "Todos", "24", "6", "73,2 %"],
          ["E", "20 % de los pesos", "4", "6", "86,4 %"], ["F", "10 % de los pesos", "4", "6", "86,7 %"], ["G (elegida)", "20 % de los pesos", "4", "6", "89,2 %"]]


def constantes_motor():
    arbol = ast.parse((RAIZ / "gemelo" / "motor" / "servicio.py").read_text(encoding="utf-8"))
    d = {}
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign):
            for t in nodo.targets:
                if isinstance(t, ast.Name):
                    try:
                        d[t.id] = ast.literal_eval(nodo.value)
                    except ValueError:
                        pass
                elif isinstance(t, ast.Tuple) and isinstance(nodo.value, ast.Tuple):
                    for a, b in zip(t.elts, nodo.value.elts):
                        d[a.id] = ast.literal_eval(b)
    return d


def estado_actual():
    db = sqlite3.connect(MOTOR / "gemelo.db")
    gen, ult = db.execute("SELECT MAX(id), MAX(t) FROM generaciones").fetchone()
    camp = db.execute("SELECT id, gen, acierto, pseudo_r2, r2_pose FROM campeonas ORDER BY id DESC LIMIT 1").fetchone()
    n_camp = db.execute("SELECT COUNT(*) FROM campeonas").fetchone()[0]
    lotes = db.execute("SELECT COUNT(*), SUM(ventanas), SUM(personas) FROM lotes").fetchone()
    d2 = sqlite3.connect(MOTOR.parent / "archivo_2canales" / "gemelo.db")
    g2 = d2.execute("SELECT MAX(id) FROM generaciones").fetchone()[0]
    a2 = d2.execute("SELECT MAX(acierto) FROM campeonas").fetchone()[0]
    with open(MOTOR / "neuro" / "poblacion.pkl", "rb") as f:
        pob = pickle.load(f)
    with open(MOTOR / "bebe" / "bebe.pkl", "rb") as f:
        b = pickle.load(f)
    return dict(gen=gen, ult=time.strftime("%d/%m/%Y %H:%M", time.localtime(ult)), camp=camp, n_camp=n_camp, lotes=lotes, g2=g2, a2=a2, pob=pob, b=b)


def historial_git():
    r = subprocess.run(["git", "log", "--date=format:%d/%m/%Y %H:%M", "--format=%h|%ad|%s", "--reverse"], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8")
    return [x.split("|", 2) for x in r.stdout.strip().splitlines()]


def construir(ruta):
    D = Documento("Informe de desarrollo del software", "Competencia de manos, neuroevolución y mano bebé del gemelo digital de la prótesis",
                  "IS-GDM-01", fecha="9 de noviembre de 2026")
    D.indice()
    E = estado_actual()
    K = constantes_motor()
    pob, b = E["pob"], E["b"]
    c3, c1, c4, c5, c6, c8, c9 = (R[k]["resultado"] for k in ("C3", "C1", "C4", "C5", "C6", "C8", "C9"))
    giro = {x["giro_deg"]: x["acierto"] for x in c4["giro"]}
    cn = pob.campeon_metricas
    u = b.curva[-1]
    parecidos = sum(h["parecido"] is not None for h in b.habilidades)

    # ------------------------------------------------------------------
    D.h1("Resumen ejecutivo")
    D.p("Entre el 22 de octubre y el 9 de noviembre de 2026 se desarrollaron tres programas de escritorio que hacen aprender a la mano "
        "de la prótesis en un gemelo digital, sin persona ni hardware. Este informe recoge qué se pidió, cuándo, qué se construyó y qué "
        "resultados dio.")
    D.tabla(["Programa", "Qué hace", "Estado guardado", "Mejor resultado"], [
        ["**Competencia de manos**", "10 redes entrenan con gradiente; por generaciones, las peores copian a las mejores y mutan.",
         f"{E['gen']} generaciones (10 electrodos)", f"Campeona #{E['camp'][0]}: {pc(E['camp'][2], 2)} de acierto en el examen"],
        ["**Neuroevolución**", "150 redes nacen casi vacías y evolucionan sin gradiente hasta seguir a una persona.",
         f"{pob.gen} generaciones, {len(pob.especies)} especies", f"Campeón #{cn['n']}: {pc(cn['acierto'])} de acierto"],
        ["**Mano bebé**", "Con 10 músculos, sin metas predeterminadas, descubre sola lo que su cuerpo puede hacer.",
         f"{b.intentos:,} intentos".replace(",", "."), f"{len(b.habilidades)} habilidades; R² del control {n(u[1], 3)}"],
    ], "Los tres programas (estado a la fecha del informe).", anchos=[3.3, 6.2, 3.3, 3.7])
    D.p("**Conclusión.** Los tres programas funcionan, guardan lo aprendido y lo retoman al abrir. La competencia es la que mejor reconoce "
        "gestos y es la candidata para la prótesis; la neuroevolución produce redes mucho más pequeñas, con menos acierto; la mano bebé "
        "es un estudio de aprendizaje autónomo y no controla la prótesis.")
    D.p("**Dos advertencias.** Todos los resultados son en simulación: todavía no hay datos de personas reales. Además, desde el 9 de "
        "noviembre se detectó que Windows bloquea PyTorch en este equipo, por lo que la competencia no puede ejecutarse hasta resolverlo (sección 8).")

    # ------------------------------------------------------------------
    D.h1("Alcance y fuentes")
    D.p("El informe cubre el software de iteración por generaciones (competencia y neuroevolución) y la mano bebé. No cubre el hardware, "
        "el firmware ni el cuaderno de entrenamiento inicial, que se mencionan solo como contexto y están descritos en el Documento técnico "
        "DT-GDM-01. El software se desarrolló con un asistente de programación con inteligencia artificial, a partir de las solicitudes "
        "del equipo; las decisiones de alcance fueron del equipo.")
    D.tabla(["Fuente", "Uso en este informe"], [
        ["Solicitudes al asistente, con fecha y hora de envío (Colombia, UTC−5)", "Cronología (sección 3). Se redactaron formalmente y se omitieron los mensajes de seguimiento."],
        ["Historial del repositorio (git)", "Fechas de cada entrega (anexo B)."],
        ["Registros de las pruebas técnicas (DP-GDM-01)", "Resultados y figuras."],
        ["Estado guardado por cada programa", "Cifras de avance. La competencia corrió por última vez el " + E["ult"] + "."],
        ["Capturas de pantalla del 28/10/2026", "Figuras de las aplicaciones. Muestran el estado de ese día, anterior al estado guardado actual."],
    ], "Fuentes de información.", anchos=[8, 8.5])

    # ------------------------------------------------------------------
    D.h1("Cronología del desarrollo")
    D.p("Todas las fechas son de 2026.")
    D.tabla(["Fecha y hora", "Solicitud del equipo", "Resultado"], [list(x) for x in CRONOLOGIA], "Solicitudes y resultados, en orden.", anchos=[2.6, 8.2, 5.7], tam=9)
    D.tabla(["Fecha", "Competencia", "Neuroevolución", "Mano bebé"], HITOS, "Avance de cada programa por fecha.", anchos=[1.5, 5, 5, 5], tam=9)

    # ------------------------------------------------------------------
    D.h1("Software de iteración por generaciones")
    D.h2("Competencia de manos")
    D.p("**Objetivo.** Mantener una población de redes que mejora sin intervención y conservar siempre la mejor, lista para pasarla al "
        "microcontrolador. Se abre con el acceso directo «Gemelo de la mano».")
    D.p("**Una generación:**")
    D.vinetas([
        "Procesos en paralelo simulan personas nuevas, con sus sesiones y calibraciones, y guardan cada lote en disco.",
        f"Cada una de las {K['N_MANOS']} manos entrena {K['PASOS_POR_GENERACION']} pasos con lotes de {K['LOTE']} ventanas tomadas de lo acumulado (hasta {K['MEMORIA_MAX']:,} en memoria).".replace(",", "."),
        "Todas rinden el mismo examen: 20 personas que ninguna mano vio.",
        f"Las {K['REEMPLAZOS']} peores copian los pesos de una de las {K['REEMPLAZOS']} mejores y mutan su tasa de aprendizaje y otros parámetros; cada {K['CADA_INMIGRANTE']} generaciones entra una mano nueva al azar.",
        f"Si la mejor supera a la campeona por más de {n(K['MEJORA_CAMPEONA'], 3)} de puntaje, se corona y se guarda: modelo, código C para el ESP32, ficha y fotos de sus 7 gestos.",
    ], numeradas=True)
    D.p("El puntaje combina acierto (60 %), pseudo-R² de los gestos (20 %) y R² de la pose de los dedos (20 %). La ventana muestra las 10 "
        "manos InMoov en 3D, la red neuronal con sus activaciones, el recorrido de la señal por los electrodos y el amplificador, las "
        "campeonas y los datos guardados.")
    D.fig(CAPTURAS / "pestana_0.png", "Competencia de manos: las 10 manos, el ranking del examen y el puntaje por generación (captura del 28/10/2026).", 15.5)
    D.fig(CAPTURAS / "pestana_2.png", "Pestaña del gemelo: corte del antebrazo con los 10 electrodos y la señal en cada etapa (captura del 28/10/2026).", 15.5)
    D.p("**Resultados.** Al pasar de 2 a 10 electrodos, el 27 de octubre, la competencia se reinició desde cero y superó rápidamente la mejor marca anterior.")
    D.tabla(["Configuración", "Generaciones", "Mejor acierto", "Pseudo-R²", "R² de la pose"], [
        ["2 electrodos (archivada)", E["g2"], pc(E["a2"], 2), n(c9["2 electrodos"]["pseudo_r2"], 3), n(c9["2 electrodos"]["r2_pose"], 3)],
        [f"10 electrodos, campeona #{c3['campeona']} (usada en las pruebas)", f"{c3['generacion']}", pc(c3["acierto"], 2), n(c3["pseudo_r2"], 3), n(c3["r2_pose"], 3)],
        [f"10 electrodos, campeona #{E['camp'][0]} (la más reciente)", f"{E['camp'][1]}", pc(E["camp"][2], 2), n(E["camp"][3], 3), n(E["camp"][4], 3)],
    ], "Resultados de la competencia en el examen de 20 personas.", anchos=[7.5, 2.2, 2.3, 2.2, 2.3], centrar=[1, 2, 3, 4])
    D.fig(fig("C9_convergencia.png"), "Convergencia con 2 y con 10 electrodos (escala logarítmica de generaciones).", 15.5)
    D.p(f"Con personas nuevas distintas del examen, la campeona #{c3['campeona']} tolera bien ruido y fatiga, pero baja si los electrodos giran después "
        f"de calibrar: {pc(giro[10])} a 10° y {pc(giro[30])} a 30°; recalibrar devuelve al menos {pc(min(x['acierto'] for x in c4['recalibrado']))}. "
        f"Si un electrodo se despega, cae hasta {pc(c5['peor']['campeona'], 0)}. Una red de referencia entrenada simulando electrodos "
        f"despegados se mantuvo sobre {pc(c5['peor']['mlp_con_apagados'], 0)}, pero ese entrenamiento todavía no está en la competencia.")
    D.nota("Nota metodológica: el mismo examen sirve para elegir a la campeona, así que su cifra puede ser algo optimista. Por eso las pruebas de "
           "robustez usan personas simuladas aparte.")

    D.h2("Neuroevolución")
    D.p("**Objetivo.** Responder a la idea del video del 26 de octubre: muchos agentes que empiezan sin saber nada y aprenden por selección "
        "natural, a la vista. Usa NEAT (Stanley y Miikkulainen, 2002), sin gradiente y sin datos etiquetados para entrenar. Se abre con el "
        "acceso directo «Neuroevolución de la mano».")
    D.p("**La pista.** Una persona del gemelo hace 21 gestos al azar (unos 29 s). Cada mano solo escucha los electrodos; si hace el gesto "
        "equivocado medio segundo seguido, choca y se apaga, como los carros del video.")
    D.p("**Una generación:**")
    D.vinetas([
        f"Las {pob.n} redes recorren 6 pistas de personas distintas; una se reproduce en la ventana.",
        "Aptitud de cada red: segundos que aguanta sin chocar + 30 × su acierto.",
        "La mejor rinde un examen con 8 personas que no estaban en las pistas; si mejora, se exporta como campeón.",
        "Las redes se agrupan por parecido en especies; las especies con mejor aptitud tienen más hijos y las que no mejoran en 20 generaciones se extinguen.",
        "Los hijos se cruzan y mutan: cambian pesos, agregan conexiones y agregan neuronas, de modo que la red crece.",
    ], numeradas=True)
    D.fig(CAPTURAS / "neuroevolucion.png", "Neuroevolución: 30 de las 150 manos (color = especie, gris = chocó), la pista y la red de la mejor (captura del 28/10/2026).", 15.5)
    D.p("**Decisión de ajuste.** La primera versión se estancaba. Se compararon configuraciones durante 300 generaciones, entonces con 2 "
        "electrodos, y la clave fue no mutar todos los pesos a la vez. Un clasificador lineal llegaba a 96 % con esos datos, lo que indicó "
        "que el problema estaba en la búsqueda y no en la tarea.")
    D.tabla(["Configuración", "Pesos que mutan", "Entradas iniciales por gesto", "Pistas por generación", "Acierto (mejor de las últimas 10 generaciones)"], AJUSTE,
            "Ajuste de la evolución (generación 300).", anchos=[3, 3.5, 3.3, 2.7, 4], centrar=[1, 2, 3, 4])
    neat_n = len(c6["neat"])
    D.p(f"**Resultados con 10 electrodos.** En {neat_n} ejecuciones independientes de {c6['generaciones']} generaciones se alcanzó {pc(c6['neat_media'])} ± "
        f"{pc(c6['neat_sd'])}. Con las mismas personas, modelos entrenados con gradiente llegan a {pc(c6['logistica_media'])} (regresión logística) y "
        f"{pc(c6['mlp_media'])} (red neuronal). La ejecución de la aplicación llegó a la generación {pob.gen}; su campeón #{cn['n']}, de la generación {cn['gen']}, "
        f"tiene {pc(cn['acierto'])} de acierto y aguanta {n(cn['aguante'])} s de {n(cn['largo'])}, con pseudo-R² de {n(cn['pseudo_r2'], 3)} y R² de la pose de {n(cn['r2_pose'], 3)}.")
    D.fig(fig("C6_neat_vs_gradiente.png"), "Neuroevolución frente a modelos entrenados con gradiente.", 15.5)
    D.fig(fig("T_red_neat.png"), "Red del campeón NEAT: solo usa parte de las 120 entradas y creció unas pocas neuronas.", 14)

    D.h2("Comparación de los dos programas")
    D.tabla(["Aspecto", "Competencia", "Neuroevolución"], [
        ["Cómo aprende", "Gradiente, con gestos etiquetados", "Selección y mutación, sin gradiente"],
        ["Población", f"{K['N_MANOS']} redes", f"{pob.n} redes (30 visibles)"],
        ["Tamaño de la red", "Fijo por genoma (la campeona de las pruebas: 120-48-24 + salidas)", "Crece; menos de 100 conexiones en las pruebas"],
        ["Dónde corre", "GPU (o CPU sin GPU)", "CPU"],
        ["Acierto en su examen de personas nuevas", f"{pc(E['camp'][2], 2)} (20 personas)", f"{pc(cn['acierto'])} (8 personas)"],
        ["Fortaleza", "Reconoce casi perfecto", "Red mínima, fácil de inspeccionar"],
        ["Depende de PyTorch", "Sí (hoy bloqueado)", "No"],
    ], "Competencia frente a neuroevolución.", anchos=[4.5, 6, 6])

    # ------------------------------------------------------------------
    D.h1("Mano bebé")
    D.p("**Objetivo.** Ver qué puede aprender una mano por sí sola, como un bebé. No reconoce señales de electrodos ni controla la prótesis: "
        "es un estudio de aprendizaje autónomo sobre un modelo de 10 músculos y tendones acoplados. Se abre con el acceso directo «Mano "
        "bebé (entrenamiento libre)».")
    D.h2("Evolución: tres versiones")
    D.tabla(["Versión", "Fecha", "Idea", "Por qué cambió"], [
        ["1", "23/10", "Balbuceo y metas propias, muchas cerca de lo que ya sabía; a lo que hallaba le ponía el nombre del gesto conocido más parecido.", "«Está haciendo los mismos gestos» (23/10 19:06)."],
        ["2", "23/10", "Imitaba 33 gestos de personas reales (HaGRID v2); en una prueba aprendió 15 y marcó 18 como imposibles.", "«Le estás dando escenarios predeterminados» (26/10 15:46)."],
        ["3", "26/10", "Autónoma, desde cero, sin lista de gestos ni tope. La versión anterior se guardó como respaldo.", "Versión vigente."],
    ], "Versiones de la mano bebé.", anchos=[1.5, 1.5, 8, 5.5])
    D.h2("Cómo aprende")
    D.vinetas([
        "**Balbuceo** (300 intentos): activa músculos al azar y observa qué pasa.",
        "**Exploración**: se propone sus propias metas y elige entre inventar una pose, llevar más lejos algo que sabe, practicar lo nuevo o balbucear. Prefiere lo que más le está enseñando.",
        "**Habilidad**: una pose que logra repetir 3 veces seguidas. El repertorio no tiene tope; cuando no encuentra nada nuevo, sube de «finura» y distingue detalles más pequeños.",
        "**Espejo de internet**: compara lo que hace con 19.800 manos reales de HaGRID v2 solo para anotar parecidos; no le impone metas.",
    ])
    D.fig(CAPTURAS / "mano_bebe.png", "Mano bebé: decisiones, repertorio, espejo de internet y curvas de R² (captura del 28/10/2026).", 15.5)
    D.h2("Resultados")
    D.p(f"Tres nacimientos independientes de {c8['intentos']:,} intentos dieron entre {min(s['habilidades'] for s in c8['semillas'])} y "
        f"{max(s['habilidades'] for s in c8['semillas'])} habilidades, con R² del control entre {n(min(s['r2_control'] for s in c8['semillas']), 3)} y "
        f"{n(max(s['r2_control'] for s in c8['semillas']), 3)}.".replace(",000", ".000"))
    D.fig(fig("C8_bebe.png"), "Crecimiento del repertorio y R² del control en tres nacimientos.", 15.5)
    D.p(f"El estado guardado de la mano suma {b.intentos:,} intentos, {len(b.habilidades)} habilidades, finura {b.nivel}, R² del control de {n(u[1], 3)}, "
        f"cobertura de {pc(u[3], 0)} de lo que su cuerpo permite, y {parecidos} habilidades con un gesto humano parecido en el espejo.".replace(",", ".", 1))

    # ------------------------------------------------------------------
    D.h1("Verificación")
    D.p("Estas son las pruebas relacionadas con los tres programas; el detalle completo está en DP-GDM-01.")
    ids = ["C1", "C3", "C4", "C5", "C6", "C8", "C9", "G1", "G2", "G3", "G4"]
    D.tabla(["Prueba", "Nombre", "Estado", "Resultado"], [[k, R[k]["nombre"], estado(k), es(R[k]["conclusion"])] for k in ids], "Pruebas relacionadas con este software.", anchos=[1.2, 4, 1.9, 9.4], tam=8.5)

    D.h1("Errores encontrados y corregidos")
    D.tabla(["Fecha", "Programa", "Síntoma", "Causa", "Corrección"], ERRORES, "Errores corregidos durante el desarrollo.", anchos=[1.3, 2.4, 4, 4.6, 4.2], tam=8.5)

    D.h1("Estado actual y limitaciones")
    D.vinetas([
        "**Bloqueo de Windows (09/11/2026).** El Control inteligente de aplicaciones está activo y bloquea librerías de PyTorch: **la competencia no puede ejecutarse** "
        "mientras siga así. La neuroevolución y la mano bebé cargan sin PyTorch, aunque el 09/11 el sistema también bloqueó una vez un archivo de scikit-learn, "
        "que ambas usan. Sus modelos y datos guardados no se ven afectados.",
        "**Solo simulación.** Ningún resultado se ha validado con personas reales. Las pruebas indican que el espectro de la señal simulada es más alto que el del EMG real (pruebas A4 y A6).",
        "**Falla de electrodo.** La campeona no tolera un electrodo despegado; la mitigación probada no se ha incorporado al entrenamiento.",
        "**La neuroevolución no supera al gradiente** con los mismos datos.",
        "**La mano bebé no se conecta a la prótesis.** Sus habilidades viven en su propio modelo de 10 músculos.",
        "**Las capturas son del 28/10/2026** y no reflejan el avance posterior.",
    ])
    D.h1("Trabajo siguiente")
    D.vinetas([
        "Resolver el bloqueo de Windows para volver a ejecutar la competencia.",
        "Incorporar electrodos despegados al entrenamiento de la competencia y de la neuroevolución.",
        "Bajar las bandas de frecuencia del modelo muscular y repetir la validación.",
        "Validar con datos reales cuando exista el hardware.",
    ])

    D.salto()
    D.h1("Anexos")
    D.h2("Anexo A: cómo ejecutar")
    D.tabla(["Programa", "Acceso directo en el escritorio", "Comando"], [
        ["Competencia", "Gemelo de la mano", "python app_gemelo.py"],
        ["Neuroevolución", "Neuroevolución de la mano", "python app_gemelo.py --neuro"],
        ["Mano bebé", "Mano bebé (entrenamiento libre)", "python app_gemelo.py --libre"],
    ], "Ejecución. Los datos se guardan en %USERPROFILE%\\gemelo_mano\\motor\\.", anchos=[3.5, 6.5, 6.5])
    D.h2("Anexo B: historial de entregas")
    D.tabla(["Commit", "Fecha", "Descripción"], [[a, b_, c] for a, b_, c in historial_git()], "Historial completo del repositorio github.com/roldan0411/mano.", anchos=[1.8, 3.2, 11.5], tam=8.5)
    return D.guardar(ruta)


def a_pdf(docx):
    ps = (f"$w = New-Object -ComObject Word.Application; $w.Visible = $false; $d = $w.Documents.Open('{docx}'); "
          "foreach ($t in $d.TablesOfContents) { $t.Update() }; $d.Fields.Update() | Out-Null; $d.Save(); "
          f"$d.ExportAsFixedFormat('{Path(docx).with_suffix('.pdf')}', 17); $d.Close($false); $w.Quit()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


if __name__ == "__main__":
    ruta = construir(AQUI / "Informe_software_IS-GDM-01.docx")
    print("escrito", ruta, flush=True)
    a_pdf(ruta)
    print("pdf", ruta.with_suffix(".pdf"))
