"""
Artículo científico (formato IFAC de la plantilla de la Universidad de Pamplona, 2006)
=====================================================================================
Carta; márgenes 3,5 / 2,5 / 2,3 / 2,3 cm; título Times 16 negrita a 6 cm del borde; autores Times 12;
afiliación Times 11; resumen y palabras clave Times 10; texto Times 10 justificado a dos columnas de
7,5 cm con 1,5 cm de separación; secciones Times 12 negrita, subsecciones Times 10 negrita; títulos de
figuras y tablas centrados, negrita, Times 10; sin numerar páginas ni encabezados o pies.

  python docs/articulo.py          → docs/Articulo_gemelo_mano.docx y .pdf
"""

import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from datos_documentos import FIGURAS, R, n, pc  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_SECTION  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402
from latex2mathml.converter import convert  # noqa: E402
from lxml import etree  # noqa: E402
from word import _XSLT  # noqa: E402

FIG = AQUI / "articulo_figuras"
COL = Cm(7.5)
TNR = "Times New Roman"


class Articulo:
    def __init__(self):
        self.d = Document()
        self.n = {"sec": 0, "sub": 0, "fig": 0, "tab": 0, "ec": 0}
        s = self.d.sections[0]
        s.page_width, s.page_height = Cm(21.59), Cm(27.94)
        s.top_margin, s.bottom_margin, s.left_margin, s.right_margin = Cm(3.5), Cm(2.5), Cm(2.3), Cm(2.3)
        st = self.d.styles["Normal"]
        st.font.name, st.font.size = TNR, Pt(10)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), TNR)
        pf = st.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(0), 1.0

    # ---------- utilidades ----------
    def par(self, texto="", tam=10, negrita=False, centro=False, justificar=True, antes=0, despues=0, cursiva=False):
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if centro else (WD_ALIGN_PARAGRAPH.JUSTIFY if justificar else WD_ALIGN_PARAGRAPH.LEFT)
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(antes), Pt(despues)
        self._runs(p, texto, tam, negrita, cursiva)
        return p

    def _runs(self, p, texto, tam=10, negrita=False, cursiva=False):
        import re
        for parte in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\])", texto):
            if not parte:
                continue
            if parte.startswith("[["):                                 # campo que debe llenar el autor
                r = p.add_run(parte[2:-2])
                r.font.highlight_color = 7                              # amarillo
            elif parte.startswith("**"):
                r = p.add_run(parte[2:-2])
                r.bold = True
            elif parte.startswith("*"):
                r = p.add_run(parte[1:-1])
                r.italic = True
            else:
                r = p.add_run(parte)
            r.font.size, r.font.name = Pt(tam), TNR
            if negrita:
                r.bold = True
            if cursiva:
                r.italic = True
        return p

    def blanco(self, tam=10, k=1):
        for _ in range(k):
            p = self.d.add_paragraph()
            p.add_run("").font.size = Pt(tam)
            p.paragraph_format.line_spacing = Pt(tam * 1.15)

    def columnas(self, num):
        """Sección continua con num columnas (7,5 cm y 1,5 cm de separación)."""
        s = self.d.add_section(WD_SECTION.CONTINUOUS)
        cols = s._sectPr.xpath("./w:cols")
        c = cols[0] if cols else OxmlElement("w:cols")
        if not cols:
            s._sectPr.append(c)
        for hijo in list(c):
            c.remove(hijo)
        c.set(qn("w:num"), str(num))
        if num == 2:
            c.set(qn("w:space"), str(int(Cm(1.5).twips)))
            c.set(qn("w:equalWidth"), "0")
            for _ in range(2):
                col = OxmlElement("w:col")
                col.set(qn("w:w"), str(int(COL.twips)))
                col.set(qn("w:space"), str(int(Cm(1.5).twips)))
                c.append(col)
        else:
            c.set(qn("w:equalWidth"), "1")
        return s

    # ---------- estructura ----------
    def seccion(self, titulo):
        self.n["sec"] += 1
        self.n["sub"] = 0
        return self.par(f"{self.n['sec']}. {titulo}", 12, negrita=True, justificar=False, antes=12, despues=11)

    def subseccion(self, titulo):
        self.n["sub"] += 1
        return self.par(f"{self.n['sec']}.{self.n['sub']}. {titulo}", 10, negrita=True, justificar=False, antes=10, despues=10)

    def texto(self, t):
        return self.par(t, 10, despues=10)                               # doble separación entre párrafos

    def lista(self, items):
        for k, it in enumerate(items):
            p = self.par(it, 10, despues=4 if k < len(items) - 1 else 10)
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.first_line_indent = Cm(-0.35)
            p.runs[0].text = "• " + p.runs[0].text

    def ec(self, latex):
        self.n["ec"] += 1
        p = self.d.add_paragraph()
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(4), Pt(10)
        p.paragraph_format.tab_stops.add_tab_stop(int(COL / 2), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(COL, WD_TAB_ALIGNMENT.RIGHT)
        p.add_run("\t")
        p._p.append(_XSLT(etree.fromstring(convert(latex).encode())).getroot())
        r = p.add_run(f"\t({self.n['ec']})")
        r.font.size = Pt(10)

    def figura(self, ruta, titulo, ancho=COL):
        self.n["fig"] += 1
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before, p.paragraph_format.keep_with_next = Pt(18), True
        p.add_run().add_picture(str(ruta), width=ancho)
        self.par(f"Figura {self.n['fig']}. {titulo}", 10, negrita=True, centro=True, antes=4, despues=18)
        return self.n["fig"]

    def tabla(self, titulo, encabezado, filas, anchos):
        self.n["tab"] += 1
        p = self.par(f"Tabla {self.n['tab']}. {titulo}", 10, negrita=True, centro=True, antes=18, despues=4)
        p.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(encabezado))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for fila, datos in ((t.rows[0], encabezado), *[(None, f) for f in filas]):
            celdas = (fila or t.add_row()).cells
            for k, v in enumerate(datos):
                celdas[k].text = ""
                q = celdas[k].paragraphs[0]
                self._runs(q, str(v), 8, negrita=fila is not None)
                q.alignment = WD_ALIGN_PARAGRAPH.LEFT if k == 0 else WD_ALIGN_PARAGRAPH.CENTER
                celdas[k].width = Cm(anchos[k])
        self.blanco(10)
        return self.n["tab"]

    def referencias(self, refs):
        for clave, texto in refs:
            p = self.d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Cm(0.6), Cm(-0.6)
            p.paragraph_format.space_after = Pt(4)
            self._runs(p, f"[{clave}] {texto}", 9)


def cientifico(v):
    """1.6e-05 → 1,6·10⁻⁵"""
    m, e = f"{v:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m.replace('.', ',')}·10{str(int(e)).translate(sup)}"


def construir(ruta):
    A = Articulo()
    c1, c3, c4, c5, c6, c7, c8, c9 = (R[k]["resultado"] for k in ("C1", "C3", "C4", "C5", "C6", "C7", "C8", "C9"))
    b2, d1, d2, d4 = (R[k]["resultado"] for k in ("B2", "D1", "D2", "D4"))
    giro = {x["giro_deg"]: x["acierto"] for x in c4["giro"]}
    recal = min(x["acierto"] for x in c4["recalibrado"])
    cumplen = sum(1 for k in R if R[k]["cumple"])

    # ---------- cabecera (una columna) ----------
    t = A.par("Gemelo digital de una prótesis mioeléctrica de mano con diez electrodos: reconocimiento de gestos mediante "
              "competencia de redes, neuroevolución y aprendizaje autónomo", 16, negrita=True, centro=True, antes=71)
    A.blanco(10, 2)
    A.par("Roldán Rodríguez", 12, centro=True)
    A.par("Código: 01251152003", 12, centro=True)
    A.blanco(10)
    for linea in ("[[Universidad]], Facultad de [[Facultad]],", "Programa de Ingeniería en Ciencias Computacionales e Inteligencia Artificial",
                  "Bucaramanga, Santander, Colombia", "roldanrodriguez11@gmail.com", "Noviembre-2026"):
        A.par(linea, 11, centro=True)
    A.blanco(10, 3)
    A.par("Resumen", 12, negrita=True, justificar=False, despues=6)
    A.par("Se presenta el avance del diseño de una prótesis mioeléctrica de mano de bajo costo, impresa en 3D y controlada por un "
          "ESP32 que lee diez electrodos de superficie del antebrazo (cinco sobre los flexores y cinco sobre los extensores). Para "
          "entrenar y verificar el sistema antes de contar con datos reales se construyó un gemelo digital que simula diez músculos, el "
          "acople geométrico músculo–electrodo, el amplificador de instrumentación, el filtro, el conversor y las diferencias entre "
          "personas y sesiones. Sobre él se comparan tres formas de aprendizaje: una competencia de diez redes entrenadas con "
          "gradiente (Population-Based Training), una neuroevolución NEAT de 150 redes que aprenden desde cero y una mano que "
          f"descubre de forma autónoma lo que su cuerpo puede hacer. Con personas que el modelo nunca vio, la red campeona alcanza "
          f"un acierto de {pc(c3['acierto'], 2)} en siete gestos (pseudo-R² de McFadden {n(c3['pseudo_r2'], 3)}; R² de la pose "
          f"{n(c3['r2_pose'], 3)}); pasar de dos a diez electrodos sube el acierto de {pc(c1['acierto']['2'])} a "
          f"{pc(c1['acierto']['10'])} y el R² de la pose de {n(c1['r2_pose']['2'], 3)} a {n(c1['r2_pose']['10'], 3)}. El firmware "
          f"reproduce exactamente las predicciones del gemelo y se estima en {n(d2['esp32_total_ms'], 1)} ms por ventana en el ESP32. "
          f"De 26 pruebas técnicas, {cumplen} cumplen; las restantes señalan el trabajo pendiente: el espectro simulado, la falla de "
          "un electrodo y la latencia.", 10)
    A.blanco(10, 2)
    A.par("**Palabras clave:** Electromiografía de superficie, Gemelo digital, Prótesis mioeléctrica, Redes neuronales, "
          "Neuroevolución, Sistemas embebidos.", 10)
    A.par("", 10, antes=40)                                              # el texto empieza a 2 cm de las palabras clave

    # ---------- cuerpo (dos columnas) ----------
    A.columnas(2)
    A.seccion("Introducción")
    A.texto("Una prótesis mioeléctrica interpreta la actividad eléctrica de los músculos del muñón para mover una mano artificial. "
            "El enfoque de reconocimiento de patrones —extraer características de ventanas cortas de la electromiografía de superficie "
            "(sEMG) y clasificarlas— se estableció con las características de dominio del tiempo de [Hudgins93] y se volvió práctico "
            "en tiempo real con esquemas como el de [Englehart03]; revisiones posteriores confirman que esas características siguen "
            "entre las más útiles [Phinyomark12]. Hoy existen brazaletes de bajo costo con varios canales, como el que se usó para "
            "registrar el conjunto de datos de [Lobov18].")
    A.texto("Dos problemas encarecen el desarrollo: registrar EMG de muchas personas requiere tiempo, equipo y consentimiento, y los "
            "modelos se degradan cuando cambian la persona, la piel o la posición de los electrodos [Lobov18]. Un gemelo digital —una "
            "réplica en software del sistema físico que comparte sus modelos y su cadena de procesamiento [Grieves17][Tao19]— permite "
            "entrenar y probar antes de tener el hardware, siempre que su fidelidad se verifique.")
    A.texto("Este artículo reporta el avance del proyecto y sus aportes:")
    A.lista([
        "Un gemelo digital del antebrazo con diez electrodos, acople geométrico y la cadena de adquisición por etapas (AD620, "
        "pasa-banda y ADC de 12 bits), con personas y sesiones variables.",
        "Una competencia de diez redes entrenadas sin parar con Population-Based Training [Jaderberg17], con métricas R² siempre visibles.",
        "Una neuroevolución con NEAT [Stanley02] en la que 150 redes aprenden desde cero a seguir la señal de una persona.",
        "Una mano que aprende de forma autónoma con exploración guiada por metas propias y curiosidad [Rolf10][Baranes13].",
        "Un firmware para el ESP32 verificado contra el gemelo y una batería de 26 pruebas técnicas.",
    ])
    A.texto("La sección 2 describe el sistema y los métodos, la sección 3 los resultados, la sección 4 su discusión y la sección 5 "
            "las conclusiones y el trabajo futuro.")

    A.seccion("Materiales y Métodos")
    A.subseccion("Arquitectura del sistema")
    A.texto("La prótesis usa la mano InMoov i2 [Langevin] con cinco servos de dedos y uno de muñeca. Cada electrodo pasa por un "
            "AD620 [AD620] y un pasa-banda activo; un multiplexor CD74HC4067 lleva los diez canales a un pin del ADC del ESP32, que "
            "calcula las características, ejecuta la red y mueve los servos. El gemelo digital replica cada etapa en Python (Fig. 1).")
    A.columnas(1)
    A.figura(FIGURAS / "T_arquitectura.png", "Arquitectura: cadena física de la prótesis (arriba) y su réplica en el gemelo digital (abajo).", Cm(16.5))
    A.columnas(2)
    A.subseccion("Antebrazo y electrodos")
    A.texto("Cada músculo m genera ruido blanco filtrado en su banda, modulado por su activación, la fatiga y la ganancia propia de "
            "cada persona. Los músculos y los electrodos ocupan un ángulo θ en el corte del antebrazo (Fig. 2) y los músculos una "
            "profundidad p. El acople del músculo m al electrodo c es:")
    A.ec(r"w_{cm}\propto e^{-\Delta\theta_{cm}^{2}/2\sigma^{2}}\,e^{-(1-p_m)/\lambda}\,e^{-|z_c|/\lambda_z}")
    A.texto("con σ = 38°, λ = 0,35 y λz = 3 cm, más un término de conducción de volumen, normalizado por electrodo. En cada sesión "
            "los diez electrodos giran juntos alrededor del brazo y cada uno queda además un poco corrido, en proporción al nivel de "
            "dificultad de la persona (cinco niveles de ruido, interferencia de 60 Hz, desplazamiento, fatiga y variabilidad).")
    A.figura(FIG / "corte.png", "Corte del antebrazo: diez músculos (radio = profundidad) y diez electrodos (negro: flexores; gris: extensores).")
    A.subseccion("Cadena de adquisición")
    A.texto("La tensión de cada electrodo, sumada al desfase continuo de los electrodos δc, se amplifica con la ganancia del AD620 y "
            "se recorta en sus rieles; luego un pasa-banda inversor (R1 = 10 kΩ, C1 = 1 µF, Rf = 100 kΩ, Cf = 3,3 nF) fija los cortes "
            "y una ganancia de 10, y la señal se desplaza a 1,65 V para el ADC de 12 bits a 1 kHz:")
    A.ec(r"G = 1 + \frac{49{,}4\ \mathrm{k\Omega}}{R_G}\approx 106")
    A.ec(r"f_L = \frac{1}{2\pi R_1 C_1}\approx 15{,}9\ \mathrm{Hz},\quad f_H = \frac{1}{2\pi R_f C_f}\approx 482\ \mathrm{Hz}")
    A.subseccion("Características, calibración y contexto")
    A.texto("Cada 100 ms se toma una ventana de N = 200 muestras por electrodo y se calculan MAV, RMS, ZC y WL [Hudgins93]: 40 "
            "valores. Al colocar los electrodos la persona hace cada gesto una vez; con esas ventanas C se calcula una referencia por "
            "característica y todas se expresan relativas a ella:")
    A.ec(r"\tilde{F}_k = \ln(1+F_k) - \frac{1}{|C|}\sum_{w\in C}\ln\left(1+F_k^{(w)}\right)")
    A.texto("La entrada de la red junta las tres últimas ventanas calibradas: 120 valores.")
    A.subseccion("Modelos de aprendizaje")
    A.texto("**Competencia (PBT).** Diez redes densas (normalización, una o dos capas ReLU, salidas de gesto y de pose de seis grados "
            "de libertad) se entrenan por turnos en la GPU con entropía cruzada más error cuadrático de la pose. Tras cada generación "
            "rinden un examen fijo de 20 personas que ninguna vio; las tres peores copian a una de las tres mejores y mutan su genoma "
            "[Jaderberg17]. El puntaje combina acierto y ajuste:")
    A.ec(r"S = 0{,}6\,\mathrm{acierto} + 0{,}2\,R^{2}_{\mathrm{McF}} + 0{,}2\,R^{2}_{\mathrm{pose}}")
    A.texto("**Neuroevolución (NEAT).** 150 redes nacen casi vacías y evolucionan sin gradiente: mutan pesos, agregan conexiones y "
            "neuronas, se cruzan alineando sus genes por innovación y se agrupan en especies [Stanley02]. La «pista» es una persona "
            "del gemelo que hace 21 gestos al azar; una red choca si se equivoca 0,5 s seguidos. Su aptitud, en seis personas por "
            "generación, es:")
    A.ec(r"f = \bar{T}_{\mathrm{sin\ chocar}} + 30\cdot\mathrm{acierto}")
    A.texto("**Aprendizaje contrastivo.** Un codificador 1D-CNN sobre la señal cruda se entrena sin etiquetas acercando dos vistas "
            "físicas de la misma contracción (electrodos recolocados) con la pérdida NT-Xent [Chen20]; existe también la variante "
            "supervisada [Khosla20].")
    A.texto("**Mano autónoma.** Una mano virtual controla diez músculos y una biomecánica de tendones acoplados que no conoce. Aprende "
            "un modelo inverso y uno directo en línea, se propone sus propias metas y elige su estrategia según el progreso que le da "
            "cada una [Rolf10][Baranes13]. Una pose que logra repetir tres veces es una habilidad; su repertorio no tiene tope.")
    A.subseccion("Firmware")
    A.texto("El ESP32 lee los diez electrodos por el multiplexor a 1 kHz, calibra al encender y ejecuta en C las mismas operaciones "
            "que Python; la red exportada decide cada 100 ms y un filtro mantiene el gesto si la confianza es menor a 0,60 y aplica "
            "mayoría de tres decisiones.")
    A.subseccion("Protocolo de evaluación")
    A.texto("Toda evaluación de reconocimiento usa personas que el modelo no vio. Se reportan el acierto, el R² de la pose y el "
            "pseudo-R² de McFadden [McFadden74] respecto de adivinar uniformemente entre siete gestos:")
    A.ec(r"R^{2}_{\mathrm{McF}} = 1-\frac{\ln\hat{L}_{\mathrm{modelo}}}{\ln\hat{L}_{\mathrm{nulo}}}")
    A.texto("Se ejecutaron 26 pruebas en cinco grupos: fidelidad del gemelo, procesamiento, modelos, embebido y software, en una "
            "portátil con GPU RTX 4050 (PyTorch 2.14 con CUDA 12.6). Las comparaciones usan cinco particiones por persona y prueba t "
            "pareada, o seis semillas y prueba de Wilcoxon. No se incluyen pruebas con hardware ni con personas reales.")

    A.seccion("Resultados")
    A.subseccion("Fidelidad del gemelo")
    A.texto(f"Cada electrodo capta como máximo el músculo superficial de su grupo más cercano; la ganancia simulada del AD620 "
            f"coincide con la fórmula del fabricante (error {n(R['A2']['resultado']['error_max_pct'], 3)} %) y los cortes del filtro "
            f"quedan a {n(R['A3']['resultado']['digital']['f_baja'])} y {R['A3']['resultado']['digital']['f_alta']:.0f} Hz (diseño: "
            "15,9 y 482 Hz). Frente al EMG real de [Lobov18], la forma de la señal coincide (cociente RMS/MAV "
            f"{n(R['A6']['resultado']['rms_mav']['sim_mediana'], 3)} frente a {n(R['A6']['resultado']['rms_mav']['real_mediana'], 3)}), pero "
            f"el espectro simulado está corrido hacia arriba (frecuencia mediana {R['A6']['resultado']['mdf']['sim_mediana']:.0f} frente a "
            f"{R['A6']['resultado']['mdf']['real_mediana']:.0f} Hz en la misma banda).")
    A.subseccion("Reconocimiento de gestos")
    A.texto(f"La red campeona (120-48-24 con salidas de gesto y pose) obtuvo en las 20 personas del examen un acierto de "
            f"{pc(c3['acierto'], 2)} (IC 95 %: {pc(c3['ic95_acierto'][0])}–{pc(c3['ic95_acierto'][1])}), un recall mínimo de "
            f"{pc(c3['recall_min'])} (puño), un pseudo-R² de {n(c3['pseudo_r2'], 3)} y un R² de la pose de {n(c3['r2_pose'], 3)} "
            "(Fig. 3).")
    A.figura(FIG / "confusion.png", "Matriz de confusión de la campeona en 20 personas nuevas (% por fila).")
    A.subseccion("Diez electrodos frente a dos")
    A.texto(f"Con el mismo modelo y las mismas personas, dos electrodos (uno sobre flexores y otro sobre extensores, el diseño "
            f"inicial) logran {pc(c1['acierto']['2'])} y diez, {pc(c1['acierto']['10'])} (p = {cientifico(c1['acierto']['p'])}); la mayor "
            f"mejora está en el R² de la pose ({n(c1['r2_pose']['2'], 3)} → {n(c1['r2_pose']['10'], 3)}) (Fig. 4). La competencia "
            f"con diez electrodos supera el 99 % desde la generación {c9['10 electrodos']['gen_99']}, mientras con dos electrodos la "
            f"mejor marca fue {pc(c9['2 electrodos']['mejor_acierto'])} tras {c9['2 electrodos']['generaciones']} generaciones.")
    A.figura(FIG / "dos_vs_diez.png", "Dos frente a diez electrodos (media y desviación en cinco particiones por persona).")
    A.subseccion("Robustez")
    A.texto(f"La calibración de la sesión sube el acierto en personas nuevas de {pc(b2['sin_calibrar_media'])} a "
            f"{pc(b2['calibrado_media'])}. El ruido (hasta cinco veces el nominal) y la fatiga casi no afectan; lo que más degrada es "
            f"que los electrodos giren después de calibrar: {pc(giro[10])} a 10° y {pc(giro[30])} a 30°, y recalibrar devuelve al "
            f"menos {pc(recal)} (Fig. 5). Si un electrodo se despega, la red cae hasta {pc(c5['peor']['campeona'], 0)}; entrenada con "
            f"electrodos apagados al azar se mantiene sobre {pc(c5['peor']['mlp_con_apagados'], 0)} (Fig. 6).")
    A.figura(FIG / "robustez.png", "Acierto cuando los electrodos giran, sin recalibrar y recalibrando.")
    A.figura(FIG / "falla.png", "Acierto con cada electrodo despegado después de calibrar.")
    A.subseccion("Neuroevolución, aprendizaje contrastivo y mano autónoma")
    A.texto(f"NEAT pasó de redes casi vacías a {pc(c6['neat_media'])} ± {pc(c6['neat_sd'])} en {c6['generaciones']} generaciones "
            f"(seis semillas), con redes de unas {round(sum(f['conexiones'] for f in c6['neat']) / len(c6['neat']))} conexiones; con "
            f"los mismos datos, la regresión logística logra {pc(c6['logistica_media'])} y un MLP {pc(c6['mlp_media'])} "
            f"(Wilcoxon p = {n(c6['wilcoxon_p'], 3)}) (Fig. 7). El codificador contrastivo, sin etiquetas, alcanzó "
            f"{pc(c7['tabla'][0]['contrastivo'])} con el 1 % de las etiquetas, por debajo de las características clásicas "
            f"({pc(c7['tabla'][0]['caracteristicas'])}). La mano autónoma descubrió {min(s['habilidades'] for s in c8['semillas'])}–"
            f"{max(s['habilidades'] for s in c8['semillas'])} habilidades en 15 000 intentos en tres nacimientos, con R² del control "
            f"entre {n(min(s['r2_control'] for s in c8['semillas']), 3)} y {n(max(s['r2_control'] for s in c8['semillas']), 3)}.")
    A.figura(FIG / "neat.png", "Neuroevolución frente a modelos entrenados con gradiente (mismas personas).")
    A.subseccion("Firmware")
    A.texto(f"Compilado en el PC con los mismos archivos del ESP32, el firmware dio el mismo gesto que Python en el "
            f"{pc(d1['peor']['mismo_gesto_red'], 0)} de 1 470 ventanas (diferencia máxima {cientifico(d1['peor']['dif_neat'])}). La red "
            f"requiere {d2['mac_red']} multiplicaciones-acumulaciones y {round(d2['flash_modelo_bytes'] / 1024)} KB, con un tiempo "
            f"estimado de {n(d2['esp32_total_ms'], 1)} ms por ventana. La latencia mediana del gesto a la decisión fue de "
            f"{d4['mediana_ms']:.0f} ms (200 ms sin el filtro de mayoría).")
    A.tabla("Resumen de resultados (personas no vistas).", ["Indicador", "Valor"], [
        ["Acierto (7 gestos)", pc(c3["acierto"], 2)],
        ["Pseudo-R² · R² pose", f"{n(c3['pseudo_r2'], 3)} · {n(c3['r2_pose'], 3)}"],
        ["2 → 10 electrodos", f"{pc(c1['acierto']['2'])} → {pc(c1['acierto']['10'])}"],
        ["Calibración", f"{pc(b2['sin_calibrar_media'])} → {pc(b2['calibrado_media'])}"],
        ["Giro 30°: sin / con recal.", f"{pc(giro[30])} / {pc(c4['recalibrado'][-1]['acierto'])}"],
        ["NEAT · MLP", f"{pc(c6['neat_media'])} · {pc(c6['mlp_media'])}"],
        ["Firmware = Python", pc(d1["peor"]["mismo_gesto_red"], 0)],
        ["Latencia mediana", f"{d4['mediana_ms']:.0f} ms"],
    ], [4.2, 3.3])

    A.seccion("Discusión")
    A.texto("Los resultados muestran que, en simulación, diez electrodos y la calibración de la sesión dan un reconocimiento casi "
            "perfecto en personas nuevas, y que la información adicional se nota sobre todo en la estimación de la pose de los dedos. "
            "La principal limitación es que todo se mide en el gemelo: su espectro es más alto que el del EMG real, así que las cifras "
            "absolutas deben confirmarse con personas. Dos hallazgos orientan el diseño físico: la red debe entrenarse con electrodos "
            "que pierden contacto —crítico si se usan electrodos secos en un brazalete— y conviene recalibrar cada vez que se coloca el "
            "brazalete. La latencia de 300 ms está en el límite; [Farrell07] ubica el retardo óptimo de una prótesis entre 100 y 125 ms, "
            "y una ventana de 150 ms, que en las pruebas no redujo el acierto, ayudaría a acercarse. La neuroevolución no supera al "
            "gradiente, pero produce redes de menos de cien conexiones fáciles de interpretar; y el aprendizaje contrastivo no aportó con "
            "tan pocas épocas frente a características ya muy informativas.")
    A.seccion("Conclusiones y Trabajo Futuro")
    A.texto("Se construyó y verificó un gemelo digital de una prótesis mioeléctrica con diez electrodos que permite entrenar redes "
            "pequeñas para un ESP32 antes de disponer de datos reales, con un firmware que reproduce exactamente al gemelo. Como trabajo "
            "futuro se propone: ajustar el espectro del modelo muscular y revalidarlo con EMG real; entrenar con electrodos "
            "despegados al azar; reducir la latencia; construir el brazalete y medir en la placa el tiempo de inferencia; y validar con "
            "personas mediante un conjunto de datos propio y pruebas funcionales.")
    A.par("Agradecimientos", 12, negrita=True, justificar=False, antes=12, despues=11)
    A.texto("El autor agradece al programa de Ingeniería en Ciencias Computacionales e Inteligencia Artificial y a [[docentes / "
            "asignatura]] por el apoyo durante el desarrollo del proyecto.")
    A.par("Referencias", 12, negrita=True, justificar=False, antes=12, despues=11)
    A.referencias([
        ("AD620", "Analog Devices. 'AD620: Low Cost Low Power Instrumentation Amplifier'. Hoja de datos."),
        ("Baranes13", "A. Baranes, P.-Y. Oudeyer. 'Active learning of inverse models with intrinsically motivated goal exploration in "
                      "robots'. Robotics and Autonomous Systems, 61(1). pp. 49-73. (2013)."),
        ("Chen20", "T. Chen, S. Kornblith, M. Norouzi, G. Hinton. 'A Simple Framework for Contrastive Learning of Visual "
                   "Representations'. Proceedings of ICML. (2020)."),
        ("Englehart03", "K. Englehart, B. Hudgins. 'A robust, real-time control scheme for multifunction myoelectric control'. IEEE "
                        "Transactions on Biomedical Engineering, 50(7). pp. 848-854. (2003)."),
        ("Farrell07", "T. R. Farrell, R. F. Weir. 'The optimal controller delay for myoelectric prostheses'. IEEE Transactions on "
                      "Neural Systems and Rehabilitation Engineering, 15(1). pp. 111-118. (2007)."),
        ("Grieves17", "M. Grieves, J. Vickers. 'Digital Twin: Mitigating Unpredictable, Undesirable Emergent Behavior in Complex "
                      "Systems'. En Transdisciplinary Perspectives on Complex Systems. pp. 85-113. Ed. Springer. (2017)."),
        ("Hudgins93", "B. Hudgins, P. Parker, R. N. Scott. 'A new strategy for multifunction myoelectric control'. IEEE Transactions "
                      "on Biomedical Engineering, 40(1). pp. 82-94. (1993)."),
        ("Jaderberg17", "M. Jaderberg et al. 'Population Based Training of Neural Networks'. arXiv:1711.09846. (2017)."),
        ("Khosla20", "P. Khosla et al. 'Supervised Contrastive Learning'. Advances in Neural Information Processing Systems 33. (2020)."),
        ("Langevin", "G. Langevin. 'InMoov: open-source 3D printed life-size robot'. inmoov.fr."),
        ("Lobov18", "S. Lobov, N. Krilova, I. Kastalskiy, V. Kazantsev, V. A. Makarov. 'Latent Factors Limiting the Performance of "
                    "sEMG-Interfaces'. Sensors, 18(4). 1122. (2018)."),
        ("McFadden74", "D. McFadden. 'Conditional logit analysis of qualitative choice behavior'. En P. Zarembka (ed.), Frontiers in "
                       "Econometrics. pp. 105-142. Ed. Academic Press. (1974)."),
        ("Phinyomark12", "A. Phinyomark, P. Phukpattaranont, C. Limsakul. 'Feature reduction and selection for EMG signal "
                         "classification'. Expert Systems with Applications, 39(8). pp. 7420-7431. (2012)."),
        ("Rolf10", "M. Rolf, J. J. Steil, M. Gienger. 'Goal babbling permits direct learning of inverse kinematics'. IEEE "
                   "Transactions on Autonomous Mental Development, 2(3). pp. 216-229. (2010)."),
        ("Stanley02", "K. O. Stanley, R. Miikkulainen. 'Evolving Neural Networks through Augmenting Topologies'. Evolutionary "
                      "Computation, 10(2). pp. 99-127. (2002)."),
        ("Tao19", "F. Tao, H. Zhang, A. Liu, A. Y. C. Nee. 'Digital Twin in Industry: State-of-the-Art'. IEEE Transactions on "
                  "Industrial Informatics, 15(4). pp. 2405-2415. (2019)."),
    ])
    A.par("Páginas consultadas: archive.ics.uci.edu (EMG data for gestures) · github.com/hukenovs/hagrid · github.com/roldan0411/mano",
          9, justificar=False, antes=4)
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    A.d.save(ruta)
    return ruta


def a_pdf(docx):
    ps = (f"$w = New-Object -ComObject Word.Application; $w.Visible = $false; $d = $w.Documents.Open('{docx}'); "
          f"$d.ExportAsFixedFormat('{Path(docx).with_suffix('.pdf')}', 17); $d.Close($false); $w.Quit()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


if __name__ == "__main__":
    ruta = construir(AQUI / "Articulo_gemelo_mano.docx")
    print("escrito", ruta)
    a_pdf(ruta)
    print("pdf", ruta.with_suffix(".pdf"))
