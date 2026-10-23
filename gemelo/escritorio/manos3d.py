"""
Las 10 manos InMoov en 3D (OpenGL, dentro de la ventana)
========================================================

Cada mano son las piezas STL de la InMoov i2 armadas con la misma cinemática
del gemelo (gemelo.mano_inmoov.poses_mundo). Encima de cada una, la pose
objetivo en azul transparente.
"""

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.shaders import FragmentShader, ShaderProgram, VertexShader
from pyqtgraph.Qt import QtCore, QtGui

from .. import config, mano_inmoov

CACHE = config.RAIZ_DATOS / "motor" / "mano_ligera.npz"
SEPARACION_X, FILA_ARRIBA, FILA_ABAJO = 270.0, 150.0, -330.0
COLOR_MANO = (0.90, 0.88, 0.83, 1.0)
COLOR_LIDER = (0.60, 0.77, 1.0, 1.0)
COLOR_OBJETIVO = (0.30, 0.60, 1.0, 0.13)

# Luz de frente (desde la cámara) y de dos caras, con luz ambiente: el plástico de la InMoov se ve claro
VERTICE = """
    uniform mat4 u_mvp;
    uniform mat3 u_normal;
    attribute vec4 a_position;
    attribute vec3 a_normal;
    attribute vec4 a_color;
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        v_normal = normalize(u_normal * a_normal);
        v_color = a_color;
        gl_Position = u_mvp * a_position;
    }
"""
FRAGMENTO = """
    #ifdef GL_ES
    precision mediump float;
    #endif
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        vec3 n = normalize(v_normal);
        float frente = abs(n.z);
        float llave = max(dot(n, normalize(vec3(0.45, 0.6, 0.65))), 0.0);
        vec3 rgb = v_color.rgb * (0.38 + 0.42 * frente + 0.28 * llave);
        gl_FragColor = vec4(min(rgb, vec3(1.0)), v_color.a);
    }
"""
LUZ = ShaderProgram("mano_luz_frontal", [VertexShader(VERTICE), FragmentShader(FRAGMENTO)])
# la mano del gemelo tiene +Y hacia los dedos; en la escena +Z es arriba
GIRO = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def cargar_piezas():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["piezas"].item(), d["plantilla"].item()
    piezas, plantilla = mano_inmoov.construir(caras_falange=900, caras_palma=2400)
    datos = {k: (np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)) for k, m in piezas.items()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, piezas=np.array(datos, dtype=object), plantilla=np.array(plantilla, dtype=object))
    return datos, plantilla


def _qm(m):
    return QtGui.QMatrix4x4(*np.asarray(m, float).ravel().tolist())


class Mano:
    def __init__(self, vista, piezas, plantilla, base, color, opciones="opaque", sombreado=LUZ):
        self.plantilla, self.base = plantilla, base
        self.items = {}
        for k, (V, F) in piezas.items():
            it = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=V, faces=F), smooth=True, shader=sombreado,
                               color=color, glOptions=opciones)
            vista.addItem(it)
            self.items[k] = it
        self.color = color

    def colorear(self, color):
        if color != self.color:
            for it in self.items.values():
                it.setColor(color)
            self.color = color

    def poner(self, dof):
        T = mano_inmoov.poses_mundo(np.asarray(dof, float), self.plantilla)
        for k, it in self.items.items():
            it.setTransform(_qm(self.base @ T[k]))


class EscenaManos(gl.GLViewWidget):
    def __init__(self, n=10, parent=None):
        super().__init__(parent)
        self.setBackgroundColor((22, 28, 34))
        self.setCameraPosition(distance=1850, elevation=6, azimuth=90)
        self.opts["center"] = QtGui.QVector3D(0, 0, 45)              # deja ver las etiquetas de la fila de arriba
        piezas, plantilla = cargar_piezas()
        self.reales, self.objetivos, self.etiquetas, self.posiciones = [], [], [], []
        fuente = QtGui.QFont("Segoe UI", 10)
        fuente_b = QtGui.QFont("Segoe UI", 10, QtGui.QFont.Weight.DemiBold)
        for k in range(n):
            x = -((k % 5) - 2) * SEPARACION_X              # la cámara mira desde +y: mano 1 a la izquierda
            z = FILA_ARRIBA if k < 5 else FILA_ABAJO
            base = np.eye(4)
            base[:3, 3] = (x, 0, z)
            base = base @ GIRO
            self.posiciones.append((x, z))
            antebrazo = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=1, cols=28, radius=[34.0, 29.0], length=110.0),
                                      smooth=True, shader=LUZ, color=(0.42, 0.45, 0.5, 1.0))
            antebrazo.translate(x, 0, z - 112)
            self.addItem(antebrazo)
            self.reales.append(Mano(self, piezas, plantilla, base, COLOR_MANO))
            self.objetivos.append(Mano(self, piezas, plantilla, base, COLOR_OBJETIVO, "translucent"))
            centro = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignBottom
            par = []
            for dz, f in ((262, fuente_b), (236, fuente)):
                et = gl.GLTextItem(pos=(x, 90, z + dz), text="", color=(230, 236, 240, 255), font=f, alignment=centro)
                self.addItem(et)
                par.append(et)
            self.etiquetas.append(par)
        for m in self.reales + self.objetivos:
            m.poner([0.25, 0.25, 0.25, 0.25, 0.25, 0.0])

    def actualizar(self, poses, objetivo, textos, lider):
        for k, m in enumerate(self.reales):
            m.poner(poses[k])
            m.colorear(COLOR_LIDER if k == lider else COLOR_MANO)
            self.objetivos[k].poner(objetivo)
            arriba, abajo = textos[k]
            color = (140, 190, 255, 255) if k == lider else (230, 236, 240, 255)
            self.etiquetas[k][0].setData(text=arriba, color=color)
            self.etiquetas[k][1].setData(text=abajo, color=(120, 220, 160, 255) if abajo.startswith("✓") else (240, 120, 110, 255))
