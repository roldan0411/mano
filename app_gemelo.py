"""
============================================================
 PRÓTESIS MIOELÉCTRICA CON RECONOCIMIENTO DE GESTOS POR IA
 Gemelo de la mano — aplicación de escritorio
------------------------------------------------------------
 Roldán Rodríguez — Ing. CCIA
============================================================

Abre la ventana y arranca el motor de entrenamiento en un proceso aparte (GPU).
Mientras la ventana esté abierta, el gemelo genera datos, los guarda y las 10
manos entrenan y compiten sin parar. Al cerrar, se guarda la población y al
volver a abrir continúa donde quedó.

Abrir: doble clic en "Gemelo de la mano.bat", o
  C:\\Users\\rolda\\gemelo_mano\\venv\\Scripts\\python.exe app_gemelo.py

Requisitos: cargador conectado (en batería Windows apaga la GPU y el motor se pausa).
"""

import os
import sys

# Antes de cualquier import pesado: los procesos hijos heredan estas variables
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    import multiprocessing as mp

    from PySide6 import QtGui, QtWidgets

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gemelo.escritorio.lanzador import correr_motor
    from gemelo.escritorio.ventana import QSS, Ventana

    solo_libre = "--libre" in sys.argv                # solo la mano bebé, sin la competencia ni la GPU
    solo_neuro = "--neuro" in sys.argv                # solo la neuroevolución (CPU)
    if not (solo_libre or solo_neuro):
        ctx = mp.get_context("spawn")
        salida, entrada = ctx.Queue(maxsize=6), ctx.Queue()
        motor = ctx.Process(target=correr_motor, args=(salida, entrada), name="motor-gemelo", daemon=False)
        motor.start()

    if sys.platform == "win32":                     # la barra de tareas muestra el ícono propio, no el de Python
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RoldanRodriguez.GemeloDeLaMano")
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Gemelo de la mano")
    app.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemelo", "escritorio", "icono.ico")))
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    if solo_libre:
        from gemelo.escritorio.libre import VentanaLibre
        ventana = VentanaLibre(solo=True)
    elif solo_neuro:
        from gemelo.escritorio.neuro import VentanaNeuro
        ventana = VentanaNeuro(solo=True)
    else:
        ventana = Ventana(salida, entrada, motor)
    ventana.show()
    codigo = app.exec()
    if not (solo_libre or solo_neuro) and motor.is_alive():
        motor.terminate()
    sys.exit(codigo)


if __name__ == "__main__":
    main()
