# Prótesis mioeléctrica con reconocimiento de gestos por IA — gemelo digital

Roldán Rodríguez — Ing. CCIA

Mano impresa con 5 servos de dedos y un servo de muñeca, controlada por un ESP32 que lee **2 canales EMG** del antebrazo (AD620 + pasa-banda, ADC de 12 bits a 1 kHz) y reconoce 7 gestos: reposo, abierta, puño, pinza, señalar, flexión y extensión de muñeca.

El **gemelo digital** simula todo el sistema (10 músculos, electrodos, circuito, ADC, servos, tendones) para que la IA aprenda sin persona ni hardware; lo aprendido se pasa al ESP32.

## Estructura

```
notebooks/gemelo_mano.ipynb   ← EL proceso completo, en orden ("Ejecutar todo")
gemelo/                       ← módulos que usa el notebook
  datos_emg.py                  D1: EMG sintético con dos vistas físicas por ventana
  datos_poses.py                D2: poses de mano de 21 puntos (topología MediaPipe)
  datos_externos.py             D4: EMG real público (UCI) en el formato del gemelo
  contrastivo.py                codificador 1D-CNN, SimCLR + SupCon, cabezas multitarea
  formas.py                     red de formas de la mano (autoencoder, latente 3D)
  estudiante.py                 red destilada para el ESP32
  refuerzo.py                   agente de servos (REINFORCE)
  metricas.py                   R², pseudo-R² de McFadden, completitud de grupos
  vivo.py                       panel en vivo y animación (Plotly)
  tablero_vivo.py               tablero del entrenamiento en el navegador (http://127.0.0.1:8765)
  mano_inmoov.py                arma la mano InMoov i2 desde sus STL con la cinemática del gemelo
  episodios.py                  "intentos" de la mano por época/iteración para el visor 3D
  web/                          Three.js r147 (MIT) para el visor, funciona sin internet
capturar_poses.py             ← D3: poses reales con la webcam (MediaPipe)
ia_mano.py                    ← núcleo común Python / ESP32 (características, red, exportación a C)
simulador.py                  ← física del gemelo (antebrazo, circuito, mano)
gemelo_digital.py             ← app de escritorio del gemelo (PySide6)
entrenar_modelo.py            ← entrenamiento con datos reales del brazo
protesis_tiempo_real.ino      ← firmware del ESP32
modelos/                      ← exportables: .h5, .keras, modelos/esp32/*.h, ficha_modelo.json
reportes/                     ← animación del contrastivo, espacio de formas, resumen de R²
```

## Datasets

| | Fuente | Tamaño | Uso |
|---|---|---|---|
| D1 | Gemelo digital (160 personas virtuales, 5 niveles de dificultad) | ~80.640 ventanas × 2 vistas | contrastivo, gesto, pose, fuerza |
| D2 | Cinemática de la mano (600 manos virtuales) | 60.000 poses | red de formas |
| D3 | Tu webcam + MediaPipe (`capturar_poses.py`) | lo que captures | poses reales |
| D4 | UCI *EMG data for gestures* (36 personas, CC BY 4.0) | ~71.000 ventanas | EMG real: no supervisado y transferencia |

Los datasets **no** van en el repositorio: se generan en `%USERPROFILE%\gemelo_mano\datos` (fuera de OneDrive). Se cambia con la variable `GEMELO_DATOS`.

Cita de D4: Lobov S., Krilova N., Kastalskiy I., Kazantsev V., Makarov V.A. *Latent Factors Limiting the Performance of sEMG-Interfaces.* Sensors 2018;18(4):1122.

## Instalación (Windows, GPU NVIDIA)

TensorFlow no usa la GPU en Windows nativo, así que Keras 3 corre sobre **PyTorch con CUDA**.

```bat
py -3.12 -m venv %USERPROFILE%\gemelo_mano\venv
%USERPROFILE%\gemelo_mano\venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
%USERPROFILE%\gemelo_mano\venv\Scripts\python -m pip install -r requirements.txt
%USERPROFILE%\gemelo_mano\venv\Scripts\python -m ipykernel install --user --name gemelo-mano --display-name "Gemelo Mano (Python 3.12 · GPU)" --env KERAS_BACKEND torch
```

Abre `notebooks/gemelo_mano.ipynb` en VS Code, elige el kernel **Gemelo Mano (Python 3.12 · GPU)** y ejecuta todo.

## Ver el entrenamiento en vivo

Al ejecutar el notebook se abre solo **http://127.0.0.1:8765/index.html** (servidor local, solo tu PC):

- **Mano InMoov en tiempo real:** en cada época/iteración la mano intenta los 7 gestos con lo que la red sabe en ese momento, a velocidad real, con la pose objetivo en azul transparente, la señal EMG sincronizada y si acierta (✓/✗). Salta siempre al intento más nuevo; el menú permite repetir cualquiera.
- La secuencia de las 13 secciones (terminada / en curso / pendiente), las figuras que se redibujan por época y la tabla de R² que va creciendo.

Al terminar queda `reportes/tablero.html`, que se abre con doble clic sin servidor.

La mano se arma desde `Inmoov-hand-i2.zip` (InMoov i2 de Gael Langevin, inmoov.fr, CC BY-NC), que debe estar en la raíz del proyecto; no se sube al repositorio.

**Requisitos al entrenar:** cargador conectado (en batería Windows puede apagar la GPU) y unos 5 GB de RAM libres (cierra pestañas y ventanas que no uses).

## Métricas

Regla del proyecto: **todo modelo reporta un R²**. En regresión (pose de 6 grados de libertad, fuerza, reconstrucción) es el R² clásico; en clasificación de gestos es el pseudo-R² de McFadden (1 − LL_modelo / LL_nulo), junto con precisión y F1. La tabla completa queda en `reportes/resumen_r2.csv` y en `modelos/ficha_modelo.json`.

## Al ESP32

Copia `modelos/esp32/modelo_red.h` y `modelos/esp32/poses_aprendidas.h` junto a `protesis_tiempo_real.ino` (`MODELO 2`). El notebook verifica que la copia en C prediga exactamente lo mismo que el modelo Keras.
