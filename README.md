# Prótesis mioeléctrica con reconocimiento de gestos por IA — gemelo digital

Roldán Rodríguez — Ing. CCIA

Mano impresa con 5 servos de dedos y un servo de muñeca, controlada por un ESP32 que lee **2 canales EMG** del antebrazo (AD620 + pasa-banda, ADC de 12 bits a 1 kHz) y reconoce 7 gestos: reposo, abierta, puño, pinza, señalar, flexión y extensión de muñeca.

El **gemelo digital** simula todo el sistema (10 músculos, electrodos, circuito, ADC, servos, tendones) para que la IA aprenda sin persona ni hardware; lo aprendido se pasa al ESP32.

## Aplicación de escritorio: el gemelo que entrena sin parar

Doble clic en **`Gemelo de la mano.bat`** (o `python app_gemelo.py` con el entorno del proyecto). Es una ventana nativa de Windows (Qt + OpenGL); el entrenamiento corre en la GPU en un proceso aparte.

Mientras la ventana está abierta:

1. Procesos de simulación generan **personas, sesiones y calibraciones** sin parar (electrodos que se mueven en cada sesión, desfase continuo, AD620 que puede saturarse) y **guardan cada lote en disco**.
2. **10 manos** (redes del tamaño del ESP32, cada una con su genoma) entrenan por turnos con todo lo acumulado.
3. Rinden el mismo **examen** (personas que ninguna vio). Las 3 peores copian a una de las 3 mejores y mutan; cada 12 generaciones entra una mano nueva al azar.
4. Si la mejor supera a la campeona, se guarda como **campeona**: modelo `.keras` y `.h5`, `modelo_red_calibrado.h` para el ESP32, ficha y **fotos de sus 7 gestos** con la mano InMoov.

Pestañas: **Competencia de manos** (las 10 manos InMoov en 3D moviéndose en tiempo real), **Red neuronal** (pesos y activaciones en vivo), **Gemelo y electrodos** (corte del antebrazo, electrodos desplazados, señal en cada etapa del amplificador, calibración), **Campeonas** (galería de fotos) y **Datos guardados**.

Todo se guarda en `%USERPROFILE%\gemelo_mano\motor\` (base SQLite, lotes, examen, población y campeonas). La población se guarda en cada generación; al reabrir, cada mano retoma su última red guardada (si alguna no se puede leer, arranca con la campeona de mayor puntaje) y la ventana muestra de inmediato el ranking, el historial y las campeonas. Si Windows apagó la GPU (en batería), entrena en la CPU, más lento; si se desconecta el cargador en pleno entrenamiento con GPU, o queda poca RAM, se pausa y sigue mostrando todo.

La entrada de estas redes son las 8 características **calibradas** (relativas a 20 s de calibración al ponerse los electrodos) de las 3 últimas ventanas: 24 valores. El firmware actual usa 8 sin calibrar, así que `modelo_red_calibrado.h` necesita la actualización del firmware con calibración.

## Neuroevolución: 150 manos aprenden desde cero a seguir a una persona

Acceso directo **"Neuroevolución de la mano"** en el escritorio, o el botón "Neuroevolución" de la aplicación (`app_gemelo.py --neuro` la abre sola; corre en CPU).

Como los carros que aprenden a manejar con algoritmos genéticos, pero la pista es **una persona del gemelo** que se pone los electrodos, se calibra y hace 21 gestos al azar (~29 s). Las manos solo escuchan los 2 canales EMG (las mismas 24 entradas calibradas del ESP32) y tienen que hacer lo que la persona quiere. Si hacen el gesto equivocado medio segundo seguido, **chocan** y se apagan.

- **NEAT** (Stanley y Miikkulainen, 2002): nadie las entrena con gradiente. Nacen casi vacías (cada gesto conectado a 4 entradas al azar) y la evolución cambia pesos, agrega conexiones y **neuronas nuevas**: la red crece. Se agrupan en **especies** por parecido; las especies que no mejoran en 20 generaciones se extinguen.
- **Fitness**: segundos sin chocar + 30 × acierto, en 6 personas distintas por generación (las pistas se renuevan: no pueden aprenderlas de memoria).
- **Examen** de cada generación: 8 personas que ninguna mano vio. Siempre se ven el acierto, el **pseudo-R²** de McFadden y el **R² de la pose** de los dedos.
- La ventana muestra 30 de las 150 manos (de la mejor a la peor, color = especie), la pista con la señal EMG y dónde chocó cada una, la red de la mejor con sus neuronas encendiéndose, y las curvas de aguante, acierto, R² y especies. Se puede ver cada generación completa (×1, ×2, ×4) o dejarla evolucionar rápido.
- Cada **campeón** (mejor acierto en el examen) se guarda con su red como código C (`red_neat.h`, función `neat_predecir`) para el ESP32. Todo queda en `%USERPROFILE%\gemelo_mano\motor\neuro\` y al reabrir sigue; "Volver a empezar" guarda la población como respaldo y nace de cero.

En una prueba de 300 generaciones (unos 60 s sin ventana) pasó de 35 % a 89 % de acierto en personas nuevas, de aguantar 1 s a 24 s de los 29 de la pista, y R² de la pose 0,785. Como referencia, un clasificador lineal entrenado con gradiente llega a 96 % con los mismos datos: la evolución todavía tiene margen.

## Mano bebé: aprende sola, desde cero, qué puede hacer su cuerpo

Acceso directo **"Mano bebé (entrenamiento libre)"** en el escritorio, o el botón "Entrenamiento libre" de la aplicación (`app_gemelo.py --libre` la abre sola, sin GPU).

Nadie le da gestos, listas ni metas. La mano controla sus **10 músculos** y una biomecánica que no conoce los traduce en dedos y muñeca, con los acoples reales (los tendones flexores mueven varios dedos a la vez):

1. **Balbuceo** (primeros 300 intentos): activa músculos al azar y aprende cómo responde su cuerpo.
2. **Explora libre**: se pone sus propias metas y decide sola cómo aprender: *inventar* una pose cualquiera, *ir más allá* de algo que ya sabe, *practicar* algo nuevo que le salió, o *balbucear*. Prefiere la estrategia que más le está enseñando (curiosidad: novedad + progreso).
3. Una **habilidad** es una pose que descubrió ella misma y ya sabe repetir a voluntad (3 veces seguidas), con foto y descripción dedo por dedo. El repertorio **no tiene tope**: cuando ya no encuentra nada nuevo, sube de **finura** (distingue detalles más finos) y sigue descubriendo.

**Internet es un espejo, no una tarea**: de cada cosa que hace mira si alguna persona real hace algo parecido (HaGRID v2, 19.800 manos) y lo anota en la habilidad.

Su cerebro: un modelo inverso (pose → músculos) y uno directo (músculos → pose) que aprenden en línea. Siempre se ve el **R² del control** (pose lograda frente a la que se propuso, en 300 poses de examen) y el **R² de su cuerpo** (modelo directo). Corre en CPU, guarda cada 20 s y al cerrar en `%USERPROFILE%\gemelo_mano\motor\bebe\`, y al reabrir continúa. "Volver a nacer" la reinicia de cero.

En una prueba desde cero descubrió su primera habilidad en el intento 318, tenía 41 a los 3.000 intentos y 132 a los 15.000 (ya en finura 2), con R² del control de 0,98.

Datos del espejo: [HaGRID v2](https://github.com/hukenovs/hagrid) — Kapitanov et al., *HaGRID – HAnd Gesture Recognition Image Dataset*, WACV 2024, licencia CC BY-SA 4.0. Solo se usan los 21 puntos de cada mano (no las fotos), descargados parcialmente (600 manos por gesto).

## Estructura

```
notebooks/gemelo_mano.ipynb   ← EL proceso completo, en orden ("Ejecutar todo")
app_gemelo.py                 ← APLICACIÓN DE ESCRITORIO (doble clic en "Gemelo de la mano.bat")
gemelo/motor/                 ← motor continuo: gemelo v2, almacén, población de 10 manos, fotos
gemelo/escritorio/            ← ventanas Qt: manos 3D (OpenGL), red, antebrazo, osciloscopios, mano bebé, neuroevolución
gemelo/neuro/                 ← neuroevolución: NEAT, la pista (persona del gemelo), exportación a C
gemelo/libre/                 ← mano bebé: cuerpo (10 músculos), cerebro que explora solo, espejo de internet (HaGRID)
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
