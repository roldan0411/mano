/*
 * ============================================================
 *  PRÓTESIS MIOELÉCTRICA CON RECONOCIMIENTO DE GESTOS POR IA
 *  Programa 3 de 3 — Clasificación en tiempo real (ESP32)
 * ------------------------------------------------------------
 *  Roldán Rodríguez — Ing. CCIA
 * ============================================================
 *
 *  Qué hace:
 *    1. Muestrea los dos canales EMG a 1 kHz.
 *    2. Cada 100 ms toma los últimos 200 ms de señal y calcula
 *       MAV, RMS, ZC y WL de cada canal (8 características).
 *    3. La red neuronal (modelo_red.h) predice el gesto y su
 *       confianza. Si la confianza es baja, no cambia de gesto.
 *    4. Un filtro de mayoría (3 predicciones) evita temblores.
 *    5. Los 5 servos de los dedos van a la pose del gesto y el
 *       servo de la muñeca (MG996R) se mueve suavemente hacia su
 *       ángulo, para no pedir picos grandes de corriente.
 *
 *  Gestos: reposo, abierta, puno, pinza, senalar (dedos) y
 *          flexion, extension (muñeca).
 *
 *  Antes de subirlo:
 *    - Copia modelo_red.h y modelo_rf.h (los genera
 *      entrenar_modelo.py) en esta misma carpeta.
 *    - Opcional: copia poses_aprendidas.h (la genera el gemelo
 *      digital con aprendizaje por refuerzo). Si está, se usan
 *      esos comandos de servo en vez de la tabla de abajo.
 *    - Instala la librería "ESP32Servo" (Gestor de librerías).
 *
 *  MODELO:
 *    2 = red neuronal (principal)
 *    1 = Random Forest (respaldo)
 *    0 = PLAN B por umbral, no necesita ningún modelo. La mano se
 *        cierra cuando el canal 1 supera UMBRAL_PUNO. Mira los
 *        valores de MAV en el Monitor Serie para calibrarlo.
 */

#define MODELO 2

#include <ESP32Servo.h>
#include <math.h>
#include <string.h>

#if MODELO == 2
  #include "modelo_red.h"
#elif MODELO == 1
  #include "modelo_rf.h"
#else
  #define VENTANA 200
  #define PASO 100
  #define UMBRAL_ZC 20
  #define NUM_CARACTERISTICAS 8
  const float UMBRAL_PUNO = 60.0;   // ajústalo con tu propia señal
#endif

// Si el modelo está menos seguro que esto, la mano mantiene el gesto anterior
const float UMBRAL_CONFIANZA = 0.60;

// ---------------- Pines ----------------
const int PIN_CANAL_1 = 34;    // flexores (antebrazo interno)
const int PIN_CANAL_2 = 35;    // extensores (antebrazo externo)
//                      pulgar indice medio anular menique
const int PIN_SERVO[5] = {13,   14,    25,   26,    27};
const int PIN_MUNECA = 33;     // servo MG996R de la muñeca

// Calibración de cada dedo (grados). Si un dedo se mueve al revés,
// intercambia sus valores de abierto y cerrado.
const int ANG_ABIERTO[5] = {0,   0,   0,   0,   0};
const int ANG_CERRADO[5] = {160, 170, 170, 170, 170};

// Calibración de la muñeca (grados). Si se dobla al revés, cambia el
// signo de MUNECA_RANGO. No pases de ±45° para no forzar la bisagra.
const int MUNECA_NEUTRO = 90;
const int MUNECA_RANGO = 40;          // grados hacia cada lado
const int MUNECA_PASO = 2;            // grados por paso (suavizado)
const int MUNECA_CADA_MS = 20;        // un paso cada 20 ms = 100 °/s

// Poses: dedos 0.0 = estirado, 1.0 = cerrado
//        muñeca -1 = flexión (hacia la palma), 0 = recta, +1 = extensión
struct Pose { const char* gesto; float dedos[5]; float muneca; };
#if __has_include("poses_aprendidas.h")
  // Comandos aprendidos por refuerzo en el gemelo digital (gemelo_digital.py)
  #include "poses_aprendidas.h"
#else
const Pose POSES[] = {
  {"reposo",    {0.2, 0.2, 0.2, 0.2, 0.2},  0.0},
  {"abierta",   {0.0, 0.0, 0.0, 0.0, 0.0},  0.0},
  {"puno",      {1.0, 1.0, 1.0, 1.0, 1.0},  0.0},
  {"pinza",     {0.8, 0.8, 0.0, 0.0, 0.0},  0.0},
  {"senalar",   {1.0, 0.0, 1.0, 1.0, 1.0},  0.0},
  {"flexion",   {0.2, 0.2, 0.2, 0.2, 0.2}, -1.0},
  {"extension", {0.2, 0.2, 0.2, 0.2, 0.2},  1.0},
};
#endif
const int NUM_POSES = sizeof(POSES) / sizeof(POSES[0]);

// ---------------- Variables ----------------
const unsigned long PERIODO_US = 1000;   // 1 kHz
Servo servos[5];
Servo servoMuneca;
int munecaActual = MUNECA_NEUTRO, munecaObjetivo = MUNECA_NEUTRO;
int contadorMuneca = 0;
int16_t buf1[VENTANA], buf2[VENTANA];    // buffers circulares
int idx = 0, muestrasNuevas = 0, muestrasTotales = 0;
unsigned long proximaMuestra;
int historial[3] = {-1, -1, -1};
int poseActual = -1;
int ultimaClase = 0;

// ---------------- Características (igual que en Python) ----------------
void caracteristicasCanal(const int16_t* b, int inicio, float* salida) {
  float x[VENTANA];
  float media = 0;
  for (int i = 0; i < VENTANA; i++) {
    x[i] = b[(inicio + i) % VENTANA];
    media += x[i];
  }
  media /= VENTANA;

  float mav = 0, rms = 0, zc = 0, wl = 0;
  for (int i = 0; i < VENTANA; i++) {
    x[i] -= media;
    mav += fabsf(x[i]);
    rms += x[i] * x[i];
  }
  for (int i = 0; i < VENTANA - 1; i++) {
    float d = x[i + 1] - x[i];
    if (x[i] * x[i + 1] < 0 && fabsf(d) >= UMBRAL_ZC) zc += 1;
    wl += fabsf(d);
  }
  salida[0] = mav / VENTANA;
  salida[1] = sqrtf(rms / VENTANA);
  salida[2] = zc;
  salida[3] = wl;
}

int buscarPose(const char* nombre) {
  for (int i = 0; i < NUM_POSES; i++)
    if (strcmp(POSES[i].gesto, nombre) == 0) return i;
  return 0;  // si no la encuentra, reposo
}

void moverMano(int p) {
  if (p == poseActual) return;
  poseActual = p;
  for (int d = 0; d < 5; d++) {
    float f = POSES[p].dedos[d];
    servos[d].write(ANG_ABIERTO[d] + f * (ANG_CERRADO[d] - ANG_ABIERTO[d]));
  }
  munecaObjetivo = MUNECA_NEUTRO + (int)(POSES[p].muneca * MUNECA_RANGO);
}

// Se llama cada 1 ms; mueve la muñeca de a pocos grados hacia su objetivo
void actualizarMuneca() {
  if (++contadorMuneca < MUNECA_CADA_MS) return;
  contadorMuneca = 0;
  if (munecaActual == munecaObjetivo) return;
  int d = munecaObjetivo - munecaActual;
  if (d > MUNECA_PASO) d = MUNECA_PASO;
  if (d < -MUNECA_PASO) d = -MUNECA_PASO;
  munecaActual += d;
  servoMuneca.write(munecaActual);
}

int mayoria(int nueva) {
  historial[0] = historial[1];
  historial[1] = historial[2];
  historial[2] = nueva;
  // Si dos de las tres últimas predicciones coinciden, gana esa
  if (historial[0] >= 0 && (historial[0] == historial[1] || historial[0] == historial[2]))
    return historial[0];
  if (historial[1] >= 0 && historial[1] == historial[2]) return historial[1];
  return nueva;
}

// ---------------- Programa principal ----------------
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  for (int d = 0; d < 5; d++) {
    servos[d].setPeriodHertz(50);
    servos[d].attach(PIN_SERVO[d], 500, 2400);
  }
  servoMuneca.setPeriodHertz(50);
  servoMuneca.attach(PIN_MUNECA, 500, 2400);
  servoMuneca.write(MUNECA_NEUTRO);
  moverMano(buscarPose("abierta"));
  delay(800);

  Serial.println(MODELO == 2 ? "Modelo: red neuronal"
               : MODELO == 1 ? "Modelo: Random Forest"
               : "Modelo: umbral (plan B)");
  proximaMuestra = micros();
}

void loop() {
  if ((long)(micros() - proximaMuestra) < 0) return;
  proximaMuestra += PERIODO_US;
  actualizarMuneca();

  buf1[idx] = analogRead(PIN_CANAL_1);
  buf2[idx] = analogRead(PIN_CANAL_2);
  idx = (idx + 1) % VENTANA;
  if (muestrasTotales < VENTANA) muestrasTotales++;
  muestrasNuevas++;

  if (muestrasTotales < VENTANA || muestrasNuevas < PASO) return;
  muestrasNuevas = 0;

  // idx apunta a la muestra más antigua de la ventana
  float carac[NUM_CARACTERISTICAS];
  caracteristicasCanal(buf1, idx, &carac[0]);
  caracteristicasCanal(buf2, idx, &carac[4]);

#if MODELO == 0
  const char* nombre = (carac[0] > UMBRAL_PUNO) ? "puno" : "abierta";
  int pose = buscarPose(nombre);
  Serial.printf("MAV c1=%.1f  c2=%.1f  -> %s\n", carac[0], carac[4], nombre);
#else
  float confianza = 0;
  int prediccion = predecir_gesto(carac, &confianza);
  if (confianza >= UMBRAL_CONFIANZA) ultimaClase = prediccion;  // si duda, no cambia
  int clase = mayoria(ultimaClase);
  const char* nombre = NOMBRES_GESTOS[clase];
  int pose = buscarPose(nombre);
  Serial.printf("Gesto: %-9s | confianza %3.0f %% | MAV c1=%.1f c2=%.1f\n",
                nombre, confianza * 100, carac[0], carac[4]);
#endif

  moverMano(pose);
}
