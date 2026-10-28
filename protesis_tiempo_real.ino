/*
 * ============================================================
 *  PRÓTESIS MIOELÉCTRICA CON RECONOCIMIENTO DE GESTOS POR IA
 *  Programa 3 de 3 — Clasificación en tiempo real (ESP32)
 * ------------------------------------------------------------
 *  Roldán Rodríguez — Ing. CCIA
 * ============================================================
 *
 *  Qué hace:
 *    1. Muestrea los 10 electrodos a 1 kHz: F1-F5 sobre los flexores
 *       y E1-E5 sobre los extensores, cada uno contra un electrodo de
 *       referencia común (en el codo).
 *    2. Al encender, CALIBRA (~28 s): la mano muestra cada gesto y la
 *       persona lo hace una vez. Así las características quedan
 *       relativas a la persona y a dónde quedaron hoy los electrodos.
 *       Mandando "c" por el Monitor Serie se vuelve a calibrar.
 *    3. Cada 100 ms toma los últimos 200 ms de cada electrodo y calcula
 *       MAV, RMS, ZC y WL (40 características), las calibra y junta las
 *       3 últimas ventanas: 120 entradas (procesamiento_emg.h, la misma
 *       cuenta que el gemelo digital en Python).
 *    4. La red predice el gesto y su confianza. Si la confianza es baja,
 *       no cambia de gesto; un filtro de mayoría (3 predicciones) evita
 *       temblores.
 *    5. Los 5 servos de los dedos van a la pose del gesto y el servo de
 *       la muñeca (MG996R) se mueve suavemente hacia su ángulo.
 *
 *  Lectura de los 10 electrodos: el ADC1 del ESP32 no tiene 10 entradas
 *  libres, así que cada electrodo (ya amplificado y filtrado) entra a un
 *  multiplexor analógico CD74HC4067 y el ESP32 lee la salida del
 *  multiplexor en un solo pin, canal por canal, en cada milisegundo. Si
 *  se usa otro conversor (por ejemplo un ADS1299), solo cambia
 *  leerElectrodos().
 *
 *  MODELO:
 *    2 = red de la competencia del gemelo (modelo_red_calibrado.h, de la
 *        carpeta de la campeona)                                  [principal]
 *    3 = red evolucionada por neuroevolución (red_neat.h, de la carpeta
 *        del campeón NEAT)
 *    1 = Random Forest entrenado con datos reales sin calibrar (modelo_rf.h,
 *        lo genera entrenar_modelo.py)
 *    0 = PLAN B por umbral, sin modelo: la mano se cierra cuando los
 *        flexores (F3) superan UMBRAL_PUNO. Mira los MAV en el Monitor Serie.
 *
 *  Antes de subirlo: copia el .h del modelo elegido junto a este archivo
 *  (y procesamiento_emg.h) e instala la librería "ESP32Servo".
 */

#define MODELO 2

#include <ESP32Servo.h>
#include <math.h>
#include <string.h>

#if MODELO == 2
  #include "modelo_red_calibrado.h"
#elif MODELO == 1
  #include "modelo_rf.h"
#endif
#include "procesamiento_emg.h"
#if MODELO == 3
  #include "red_neat.h"
#endif
#if MODELO == 0 || MODELO == 3
  #define NUM_CLASES 7
  static const char* const NOMBRES_GESTOS[NUM_CLASES] = {
    "reposo", "abierta", "puno", "pinza", "senalar", "flexion", "extension"};
#endif
#if MODELO == 0
  const float UMBRAL_PUNO = 60.0;   // ajústalo con tu propia señal
#endif

// Si el modelo está menos seguro que esto, la mano mantiene el gesto anterior
const float UMBRAL_CONFIANZA = 0.60;

// ---------------- Pines ----------------
const int PIN_MUX_SALIDA = 34;                 // salida del CD74HC4067 -> ADC1
const int PIN_MUX_SEL[4] = {18, 19, 21, 22};   // S0..S3 del multiplexor
// canal del multiplexor de cada electrodo:  F1 F2 F3 F4 F5 E1 E2 E3 E4 E5
const int CANAL_MUX[NUM_CANALES] =         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

// Calibración de la sesión: cada gesto, 1,5 s para prepararse y 2,5 s haciéndolo
const unsigned long CAL_PREPARAR_MS = 1500, CAL_GESTO_MS = 2500;

// Poses: dedos 0.0 = estirado, 1.0 = cerrado
//        muñeca -1 = flexión (hacia la palma), 0 = recta, +1 = extensión
struct Pose { const char* gesto; float dedos[5]; float muneca; };
#if __has_include("poses_aprendidas.h")
  // Comandos aprendidos por refuerzo en el gemelo digital
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
const unsigned long PERIODO_US = 1000;   // 1 kHz por electrodo
Servo servos[5];
Servo servoMuneca;
int munecaActual = MUNECA_NEUTRO, munecaObjetivo = MUNECA_NEUTRO;
int contadorMuneca = 0;
uint16_t buf[NUM_CANALES][VENTANA];      // buffers circulares, uno por electrodo
int idx = 0, muestrasNuevas = 0, muestrasTotales = 0;
unsigned long proximaMuestra;
int historial[3] = {-1, -1, -1};
int poseActual = -1;
int ultimaClase = 0;
EstadoEMG emg;
bool calibrando = false;
int calGesto = 0;
unsigned long calInicio = 0;

// ---------------- Lectura de los 10 electrodos ----------------
void leerElectrodos(uint16_t* v) {
  for (int c = 0; c < NUM_CANALES; c++) {
    int m = CANAL_MUX[c];
    for (int b = 0; b < 4; b++) digitalWrite(PIN_MUX_SEL[b], (m >> b) & 1);
    delayMicroseconds(3);                 // asentamiento del multiplexor
    v[c] = analogRead(PIN_MUX_SALIDA);
  }
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

// ---------------- Calibración de la sesión ----------------
void empezarCalibracion() {
  emg_cal_empezar(&emg);
  calibrando = true;
  calGesto = 0;
  calInicio = millis();
  moverMano(buscarPose(NOMBRES_GESTOS[0]));
  Serial.printf("Calibración: prepárate para hacer «%s»\n", NOMBRES_GESTOS[0]);
}

// true mientras la persona está haciendo el gesto (ahí se toman las ventanas)
bool avanzarCalibracion() {
  unsigned long t = millis() - calInicio;
  if (t < CAL_PREPARAR_MS) return false;
  if (t < CAL_PREPARAR_MS + CAL_GESTO_MS) return true;
  calGesto++;
  if (calGesto >= NUM_CLASES) {
    calibrando = false;
    emg_cal_terminar(&emg);
    Serial.printf("Calibración lista (%d ventanas). Ya puedes usar la mano.\n", emg.n_cal);
    return false;
  }
  calInicio = millis();
  moverMano(buscarPose(NOMBRES_GESTOS[calGesto]));
  Serial.printf("Calibración: prepárate para hacer «%s»\n", NOMBRES_GESTOS[calGesto]);
  return false;
}

// ---------------- Programa principal ----------------
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  for (int b = 0; b < 4; b++) pinMode(PIN_MUX_SEL[b], OUTPUT);

  for (int d = 0; d < 5; d++) {
    servos[d].setPeriodHertz(50);
    servos[d].attach(PIN_SERVO[d], 500, 2400);
  }
  servoMuneca.setPeriodHertz(50);
  servoMuneca.attach(PIN_MUNECA, 500, 2400);
  servoMuneca.write(MUNECA_NEUTRO);
  moverMano(buscarPose("abierta"));
  delay(800);

  Serial.println(MODELO == 2 ? "Modelo: red de la competencia del gemelo (120 entradas)"
               : MODELO == 3 ? "Modelo: red evolucionada (NEAT, 120 entradas)"
               : MODELO == 1 ? "Modelo: Random Forest (datos reales)"
               : "Modelo: umbral (plan B)");
#if MODELO >= 2
  empezarCalibracion();
#endif
  proximaMuestra = micros();
}

void loop() {
  if (Serial.available() && Serial.read() == 'c') empezarCalibracion();
  if ((long)(micros() - proximaMuestra) < 0) return;
  proximaMuestra += PERIODO_US;
  actualizarMuneca();

  uint16_t v[NUM_CANALES];
  leerElectrodos(v);
  for (int c = 0; c < NUM_CANALES; c++) buf[c][idx] = v[c];
  idx = (idx + 1) % VENTANA;
  if (muestrasTotales < VENTANA) muestrasTotales++;
  muestrasNuevas++;

  if (muestrasTotales < VENTANA || muestrasNuevas < PASO) return;
  muestrasNuevas = 0;

  // idx apunta a la muestra más antigua de la ventana
  float F[NUM_CARAC];
  emg_caracteristicas(buf, idx, F);

  if (calibrando) {
    if (avanzarCalibracion()) emg_cal_agregar(&emg, F);
    return;
  }

#if MODELO == 0
  const char* nombre = (F[2 * CARAC_POR_CANAL] > UMBRAL_PUNO) ? "puno" : "abierta";   // MAV de F3
  int pose = buscarPose(nombre);
  Serial.printf("MAV F3=%.1f  E3=%.1f  -> %s\n", F[2 * CARAC_POR_CANAL], F[7 * CARAC_POR_CANAL], nombre);
#else
  float confianza = 0;
  #if MODELO == 1
    int prediccion = predecir_gesto(F, &confianza);           // sin calibrar, como se entrenó
  #else
    float x[NUM_ENTRADAS];
    emg_entrada(&emg, F, x);                                   // calibradas + 3 ventanas
    #if MODELO == 2
      int prediccion = predecir_gesto(x, &confianza);
    #else
      float z[NUM_CLASES];
      neat_predecir(x, z);
      int prediccion = 0;
      for (int k = 1; k < NUM_CLASES; k++) if (z[k] > z[prediccion]) prediccion = k;
      float suma = 0;
      for (int k = 0; k < NUM_CLASES; k++) suma += expf(z[k] - z[prediccion]);
      confianza = 1.0f / suma;
    #endif
  #endif
  if (confianza >= UMBRAL_CONFIANZA) ultimaClase = prediccion;  // si duda, no cambia
  int clase = mayoria(ultimaClase);
  const char* nombre = NOMBRES_GESTOS[clase];
  int pose = buscarPose(nombre);
  Serial.printf("Gesto: %-9s | confianza %3.0f %% | MAV F3=%.1f E3=%.1f\n",
                nombre, confianza * 100, F[2 * CARAC_POR_CANAL], F[7 * CARAC_POR_CANAL]);
#endif

  moverMano(pose);
}
