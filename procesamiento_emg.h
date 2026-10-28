/*
 * procesamiento_emg.h — Procesamiento de los 10 electrodos (C puro, sin Arduino)
 * ---------------------------------------------------------------------------
 * Roldán Rodríguez — Ing. CCIA
 *
 * Es la misma cuenta que hace el gemelo en Python (ia_mano.caracteristicas_lote,
 * gemelo2.calibracion / calibrar y datos_emg.indices_contexto), para que la red
 * entrenada en el gemelo reciba en el ESP32 exactamente lo mismo:
 *
 *   1. Por electrodo y ventana de 200 ms: MAV, RMS, ZC y WL     -> 40 valores
 *   2. Calibración de la sesión (al ponerse los electrodos):
 *        cal = promedio de log(1 + F) mientras la persona hace cada gesto una vez
 *   3. Calibradas: Fc = log(1 + F) - cal
 *   4. Contexto: [Fc(t-2), Fc(t-1), Fc(t)]                      -> 120 entradas
 *
 * No depende de Arduino: se compila también en el PC para probar que da lo
 * mismo que Python (pruebas/prueba_firmware.py).
 */
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#define NUM_CANALES 10                 /* F1-F5 flexores, E1-E5 extensores */
#define CARAC_POR_CANAL 4              /* MAV, RMS, ZC, WL */
#define NUM_CARAC (NUM_CANALES * CARAC_POR_CANAL)
#define VENTANAS_CONTEXTO 3
#define NUM_ENTRADAS (NUM_CARAC * VENTANAS_CONTEXTO)
#ifndef VENTANA
  #define VENTANA 200                  /* muestras (200 ms a 1 kHz) */
#endif
#ifndef PASO
  #define PASO 100
#endif
#ifndef UMBRAL_ZC
  #define UMBRAL_ZC 20                 /* cuentas del ADC */
#endif

/* buf[c][i]: buffer circular de cada electrodo; inicio = índice de la muestra más antigua */
static void emg_caracteristicas(const uint16_t buf[NUM_CANALES][VENTANA], int inicio, float *F) {
  float x[VENTANA];
  for (int c = 0; c < NUM_CANALES; c++) {
    float media = 0;
    for (int i = 0; i < VENTANA; i++) {
      x[i] = (float)buf[c][(inicio + i) % VENTANA];
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
    F[c * CARAC_POR_CANAL + 0] = mav / VENTANA;
    F[c * CARAC_POR_CANAL + 1] = sqrtf(rms / VENTANA);
    F[c * CARAC_POR_CANAL + 2] = zc;
    F[c * CARAC_POR_CANAL + 3] = wl;
  }
}

typedef struct {
  float cal[NUM_CARAC];                       /* calibración de la sesión */
  double suma[NUM_CARAC];
  int n_cal;
  float hist[VENTANAS_CONTEXTO][NUM_CARAC];   /* últimas ventanas calibradas (la última es la actual) */
  int n_hist;
} EstadoEMG;

static void emg_cal_empezar(EstadoEMG *e) {
  memset(e->suma, 0, sizeof(e->suma));
  e->n_cal = 0;
}

static void emg_cal_agregar(EstadoEMG *e, const float *F) {
  for (int k = 0; k < NUM_CARAC; k++) e->suma[k] += log1p((double)F[k]);
  e->n_cal++;
}

static int emg_cal_terminar(EstadoEMG *e) {
  if (e->n_cal == 0) return 0;
  for (int k = 0; k < NUM_CARAC; k++) e->cal[k] = (float)(e->suma[k] / e->n_cal);
  e->n_hist = 0;
  return 1;
}

static void emg_reiniciar_contexto(EstadoEMG *e) { e->n_hist = 0; }

/* De las 40 características de una ventana a las 120 entradas de la red.
   Al empezar (sin ventanas anteriores) repite la primera, igual que en Python. */
static void emg_entrada(EstadoEMG *e, const float *F, float *x) {
  float fc[NUM_CARAC];
  for (int k = 0; k < NUM_CARAC; k++) fc[k] = log1pf(F[k]) - e->cal[k];
  if (e->n_hist == 0)
    for (int j = 0; j < VENTANAS_CONTEXTO; j++) memcpy(e->hist[j], fc, sizeof(fc));
  else {
    memmove(e->hist[0], e->hist[1], sizeof(float) * NUM_CARAC * (VENTANAS_CONTEXTO - 1));
    memcpy(e->hist[VENTANAS_CONTEXTO - 1], fc, sizeof(fc));
  }
  if (e->n_hist < VENTANAS_CONTEXTO) e->n_hist++;
  for (int j = 0; j < VENTANAS_CONTEXTO; j++) memcpy(&x[j * NUM_CARAC], e->hist[j], sizeof(fc));
}
