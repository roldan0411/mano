/*
 * firmware_pc.c — El procesamiento del ESP32 compilado en el PC (prueba D1)
 * -------------------------------------------------------------------------
 * Usa EXACTAMENTE los mismos archivos que el firmware: procesamiento_emg.h,
 * modelo_red_calibrado.h (campeona de la competencia) y red_neat.h (campeón NEAT).
 *
 *   firmware_pc <carpeta> <ventanas de calibración> <ventanas de la pista>
 *
 * Lee cal.bin y pista.bin (uint16 [n][200][10]) y vidx.bin (int8, índice de la
 * ventana dentro de su gesto); escribe cal_out.bin (40 floats), salida.bin
 * (por ventana: F[40], x[120], confianza, gesto de la red, puntajes NEAT[7]) y
 * tiempos.txt (µs por ventana en el PC).
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "modelo_red_calibrado.h"
#include "procesamiento_emg.h"
#include "red_neat.h"

#define POR_VENTANA (NUM_CARAC + NUM_ENTRADAS + 2 + NEAT_SALIDAS)

static uint16_t buf[NUM_CANALES][VENTANA];

static void cargar_ventana(const uint16_t *v) {
  for (int i = 0; i < VENTANA; i++)
    for (int c = 0; c < NUM_CANALES; c++) buf[c][i] = v[i * NUM_CANALES + c];
}

static void *leer(const char *dir, const char *nombre, size_t bytes) {
  char ruta[1024];
  snprintf(ruta, sizeof ruta, "%s/%s", dir, nombre);
  FILE *f = fopen(ruta, "rb");
  if (!f) { fprintf(stderr, "no pude abrir %s\n", ruta); exit(1); }
  void *p = malloc(bytes);
  if (fread(p, 1, bytes, f) != bytes) { fprintf(stderr, "faltan datos en %s\n", ruta); exit(1); }
  fclose(f);
  return p;
}

static void escribir(const char *dir, const char *nombre, const void *p, size_t bytes) {
  char ruta[1024];
  snprintf(ruta, sizeof ruta, "%s/%s", dir, nombre);
  FILE *f = fopen(ruta, "wb");
  fwrite(p, 1, bytes, f);
  fclose(f);
}

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  const char *dir = argv[1];
  int ncal = atoi(argv[2]), T = atoi(argv[3]);
  size_t tam = (size_t)VENTANA * NUM_CANALES * sizeof(uint16_t);
  uint16_t *cal = leer(dir, "cal.bin", tam * ncal);
  uint16_t *pis = leer(dir, "pista.bin", tam * T);
  int8_t *vidx = leer(dir, "vidx.bin", T);

  static EstadoEMG e;
  float F[NUM_CARAC], x[NUM_ENTRADAS], z[NEAT_SALIDAS];
  emg_cal_empezar(&e);
  for (int k = 0; k < ncal; k++) {
    cargar_ventana(cal + (size_t)k * VENTANA * NUM_CANALES);
    emg_caracteristicas(buf, 0, F);
    emg_cal_agregar(&e, F);
  }
  emg_cal_terminar(&e);
  escribir(dir, "cal_out.bin", e.cal, sizeof(e.cal));

  float *out = malloc(sizeof(float) * POR_VENTANA * T);
  for (int t = 0; t < T; t++) {
    if (vidx[t] == 0) emg_reiniciar_contexto(&e);   /* cada gesto de la pista empieza solo, como en Python */
    cargar_ventana(pis + (size_t)t * VENTANA * NUM_CANALES);
    emg_caracteristicas(buf, 0, F);
    emg_entrada(&e, F, x);
    float conf = 0;
    int g = predecir_gesto(x, &conf);
    neat_predecir(x, z);
    float *o = out + (size_t)t * POR_VENTANA;
    for (int k = 0; k < NUM_CARAC; k++) o[k] = F[k];
    for (int k = 0; k < NUM_ENTRADAS; k++) o[NUM_CARAC + k] = x[k];
    o[NUM_CARAC + NUM_ENTRADAS] = conf;
    o[NUM_CARAC + NUM_ENTRADAS + 1] = (float)g;
    for (int k = 0; k < NEAT_SALIDAS; k++) o[NUM_CARAC + NUM_ENTRADAS + 2 + k] = z[k];
  }
  escribir(dir, "salida.bin", out, sizeof(float) * POR_VENTANA * T);

  /* tiempos en el PC (repitiendo para que el reloj alcance a medir) */
  const int REP = 200;
  volatile float sumidero = 0;
  clock_t c0 = clock();
  for (int r = 0; r < REP; r++)
    for (int t = 0; t < T; t++) { cargar_ventana(pis + (size_t)t * VENTANA * NUM_CANALES); emg_caracteristicas(buf, 0, F); sumidero += F[0]; }
  double us_car = (double)(clock() - c0) / CLOCKS_PER_SEC * 1e6 / (REP * T);
  c0 = clock();
  for (int r = 0; r < REP * 5; r++)
    for (int t = 0; t < T; t++) { float cf; sumidero += predecir_gesto(out + (size_t)t * POR_VENTANA + NUM_CARAC, &cf); }
  double us_red = (double)(clock() - c0) / CLOCKS_PER_SEC * 1e6 / (REP * 5 * T);
  c0 = clock();
  for (int r = 0; r < REP * 5; r++)
    for (int t = 0; t < T; t++) { neat_predecir(out + (size_t)t * POR_VENTANA + NUM_CARAC, z); sumidero += z[0]; }
  double us_neat = (double)(clock() - c0) / CLOCKS_PER_SEC * 1e6 / (REP * 5 * T);
  char ruta[1024];
  snprintf(ruta, sizeof ruta, "%s/tiempos.txt", dir);
  FILE *f = fopen(ruta, "w");
  fprintf(f, "%.4f %.4f %.4f\n", us_car, us_red, us_neat);
  fclose(f);
  return 0;
}
