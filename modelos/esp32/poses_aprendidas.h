/*
 * poses_aprendidas.h — Comandos de servos por gesto
 * Origen: aprendizaje por refuerzo en el gemelo digital.
 * Si este archivo está junto a protesis_tiempo_real.ino, el ESP32 lo usa
 * en lugar de la tabla de poses escrita a mano.
 * Roldán Rodríguez — Ing. CCIA
 */
#pragma once
#define POSES_APRENDIDAS 1

// dedos: 0 = estirado, 1 = cerrado.  muñeca: -1 = flexión, 0 = recta, +1 = extensión
//                pulgar  indice  medio   anular  menique   muneca
const Pose POSES[] = {
  {"reposo",    {0.375, 0.309, 0.329, 0.340, 0.399}, -0.072},
  {"abierta",   {0.058, 0.019, 0.022, 0.023, 0.031}, -0.058},
  {"puno",      {0.904, 0.921, 0.965, 0.976, 0.986}, -0.074},
  {"pinza",     {0.776, 0.664, 0.090, 0.116, 0.168}, -0.078},
  {"senalar",   {0.851, 0.030, 0.945, 0.974, 0.990}, -0.073},
  {"flexion",   {0.503, 0.511, 0.529, 0.550, 0.548}, -0.998},
  {"extension", {0.251, 0.021, 0.034, 0.030, 0.083}, +0.894}
};
