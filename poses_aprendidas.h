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
  {"reposo",    {0.390, 0.301, 0.311, 0.333, 0.406}, -0.086},
  {"abierta",   {0.067, 0.022, 0.019, 0.021, 0.042}, -0.073},
  {"puno",      {0.911, 0.921, 0.973, 0.982, 0.985}, -0.089},
  {"pinza",     {0.776, 0.675, 0.054, 0.049, 0.068}, -0.099},
  {"senalar",   {0.860, 0.024, 0.942, 0.980, 0.991}, -0.073},
  {"flexion",   {0.513, 0.503, 0.549, 0.519, 0.574}, -0.998},
  {"extension", {0.249, 0.016, 0.043, 0.035, 0.193}, +0.895}
};
