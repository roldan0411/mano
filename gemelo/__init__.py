"""
============================================================
 PRÓTESIS MIOELÉCTRICA CON RECONOCIMIENTO DE GESTOS POR IA
 Paquete gemelo — datasets, aprendizaje contrastivo y métricas
------------------------------------------------------------
 Roldán Rodríguez — Ing. CCIA
============================================================

Módulos:
  config          rutas (datos fuera de OneDrive) y semillas
  datos_emg       D1: EMG sintético con dos vistas físicas por ventana
  datos_poses     D2: poses de mano de 21 puntos (topología MediaPipe)
  datos_externos  D4: EMG real público (UCI) llevado al formato del gemelo
  metricas        R², pseudo-R² de McFadden, completitud de grupos
  contrastivo     codificador 1D-CNN, SimCLR + SupCon, cabezas multitarea
  formas          red de formas de la mano (autoencoder de poses)
  estudiante      red para el ESP32 destilada del profesor contrastivo
  refuerzo        agente de servos (REINFORCE) con R² de pose
  vivo            panel en vivo (Plotly) y animación del entrenamiento
"""
