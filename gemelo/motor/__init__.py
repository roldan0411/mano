"""
Motor del gemelo: simulación v2, almacén de datos, población de manos y servicio.

  gemelo2     antebrazo con geometría (músculos y electrodos que se mueven),
              amplificador por etapas, desfase continuo, calibración por sesión
  almacen     lotes de datos en disco, examen fijo, base SQLite, campeonas
  poblacion   10 manos candidatas que entrenan y compiten (Population-Based Training)
  fotos       fotos de los 7 gestos de cada campeona (mano InMoov)
  servicio    bucle continuo + API local para la aplicación de escritorio
"""
