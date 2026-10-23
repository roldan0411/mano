"""python -m gemelo.motor  ->  arranca el servicio del gemelo.

Este archivo no importa PyTorch a nivel de módulo: los procesos que simulan
datos lo vuelven a importar al arrancar (Windows), y así no cargan la GPU ni
gastan memoria de más.
"""

if __name__ == "__main__":
    from gemelo.motor.servicio import main

    main()
