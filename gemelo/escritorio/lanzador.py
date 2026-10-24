"""Proceso del motor. Módulo liviano: la ventana no carga PyTorch, solo el proceso hijo."""


def correr_motor(salida, entrada):
    from gemelo.motor.servicio import proceso

    proceso(salida, entrada)


def correr_bebe(salida, entrada):
    from gemelo.libre.bebe import proceso

    proceso(salida, entrada)
