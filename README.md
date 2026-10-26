# Camara Virtual Goku

Camara virtual para Windows con efecto de teletransportacion estilo Goku para usar en Discord, Teams u otras apps compatibles con camara virtual.

## Que hace

- Captura la webcam real.
- Emite una camara virtual mediante `Unity Capture`.
- Permite capturar un fondo limpio.
- Hace desaparicion y reaparicion con animacion.
- Permite activar el efecto con teclas o con el gesto del dedo en la frente.

## Requisitos

- Windows
- Python 3.12
- `Unity Capture` instalado

## Instalacion

En `cmd` o PowerShell, dentro de la carpeta del proyecto:

```bat
py -3.12 -m venv .venv
".\.venv\Scripts\python.exe" -m pip install --upgrade pip
".\.venv\Scripts\python.exe" -m pip install opencv-python mediapipe pyvirtualcam numpy
```

## Ejecucion

Sin preview:

```bat
".\.venv\Scripts\python.exe" ".\virtual_cam_test.py"
```

Con preview local:

```bat
".\.venv\Scripts\python.exe" ".\virtual_cam_test.py" --preview
```

Los modelos de `MediaPipe` se descargan automaticamente la primera vez.

## Uso

1. Ejecuta el script.
2. Abre Discord, Teams o la app que quieras usar.
3. Selecciona la camara virtual de `Unity Capture`.
4. Sal del plano y pulsa `F6` para capturar un fondo limpio.
5. Vuelve a entrar en plano.
6. Haz el gesto de Goku o usa las teclas manuales.

## Controles

- `F6`: capturar fondo limpio
- `F7`: borrar fondo limpio
- `F8`: desaparecer
- `F9`: reaparecer
- gesto Goku: indice estirado tocando la frente durante unos `0.5s`

## Notas

- Para que el efecto quede bien, la camara debe quedarse fija.
- Evita cambios fuertes de luz despues de capturar el fondo.
- Si la webcam no abre, cierra Discord, Teams, Zoom, OBS o cualquier app que pueda estar usandola.

## Licencia

Este repositorio es publico solo para visualizacion.

Consulta [`LICENSE`](./LICENSE) para los terminos completos.
