# DREAM-1 con dos Piper: despliegue en el computador del robot

Esta guía usa la integración `DreamClient` del repositorio. La estación local
controla los dos CAN, las tres cámaras y los límites de velocidad; el worker
envía observaciones a DreamAPI y recibe objetivos articulares después del IK.
Ejecuta los comandos desde la raíz de `Robotics_train_stack`. Los dos procesos
pueden compartir el mismo computador; `station_url` apunta por defecto a
`ws://127.0.0.1:8765`.

## 1. Instalar una sola vez el entorno Python

```bash
./install.sh station dream ui dev
source .venv/bin/activate
python -c 'import dream, piper_sdk; print("Dream y Piper SDK disponibles")'
```

`install.sh` crea `.venv` si falta y allí instala `piper_sdk` y `dreamlabs`.
`can_activate.sh`, que viene con `piper_sdk`, **no crea otro entorno**: cambia
los nombres, la velocidad y el estado UP de los interfaces CAN del sistema.
Es la misma operación descrita en
[`mu-stack/docs/inference/how_to_run_inference.md`](https://github.com/murobotics-ai/mu-stack/blob/main/docs/inference/how_to_run_inference.md),
pero `mu-stack` y este repositorio tienen entornos Python independientes.
Instala `ethtool`, `can-utils` y `v4l-utils` en Ubuntu si faltan:

```bash
sudo apt install ethtool can-utils v4l-utils
```

Si el terminal ha cargado ROS y mezcla su Python 3.10 con el entorno 3.12,
abre otro terminal o antepone `env -u PYTHONPATH` a cada comando `rstack`.

## 2. Configurar DreamAPI

Sigue [Set up API](https://platform.dreamlabs.ai/setup) para crear la clave y
el archivo `~/.dreamlabs/credentials`. Protege ese archivo:

```bash
chmod 600 ~/.dreamlabs/credentials
```

No incluyas la clave en un YAML ni en Git. La prueba online de `dream-preflight`
consulta los modelos disponibles; no mueve los brazos.

## 3. Identificar y activar los dos adaptadores CAN

Apaga otros procesos que usen los Piper. Conecta un adaptador USB-CAN por brazo
y verifica qué brazo corresponde a cada `bus-info`. En la máquina usada por
`mu-stack`, los ejemplos eran `1-5:1.0` izquierdo y `1-6:1.0` derecho; no los
asumas si cambió el puerto USB:

```bash
ip -br link show type can
sudo ethtool -i can0
sudo ethtool -i can1
```

Si los interfaces tienen otros nombres, usa esos nombres para `ethtool -i`.
Anota el campo `bus-info` de cada adaptador y ejecuta el wrapper de este repo,
que localiza el `can_activate.sh` dentro de **esta** `.venv` y activa 1 Mbit/s:

```bash
bash scripts/activate_piper_can.sh BUS_IZQUIERDO BUS_DERECHO
# Ejemplo solo si confirmaste el cableado físico:
# bash scripts/activate_piper_can.sh 1-5:1.0 1-6:1.0
```

El wrapper llama al script oficial del Piper SDK con `sudo`, primero para
`can0` (izquierdo) y después para `can1` (derecho), e imprime el estado final.
Antes de seguir, ambos deben estar UP a 1000000 bit/s. La configuración está en
[`piper_bimanual_dream1.yaml`](../configs/robots/piper_bimanual_dream1.yaml).
El perfil usa `sdk_interface: v2`; cambia a `v3` solo si confirmaste que **los
dos** firmwares soportan ese protocolo. La prueba de movimiento debe empezar
con un brazo por vez y comprobar el lado físico antes de permitir ambos.

## 4. Identificar las tres cámaras

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-path/*video-index0
```

Compara las rutas persistentes con `configs/robots/piper_bimanual_dream1.yaml`:
`left` es la muñeca izquierda, `right` la derecha y `top` la ZED Mini. En el
montaje de referencia las muñecas capturan 640x480 y la ZED 1344x376 (ambos
ojos); se usa solo el ojo izquierdo y Dream recibe tres JPEG de 640x360.
Corrige las rutas si el computador o el hub USB cambió. El perfil hace recorte
central de las muñecas; la corrección fisheye aproximada **no está activa**.
Para activarla en producción, primero calibra cada lente y configura su
`fisheye_calibration_file` medido.

## 5. Confirmar el montaje y el URDF

Dream usa `assets/robots/piper/piper_arm.urdf`, un modelo de **un brazo**,
para ambos lados. Su archivo `piper_arm.json` define el TCP y los ejes del
gripper. No envíes el URDF bimanual de HandUMI a `ik.arm` sin adaptar su
cadena: Dream extrae un brazo de `base` a `flange`.

El URDF bimanual reciente de HandUMI coloca los dos `base_link` paralelos a
0.68 m entre sí. Su calibración de **simulación** coloca la mesa 0.03 m bajo
el plano de las bases y el centro de referencia 0.325 m delante. Comprueba
esas medidas en el montaje físico. La configuración Dream pone el soporte
central a 0.34 m de cada base, pero su radio de 0.08 m y altura de 1.0 m aún
son supuestos; mídelo antes de mover los brazos.

`ik_region.lo.z = -0.05 m` no representa el plano de la mesa (`-0.03 m` en
la calibración de simulación). El IK del cliente no comprueba colisiones entre
ambos brazos ni da una garantía de distancia a la mesa. Evalúa las trayectorias
propuestas en un entorno despejado y mantén el paro físico accesible.

El TCP actual de `piper_arm.json` es provisional. Mide el punto entre las
puntas cerradas y los ejes del gripper instalado; puedes usar `dreamcli tcp`
como ayuda. Solo después de verificarlo en el robot físico:

```bash
rstack dream-confirm-tcp \
  --arm-file assets/robots/piper/piper_arm.json \
  --i-measured-this-on-the-real-gripper
```

Este comando escribe la confirmación en el JSON. Si cambias el URDF o el TCP,
debes confirmar de nuevo. Comprueba también los ceros y apertura de ambos
grippers, en especial si usas la punta `piper_beta` de HandUMI. El driver puede
leer una calibración local con `gripper_calibration_file` en las opciones del
robot; requiere `closed_microm` y `open_microm` por lado.

## 6. Ejecutar preflight y observación

```bash
rstack dream-preflight --offline
rstack dream-preflight --check-devices
```

El primero valida archivos y contrato sin API ni hardware. El segundo
comprueba acceso al modelo, los tres dispositivos de vídeo y `can0`/`can1`.
Si falta la confirmación del TCP, ambos deben detenerse: resuelve la medición,
no desactives `require_tcp_confirmation` para mover el robot.

La **estación conecta y habilita los motores al arrancar**, aunque aún no se
haya pulsado Start en la UI. Sostén los brazos y despeja la mesa antes de:

```bash
rstack station --config configs/robots/piper_bimanual_dream1.yaml
```

Abre `http://127.0.0.1:8080`, verifica las tres imágenes y deja el sistema
sin armar. En otro terminal con la misma `.venv`:

```bash
source .venv/bin/activate
rstack dream --config configs/deploy/dream1_piper.yaml --dry-run \
  --task "pick up the red block and place it in the blue bowl"
```

El perfil comienza con `dry_run: true` y el flag `--dry-run` lo fuerza de
nuevo. El worker llama al API, pero no envía acciones a la estación. Detén el
worker con Ctrl+C al terminar la observación.

## 7. Primera prueba de movimiento

Después de confirmar montaje, TCP, imágenes, sentido de articulaciones,
grippers y paro físico, cambia `dry_run: false` en
`configs/deploy/dream1_piper.yaml`. Inicia nuevamente el worker sin el flag:

```bash
rstack dream --config configs/deploy/dream1_piper.yaml \
  --task "pick up the red block and place it in the blue bowl"
```

En la UI pulsa `Validate`, `Arm` y `Start`. Haz la primera prueba con un
espacio despejado, una persona junto al paro físico y movimiento observado
de cerca. `Hold / Pause` detiene la sesión de acciones. Si el primer resultado
del API llega tras más de 60 s, el worker termina sin mandar movimiento;
reinícialo y repite `Validate`, `Arm`, `Start` con el modelo ya caliente.
Al cerrar la estación se deshabilita el torque de ambos brazos; sosténlos.

Los tests con `FakeDreamClient` validan el adaptador y el worker, pero **no**
demuestran todavía que una trayectoria de Dream sea segura en este montaje.
El preflight, la observación real y la primera prueba física cierran esa brecha.
