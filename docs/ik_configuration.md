# Configuración de inferencia e IK

Ejecuta los comandos desde la raíz del proyecto. Copia `.env.example` a `.env`
solo en una instalación nueva; conserva tu archivo si ya contiene credenciales.

```dotenv
DREAM_API_KEY="tu_clave"
RSTACK_DREAM_MODEL="dreamlabs/dream-1-beta"
RSTACK_DREAM_SOLVER="dls"
RSTACK_DREAM_TASK="Pick up the purple bunch of grapes and place it in the yellow basket."
RSTACK_STATION_URL="ws://127.0.0.1:8765"
RSTACK_DREAM_DRY_RUN="true"
```

Las variables exportadas tienen prioridad sobre `.env`; los valores no vacíos
sobrescriben el YAML. `--task` tiene prioridad sobre ambos; una tarea guardada
en la interfaz de la estación prevalece sobre los tres. `--dry-run` siempre
impide enviar acciones. Las credenciales nunca se guardan en los perfiles.

## Solvers

- `dls`: IK incremental local, con amortiguamiento adaptativo cerca de
  singularidades, atracción suave a la postura de referencia y límites por paso.
  Usa el URDF y el TCP del archivo `arm_file`. Recibe poses absolutas de Dream,
  calcula FK de los encoders para la observación y convierte cada pose de salida
  en un objetivo articular absoluto. Las columnas rot6 son las dos primeras
  columnas de la matriz de rotación. El punto de partida de cada petición es
  la observación medida; no se confunde un objetivo enviado con uno ejecutado.
- `traj`: IK de trayectoria del SDK, con su configuración y calentamiento.
- `tick`: IK por pasos del SDK.

DLS está disponible para Piper. El perfil de un solo brazo conserva el IK del
SDK. Los parámetros numéricos de DLS están en el bloque `dls` del perfil de
inferencia; las velocidades se limitan a la menor entre el URDF y 0.5 rad/s.
Una corrección incremental no garantiza alcanzar un objetivo inalcanzable.
El modo DLS no aplica el suavizado de trayectoria del SDK.

La teleoperación convierte el desplazamiento relativo del controlador, desde
su ancla, en una pose absoluta. Esa pose entra al mismo módulo DLS. Los deltas
no se envían directamente a `JointCtrl`: el driver recibe radianes absolutos.
El mapa de coordenadas transforma vectores del mundo del headset a la base
del robot, independientemente de las orientaciones al activar el clutch.
La postura medida al sincronizar el estado es la referencia de regularización
en VR; Dream utiliza `rest` del archivo del brazo. Los paquetes VR repetidos o
fuera de orden no generan nuevos pasos, y el intervalo de gripper también
respeta el máximo de dos periodos configurado por defecto.
Selecciona `ik.solver: dls` y ajusta `ik.dls` en
`configs/teleop/piper_vr.example.yaml`. La velocidad respeta el menor límite
entre `motion.joint_speed_cap_rad_s` y el perfil de la estación; los límites
articulares respetan también los límites mecánicos de Piper.
`motion.source_to_robot_rotation` debe ser una rotación ortonormal con
determinante +1. Los parámetros de VR se mantienen en YAML; las variables
`RSTACK_DREAM_*` afectan únicamente a la inferencia Dream.

En ambos flujos, el error de posición se expresa en metros y el de orientación
en radianes. `orientation_weight` pondera las filas de orientación del error y
del Jacobiano. Si el error angular supera `orientation_hold_rad`, ese paso
resuelve solo posición. `rest_gain` regulariza hacia la postura de referencia;
`base_damping` y `singularity_damping` estabilizan la inversión.
Los perfiles Dream rechazan claves desconocidas para detectar errores de escritura.

## Organización

- `src/robotics_stack/kinematics/`: transformaciones, FK/Jacobiano URDF y DLS.
- `src/robotics_stack/policies/`: contrato Dream y adaptación de poses a joints.
- `src/robotics_stack/robots/`: drivers, cámaras y unidades de hardware.
- `src/robotics_stack/services/`: comunicación y ciclo de ejecución.
- `src/robotics_stack/tools/`: entradas de herramientas de inspección offline.
- `scripts/`: wrappers pequeños; la activación CAN permanece como utilidad shell.
- `configs/robots/`: dispositivos, esquema, Home y límites de la estación.
- `configs/deploy/`: modelo, tarea, solver y cadencia de inferencia.
- `configs/cameras/`: perfiles de preprocesamiento y calibración visual.

Al instalar el paquete, las herramientas también se exponen como
`rstack-preflight`, `rstack-measure-latency`, `rstack-prepare-sim-replay` y
`rstack-preview-fisheye`. Cada comando acepta `--help`; los archivos de
`scripts/` conservan las mismas entradas para ejecución desde el repositorio.

## Validación

El perfil Piper inicia con `dry_run: true`. Los cambios del solver se han
comprobado con datos sintéticos; requieren revisión visual antes de movimiento.
El perfil Piper omite `ik_region` e `ik_keep_out`: son opcionales, no requisitos
de Dream ni un modelo del montaje. Si se configuran, DLS las aplica como
restricciones estrictas; el SDK puede usarlas como costes del solver.
DLS valida las regiones y cápsulas estáticas configuradas usando puntos de la
cadena y segmentos muestreados con un radio conservador. No comprueba colisiones
entre brazos, no conoce la mesa salvo que se modele y no certifica una ruta
continua libre de obstáculos. Rechaza salidas malformadas o no finitas.

```bash
rstack dream-preflight --offline
rstack dream --config configs/deploy/dream1_piper.yaml --dry-run
```

El segundo comando necesita una estación ya disponible y acceso a DreamAPI.
No arranca la estación ni habilita motores por sí mismo.
