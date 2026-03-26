2026-01-26
1. El estado actual de /out tiene las imágenes recortadas de todo el dataset de imágenes por el yolo entrenado. Tiene 4704 imágenes. He normalizado las imágenes para que todas tengan el mismo tamaño (debido a que van a ser entrada de una red convolucional) usando el tamaño de la mayor para evitar pérdida de calidad y sesgos. Además, llegados a este punto solo he pasado las imágenes de las que tenemos información (hay match imagen-dato en csv). El resultado son 3583 imágenes guardadas en /out_img_norm
2. Para el siguiente paso, deberíamos usar SOLO los bordes del alga. Para ello una idea es usar YOLOv8-seg. No sé si podríamos usar los recortes de bounding box que usamos para entrenar la instancia de YOLOV8

2026-01-28
1. He instalado labelme para tratar de segmentar imágenes y entrenar YOLOv8-seg. Mi plan inicial es segmentar unas 200 imágenes a mano para entrenar una primera iteración del yolov8-seg. Ver como generaliza en otras 100 imágenes, corregirlas si tienen errores y reentrenar. Para ello debo seguir estos pasos
- Segmentar 200 imágenes
- Dividir 80/20
- convertir a formato yolo8-seg
- Entrenar
- Inferir 100 más
- Convertir de nuevo a un formato legible para labelme
- Corregir imperfecciones
- Repetir
Hoy segmenté 23 imágenes

NOTAS DE LA REUNIÓN:
La segmentación de nuestro yolo actual es suficiente, no es necesario entrenar un YOLO mayor de momento. Tampoco debemos cambiar el YOLO usado por un YOLO-seg.
El objetivo ahora es sacar la máscara de las algas normalizadas modificando 07_normalizar_recorte.py para quitar letras y reglas para que quede solo una máscara de el alga como entrada de la CNN:
1. Todas imágenes de mismo tamaño -> Lo hacemos con 2 pasadas a el dataset. 1 para buscar la mayor y otra para normalizar (añadiendo color fondo) todas las imágenes para no perder calidad
2. Binarización de Alga / fondo

2026-02-10
Implementé la nueva normalización del recorte de YOLO en 07_normalizar_recorte.py con estos cambios:
ejecuté:
$ make apply_yolo APPLY_MODEL=runs_yolo/kelp4/weights/best.pt APPLY_NUM=0
para aplicar el YOLO a el dataset (lo había borrado antes)

el output fué:
Recortes guardados: 4705
Sin detección: 37
Errores: 155
Salida: out

ejecuté:
$ make normalize_out NORM_INPUT=out NORM_OUTPUT=out_img_norm NORM_DEBUG=1
para ver como iba, como vi que el debug daba bien le di una pasada a todo el dataset
$ make normalize_out NORM_INPUT=out NORM_OUTPUT=out_img_norm NORM_DEBUG=0

output:
  Imágenes encontradas: 4704
  Con match CSV: 3583
  Sin match CSV: 1121
  Ilegibles: 0
  Sin máscara válida: 0
  Tamaño objetivo: 4496x4496
  Guardadas: 3583
  Salida: out_img_norm

2026-02-14
Nos dimos cuenta de un fallo crítico, habíamos borrado del CSV todas las imágenes que tenían dos 0 (no tenían ningún tipo de mordida). Aunque pensábamos que no eran relevantes, es necesario que el sistema aprenda también en qué casos poner 0, para ello se debe entrenar ese tipo de algas también.

Cambié 01_script_filtro.py para que no borre las que tiene dos filas como 0:
$ make run1

output:
  Salida escrita: /home/estevo/TFG/dataset/kelp_photos_filtered.csv  (5486 filas)
  Salida escrita: /home/estevo/TFG/dataset/kelp_photos_filtered.xlsx  (5486 filas)
  Filas eliminadas por valores no numéricos: 4
  Filas con HPI e IVR ambos 0 conservadas: 1364
  Revisión de /home/estevo/TFG/dataset/Kelps_database_photos/Photos_kelps_database: 4984 archivos de imagen, 7 eliminados, 4977 quedan, 0 ignorados por extensión

Ahora, con este cambio debemos normalizar otra vez las fotos que detectó el YOLO (estas se hicieron sobre el dataset entero, solo se filtra en el paso 07)

$ make normalize_out NORM_INPUT=out NORM_OUTPUT=out_img_norm NORM_DEBUG=0

output:
  Imágenes encontradas: 4704
  Con match CSV: 4699
  Sin match CSV: 5
  Ilegibles: 0
  Sin máscara válida: 0
  Tamaño objetivo: 4496x4496
  Guardadas: 4699
  Salida: out_img_norm

2026-02-18
Como revisión final. Le he dado una vuelta a mano a todas las imágenes para quitar las imágenes que estaban mal segmentadas o que fallara en detectar alga
Eliminé 51 imágenes, la mayoría con segmentación fuera del alga.
Como nota, quizás es relevante ver que en un número signfificante de imágenes se segmenta parte de la sombra del alga como si fuese alga,  también en algunas las líneas negras rectas de la separación entre mesas. Esto puede ser negativo para el entrenamiento?¿?¿?¿?¿??¿¿??

2026-03-10
Preparación de datos para la fase CNN:
1. Creé cnn/08_build_split_manifest.py para hacer en un único paso el cruce etiqueta-imagen y generar manifest.csv + splits/train,val,test
2. Ejecuté `make cnn_prepare` y quedó: `manifest=4648`, `train=3253`, `val=697`, `test=698`.

2026-03-10
Entrenamiento base CNN:
1. Implementé `cnn/9_train_regression.py` para regresión de 2 salidas (`hpi`, `ivr`) y añadí targets de Makefile para entrenar (`cnn_train`) y prueba rápida (`cnn_smoke`).
2. Ejecuté `make cnn_smoke` (CPU, 1 época, subset pequeño) para validar pipeline end-to-end y se generaron checkpoints + `metrics.csv` en `cnn/runs/smoke`.

2026-03-11
Primer entrenamiento real CNN (EfficientNet-B0, 30 epochs, pretrained, GPU):
1. Mejor validación en epoch 18: `mae_mean=1.2671`, `mae_hpi=0.6875`, `mae_ivr=1.8467`.
2. Epoch final (30): `mae_mean=1.3782` (ligero sobreajuste al final).

Métricas que usaremos para comparar los 3 modelos:
- `mae_mean` (principal): promedio del error absoluto de HPI e IVR. Menor es mejor.
- `mae_hpi`: error absoluto medio en la predicción de mordidas de peces (HPI). Menor es mejor.
- `mae_ivr`: error absoluto medio en la predicción de mordidas de invertebrados (IVR). Menor es mejor.
- `val_loss` (secundaria): pérdida en validación para ver estabilidad y detectar sobreajuste. Menor es mejor.
2026-03-12
Segundo entrenamiento real CNN (ResNet18, 30 epochs, pretrained, GPU):
1. Ejecuté: `make cnn_train CNN_DEVICE=cuda CNN_MODEL=resnet18 CNN_PRETRAINED=1 CNN_EPOCHS=30 CNN_BATCH=4 CNN_AMP=1 CNN_WORKERS=2 CNN_RUN_DIR=cnn/runs/real_resnet18_e30`
2. Mejor validación en epoch 19: `mae_mean=1.3386`, `mae_hpi=0.6859`, `mae_ivr=1.9912`.
3. Epoch final (30): `mae_mean=1.3881`.
2026-03-25
Evaluación en test de las CNN:
1. Implementé `cnn/10_test_cnn.py` y target `make cnn_test` para inferencia sobre `cnn/splits/test.csv` usando `best.pt` de cada run.
2. El script guarda en cada run: `test_eval/metrics_test.json`, `test_eval/predictions_test.csv`, `5_real_vs_pred.png` y `5_residuals_hist.png`.

Métricas usadas (breve):
- `1_mae`: error absoluto medio en puntos de nota (principal; menor es mejor).
- `2_rmse`: penaliza más errores grandes (menor es mejor).
- `4_tolerance_accuracy`: porcentaje dentro de ±0.5 y ±1.0 (`acc_hpi`, `acc_ivr`, `acc_both`, `acc_mean`).
- `5_plots`: visualización real vs pred y distribución de residuales.

Interpretación de las 3 runs en test (checkpoint `best.pt`):
1. `convnext_tiny`: `mae_mean=1.3034` (`mae_hpi=0.6418`, `mae_ivr=1.9651`) -> mejor rendimiento global.
2. `efficientnet_b0`: `mae_mean=1.4058` (`mae_hpi=0.7636`, `mae_ivr=2.0479`) -> segundo mejor.
3. `resnet18`: `mae_mean=1.4560` (`mae_hpi=0.7552`, `mae_ivr=2.1568`) -> tercer mejor.
4. Patrón común: IVR sigue siendo la salida más difícil (errores mayores que HPI en los tres modelos).
2026-03-25
Nos hemos dado cuenta de que había que usar int para la métrica de expertos y no float, hay que reentrenar. Pero como sabemos que la mejor es ConvNeXT nos ahorra algo de trabajo.
2026-03-25
Ajuste del flujo de reentrenamiento:
1. Se mantiene `ConvNeXt-Tiny` y se separa el entrenamiento en 3 runs: `hpi`, `ivr` y `both`.
2. Se confirma uso de `CNN_AMP=1` (mixed precision en CUDA) para reducir VRAM y acelerar.
3. Siguiente paso: lanzar los 3 entrenamientos y luego `cnn_test` por cada run para comparar en test.
