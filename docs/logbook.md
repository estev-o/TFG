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
entrenamos utilizando la 9.1

con la
2026-03-26
Se cambia el problema de regresión (siempre usa float) a clasificación ordinal (CORAL-CNN) (https://www.sciencedirect.com/science/article/pii/S016786552030413X?via%3Dihub) 
de 0-N. Osea tantas clases como notas dan los expertos (al ser 7 en principio).

Se procederá a los mismos experimentos, hpi, ivr y both.
Justificación del cambio:
1. Las etiquetas reales son enteras y ordenadas, no continuas: HPI tiene clases `0..6` (7) e IVR `0..7` (8).
2. La regresión obliga a predecir float y luego redondear, mientras que el modelo ordinal aprende directamente fronteras entre niveles de mordida.
3. La formulación ordinal penaliza de forma natural más un error lejano (ej. 1->5) que uno cercano (1->2), que encaja mejor con la interpretación biológica de la escala.
4. Se mantiene `09.1` como baseline histórico de regresión para poder comparar objetivamente ambos enfoques con el mismo split de test.
EN CASO HPI SON 7 clases 0...6
EN CASO IVR SON 8 clases 0...7
EN CASO BOTH SON 56 clases. Pero no se tratan como 56 salidas, sino que se trata como VARIABLES DISTINTAS, pero comparten el mismo backbone, entonces el gradiente actualiza la CNN teniendo en cuenta los pesos compartidos de HPI e IVR, lo que prueba ser un poco mejor.


Al acabar el entrenamiento, ahora tenemos 6 runs
1.2.3 son las real_*, Donde se buscó comparar qué modelo era mejor, pero el problema era regresión. 
4.5.6 clas_ordinal_* Donde buscamos cual es el mejor tipo de clasificación para el problema, 2 cnns (1 HPI otra IVR) o una CNN [HPI,IVR]. Siendo todas estas de clasificación ordinal
2026-03-27
Comparativa breve en test (baseline regresión vs ordinal):
1. Baseline `real_convnext_tiny_e30`: `mae_hpi=0.6418`, `mae_ivr=1.9651`, `mae_mean=1.3034`.
2. Ordinal `hpi`: `mae_hpi=0.5745` (mejora clara en HPI frente al baseline).
3. Ordinal `ivr`: `mae_ivr=1.9656` (prácticamente igual al baseline en IVR).
4. Ordinal `both`: `mae_hpi=0.5630`, `mae_ivr=1.8653`, `mae_mean=1.2142` (mejor resultado global).
Conclusión: para esta fase, la opción recomendada es `clas_ordinal_both_convnext_tiny_e30`.
2026-03-27
Ajuste de entrenamiento en `both`: se añade ponderación explícita de pérdidas para priorizar IVR (`w_hpi=0.4`, `w_ivr=0.6`, normalizado), además añadimos early stopping. El resultado no es muco más distinto, hay menos error grande en IVR (RMSE), pero no es un cambio muy relevante
2026-04-16
Se hace una prueba de nuevo con ES y unos pesos de HPI 0.3 e IVR 0.7.
Se entrenó con este comando:
'make cnn_train CNN_MODEL=convnext_tiny CNN_PRETRAINED=1 CNN_TARGET=both CNN_BOTH_W_HPI=0.3 CNN_BOTH_W_IVR=0.7 CNN_ES_PATIENCE=8 CNN_RUN_DIR=cnn/runs/weighted_0703_convnext_tiny_es'
El resultado, comparado con el ponderado anterior (0.4/0.6) y con el run sin weight, fue:
- `mae_mean=1.1948`, mejor que el run sin weight (`1.2142`) y que el ponderado anterior (`1.2120`).
- `mae_ivr=1.8324`, también mejora frente a ambos runs anteriores.
- En exact match baja un poco respecto al run sin weight, pero mejora en acierto dentro de ±1 y ±2.
- En conjunto, esta configuración 0.3/0.7 es la mejor de las tres en error medio, aunque el RMSE sube ligeramente frente al ponderado anterior.

2026-04-16 (parte 2)
Se realizó un cambio en la loss ordinal inspirado en el paper "Rank consistent ordinal regression for neural networks with application to age estimation". La adición principal fue:
1. Implementación de pesos ordinales por umbral (`ordinal_importance_weights` en 09_train_ordinal.py) que pondera la loss BCE de forma individual para cada threshold (frontera entre clases ordinales), no solo entre targets (HPI/IVR).
2. La fórmula calcula el desbalance entre clases para cada umbral y aplica sqrt-normalization, siguiendo la formulación del paper.
3. Los pesos se guardan en config.json para reproducibilidad.
4. La ponderación es complementaria a los pesos de task-level (0.3/0.7) y no cambia la arquitectura.

Próximo paso: entrenar la versión con pesos ordinales por umbral y comparar directamente contra la mejor run actual (0.3/0.7, `mae_mean=1.1948`). Si la mejora es marginal (<1%), pasaremos directamente a probar ConvNeXt-Small. Si hay mejora significativa (>1%), exploraremos un CORAL aún más estricto con cabeza de softmax per-threshold.

2026-04-16 (parte 3)
Se guardó la ejecución en '3_1_CORAL_1_cnvnxt_es', la comparación de sus resultados son:
- `mae_mean=1.2242`, peor que la mejor run actual (`1.1948`, 0.3/0.7) y también peor que la ponderada anterior (`1.2120`).
- `mae_ivr=1.8582`, también queda por detrás de ambas referencias.
- En exact match y acierto dentro de ±1/±2 no mejora de forma consistente frente a 0.3/0.7.
- Conclusión: la ponderación por umbral inspirada en el paper no aporta mejora en este caso frente a la 0.3/0.7, así que la run que sigue siendo mejor overall es la 0.3/0.7. El siguiente paso razonable es probar ConvNeXt-Small si el PC lo permite; si no, ya no compensa seguir afinando esta variante antes de cambiar de backbone.
2026-04-16 (parte 4)
Se hizo un rollback a como se entrenaba antes de adaptarlo al CORAL y se decidió simplemente tratar de subir al small

2026-04-17
Comparativa entre `3_2_small_b4_cnvnxt` y la mejor run hasta ahora (`3_weighted_0307_cnvnxt_es`), manteniendo la misma idea de entrenamiento (`both`, pesos `0.3/0.7`, early stopping):
1. `convnext_small` no mejora el mejor `convnext_tiny`: `mae_mean=1.2858` frente a `1.1948`.
2. En HPI queda prácticamente igual, pero un poco peor: `mae_hpi=0.5616` frente a `0.5573`.
3. El empeoramiento viene sobre todo de IVR: `mae_ivr=2.0100` frente a `1.8324`.
4. También cae en acierto global discreto: `acc_mean exact=0.5201` vs `0.5451`, `within_1=0.7787` vs `0.8087`, `within_2=0.8288` vs `0.8395`.
5. Conclusión: en esta configuración, subir el backbone a `ConvNeXt-Small` no compensa; la mejor run sigue siendo `3_weighted_0307_cnvnxt_es`.

Pero esto puede deberse a que el batch usado en small fué 4, se repetirá el experimento con un batch de 8 para buscar mejora

15 y 26 predicciones que realmente eran 7 se predijeron como 0 y 1. Igualmente hay 17, 11 y 10 que eran 0 1 y 2 y se predijeron como 6. También hay 53 0s que se predijeron como 7. Estos fallos tan lejanos hacen que baje mucho el IVR.

2026-04-17 (parte 2)
Se decide probar una modificación más dirigida al problema real de IVR:
1. Se mantiene la loss ordinal actual como base.
2. Se añade una penalización extra de distancia solo para IVR, calculada sobre la clase esperada suave (`sum(sigmoid(logits_ivr))`) frente a la etiqueta real.
3. La idea es castigar más errores lejanos tipo `0->7`, `7->0`, `1->6`, etc., sin tocar HPI porque ahí no parece estar el problema principal.
4. Se deja como opción configurable (`none`/`huber`/`mse` + peso) para poder compararlo limpiamente contra la baseline actual.
