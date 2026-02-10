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