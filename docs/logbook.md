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