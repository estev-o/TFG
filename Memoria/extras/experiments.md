2025-12-09 - csv-filter
comando: python scripts/01_script_filtro.py
cambio: genera kelp_photos_filtered.csv/xlsx con columnas Photo_cod,HPI,IVR; elimina NaN y filas con ambos 0
resultado: 4122 filas (5490 -> 4122; 4 no numericas; 1364 ambos 0)

2025-12-09 - yolo-dataset
comando: python scripts/04_recortar_yolo.py --val_split 0.2 --overwrite
cambio: genera yolo/images, yolo/labels y data.yaml desde fotos del dataset
resultado: yolo/images con 636 train / 146 val

2025-12-09 - yolo-train-kelp4
comando: python scripts/05_entrenar_yolo.py --model yolov8n.pt --data yolo/data.yaml --epochs 60 --imgsz 640 --batch 8 --device 0 --project runs_yolo --name kelp4
cambio: entrenamiento YOLOv8n con recortes (runs_yolo/kelp4)
resultado: P=0.99895 R=1.0 mAP50=0.995 mAP50-95=0.983 (epoch 60)

2026-01-26
Normalización de las imágenes que nos recortó el YOLO. 

make normalize_out NORM_INPUT=out/out NORM_CSV=dataset/kelp_photos_filtered.csv NORM_OUTPUT=out_img_norm NORM
_BG=127

  Imágenes encontradas: 4704
  Con match CSV: 3583
  Sin match CSV: 1121
  Ilegibles: 0
  Tamaño objetivo: 4496x4496
  Guardadas: 3583
  Salida: out_img_norm