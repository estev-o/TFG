.PHONY: help clean run1 run1_debug run2 run2_debug run2_all run2_debug_individual run3 train_yolo apply_yolo normalize_out cnn_prepare cnn_train cnn_smoke

# Variables
VENV = .venv
PYTHON = $(VENV)/bin/python
SCRIPT_FILTRO = scripts/01_script_filtro.py
SCRIPT_DETECT = scripts/02_recortar_algas.py
SCRIPT_YOLO = scripts/04_recortar_yolo.py
SCRIPT_TRAIN = scripts/05_entrenar_yolo.py
SCRIPT_APPLY = scripts/06_aplicar_yolo.py
SCRIPT_NORM = scripts/07_normalizar_recortes.py
SCRIPT_CNN_PREPARE = cnn/08_build_split_manifest.py
SCRIPT_CNN_TRAIN = cnn/9_train_regression.py
OUTPUT_DIR = out
N ?= 10
VAL ?= 0.2
REVIEW ?=
REVIEW_MAX ?= 900
YOLO_DATA ?= yolo/data.yaml
YOLO_MODEL ?= yolov8n.pt
YOLO_EPOCHS ?= 60
YOLO_BATCH ?= 8
YOLO_IMG ?= 640
YOLO_DEVICE ?= cpu
APPLY_MODEL ?= runs_yolo/kelp/weights/best.pt
APPLY_DATASET ?= dataset/Kelps_database_photos/Photos_kelps_database
APPLY_NUM ?= 10
APPLY_IMG ?= 640
APPLY_OUT ?= $(OUTPUT_DIR)
APPLY_DEVICE ?= cpu
NORM_INPUT ?= out/out
NORM_CSV ?= dataset/kelp_photos_filtered.csv
NORM_OUTPUT ?= out_img_norm
NORM_OPEN_KERNEL ?= 1
NORM_KEEP_REL ?= 0.25
NORM_DEBUG ?= 0
CNN_LABELS ?= dataset/kelp_photos_filtered.csv
CNN_IMAGES ?= out_img_norm
CNN_PATTERN ?= *_yolo.png
CNN_MANIFEST ?= cnn/manifest.csv
CNN_SPLIT_DIR ?= cnn/splits
CNN_TRAIN_RATIO ?= 0.7
CNN_VAL_RATIO ?= 0.15
CNN_TEST_RATIO ?= 0.15
CNN_SEED ?= 42
CNN_TRAIN_CSV ?= cnn/splits/train.csv
CNN_VAL_CSV ?= cnn/splits/val.csv
CNN_RUN_DIR ?= cnn/runs/baseline
CNN_MODEL ?= resnet18
CNN_PRETRAINED ?= 0
CNN_EPOCHS ?= 20
CNN_BATCH ?= 8
CNN_LR ?= 1e-4
CNN_WD ?= 1e-4
CNN_LOSS ?= mae
CNN_HUBER_DELTA ?= 1.0
CNN_IMG ?= 224
CNN_WORKERS ?= 4
CNN_DEVICE ?= auto
CNN_AMP ?= 0
CNN_MAX_TRAIN ?= 0
CNN_MAX_VAL ?= 0

# Colores para output
BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[0;33m
NC = \033[0m # No Color

help: ## Muestra esta ayuda
	@echo "$(BLUE)Comandos disponibles:$(NC)"
	@grep -E '^[a-zA-Z_0-9]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Uso con N imágenes:$(NC) make run2 N=20"

clean: ## Limpia archivos de salida
	@echo "$(YELLOW)Limpiando $(OUTPUT_DIR)/...$(NC)"
	rm -rf $(OUTPUT_DIR)/*
	rm -rf yolo/*
	@echo "$(GREEN)Limpio$(NC)"

run1: ## Ejecuta filtro (N=número de imágenes, default 10)
	@echo "$(BLUE)Ejecutando filtro en $(N) imágenes...$(NC)"
	$(PYTHON) $(SCRIPT_FILTRO) --num_samples $(N)
	@echo "$(GREEN)Filtro completado$(NC)"

run1_debug: ## Dry-run del filtro (no borra fotos, solo muestra cuántas se eliminarían)
	@echo "$(BLUE)Dry-run del filtro (sin borrar nada)...$(NC)"
	$(PYTHON) $(SCRIPT_FILTRO) --dry-run
	@echo "$(GREEN)Dry-run completado$(NC)"

run2: ## Recorta y guarda algas (N=número de imágenes, default 10)
	@echo "$(BLUE)Recortando algas en $(N) imágenes...$(NC)"
	$(PYTHON) $(SCRIPT_DETECT) --num_samples $(N)
	@echo "$(GREEN)Recorte completado$(NC)"

run2_debug: ## Recorta algas con debug (N=número de imágenes, default 10)
	@echo "$(BLUE)Recortando algas en $(N) imágenes (modo debug)...$(NC)"
	$(PYTHON) scripts/03_debug_recortar_algas.py --num_samples $(N) --debug
	@echo "$(GREEN)Recorte completado$(NC)"
	@echo "$(YELLOW)Ver carpetas *_debug en $(OUTPUT_DIR)/$(NC)"

run2_all: ## Recorta TODAS las algas
	@echo "$(BLUE)Recortando TODAS las algas...$(NC)"
	$(PYTHON) $(SCRIPT_DETECT)
	@echo "$(GREEN)✓ Recorte completo$(NC)"

run2_debug_individual: ## Debug de una imagen específica (ej: make run2_debug_individual CODIGO=GI405)
	@if [ -z "$(CODIGO)" ]; then \
		echo "$(YELLOW)Error: Debes especificar CODIGO$(NC)"; \
		echo "$(YELLOW)Uso: make run2_debug_individual CODIGO=GI405$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Debug individual de imagen $(CODIGO)...$(NC)"
	$(PYTHON) scripts/03_debug_recortar_algas.py $(CODIGO)
	@echo "$(GREEN)✓ Debug completado$(NC)"

# run3 args: N=num_samples, VAL=val_split, REVIEW=--review para modo interactivo, REVIEW_MAX=900
run3: ## Genera dataset YOLO (yolo/images+labels y data.yaml)
	@echo "$(BLUE)Generando dataset para YOLO con $(N) imágenes (val_split=$(VAL))...$(NC)"
	$(PYTHON) $(SCRIPT_YOLO) --num_samples $(N) --val_split $(VAL) --overwrite $(if $(REVIEW),$(REVIEW) --review_max_size $(REVIEW_MAX),)
	@echo "$(GREEN)Dataset YOLO listo en yolo/$(NC)"

# train_yolo args: YOLO_MODEL=yolov8n.pt YOLO_DATA=yolo/data.yaml YOLO_EPOCHS YOLO_BATCH YOLO_IMG YOLO_DEVICE
train_yolo: ## Entrena YOLOv8 con el data.yaml generado
	@echo "$(BLUE)Entrenando YOLO: modelo=$(YOLO_MODEL), epochs=$(YOLO_EPOCHS), batch=$(YOLO_BATCH), imgsz=$(YOLO_IMG), device=$(YOLO_DEVICE)$(NC)"
	$(PYTHON) $(SCRIPT_TRAIN) --model $(YOLO_MODEL) --data $(YOLO_DATA) --epochs $(YOLO_EPOCHS) --batch $(YOLO_BATCH) --imgsz $(YOLO_IMG) --device $(YOLO_DEVICE)
	@echo "$(GREEN)Entrenamiento lanzado (resultados en runs_yolo/)$(NC)"

# apply_yolo args: APPLY_MODEL=best.pt APPLY_DATASET=dir APPLY_NUM APPLY_IMG APPLY_OUT APPLY_DEVICE
apply_yolo: ## Aplica YOLO a imágenes aleatorias y recorta a $(OUTPUT_DIR)/
	@echo "$(BLUE)Aplicando YOLO sobre $(APPLY_NUM) imágenes aleatorias...$(NC)"
	$(PYTHON) $(SCRIPT_APPLY) --model $(APPLY_MODEL) --dataset $(APPLY_DATASET) --num_images $(APPLY_NUM) --imgsz $(APPLY_IMG) --output_dir $(APPLY_OUT) --device $(APPLY_DEVICE)
	@echo "$(GREEN)Recortes guardados en $(APPLY_OUT)/$(NC)"

# normalize_out args: NORM_INPUT NORM_CSV NORM_OUTPUT NORM_OPEN_KERNEL NORM_KEEP_REL NORM_DEBUG(0|1)
normalize_out: ## Genera máscaras binarias en NORM_OUTPUT (debug opcional en la misma carpeta)
	@if [ -z "$(NORM_OUTPUT)" ] || [ "$(NORM_OUTPUT)" = "/" ] || [ "$(NORM_OUTPUT)" = "." ]; then \
		echo "$(YELLOW)Error: NORM_OUTPUT inválido ($(NORM_OUTPUT))$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Limpiando $(NORM_OUTPUT)/ antes de normalizar...$(NC)"
	@mkdir -p $(NORM_OUTPUT)
	@find $(NORM_OUTPUT) -mindepth 1 -maxdepth 1 -exec rm -rf {} +
	@echo "$(BLUE)Normalizando recortes en $(NORM_INPUT) usando $(NORM_CSV) (open_kernel=$(NORM_OPEN_KERNEL), keep_rel=$(NORM_KEEP_REL), debug=$(NORM_DEBUG))...$(NC)"
	$(PYTHON) $(SCRIPT_NORM) --input_dir $(NORM_INPUT) --csv $(NORM_CSV) --output_dir $(NORM_OUTPUT) --open_kernel $(NORM_OPEN_KERNEL) --keep_rel $(NORM_KEEP_REL) $(if $(filter 1 true TRUE yes YES,$(NORM_DEBUG)),--debug,)
	@echo "$(GREEN)Normalización completa en $(NORM_OUTPUT)/$(NC)"

cnn_prepare: ## Genera manifest.csv y train/val/test para CNN en un único paso
	@echo "$(BLUE)Generando manifest + split CNN (seed=$(CNN_SEED))...$(NC)"
	$(PYTHON) $(SCRIPT_CNN_PREPARE) --labels-csv $(CNN_LABELS) --images-dir $(CNN_IMAGES) --pattern "$(CNN_PATTERN)" --manifest-output $(CNN_MANIFEST) --split-output-dir $(CNN_SPLIT_DIR) --train-ratio $(CNN_TRAIN_RATIO) --val-ratio $(CNN_VAL_RATIO) --test-ratio $(CNN_TEST_RATIO) --seed $(CNN_SEED)
	@echo "$(GREEN)CNN preparada: $(CNN_MANIFEST) y $(CNN_SPLIT_DIR)/{train,val,test}.csv$(NC)"

cnn_train: ## Entrena la CNN de regresión (HPI, IVR)
	@echo "$(BLUE)Entrenando CNN $(CNN_MODEL) en $(CNN_DEVICE) (epochs=$(CNN_EPOCHS), batch=$(CNN_BATCH))...$(NC)"
	$(PYTHON) $(SCRIPT_CNN_TRAIN) --train-csv $(CNN_TRAIN_CSV) --val-csv $(CNN_VAL_CSV) --out-dir $(CNN_RUN_DIR) --model $(CNN_MODEL) --img-size $(CNN_IMG) --epochs $(CNN_EPOCHS) --batch-size $(CNN_BATCH) --lr $(CNN_LR) --weight-decay $(CNN_WD) --loss $(CNN_LOSS) --huber-delta $(CNN_HUBER_DELTA) --workers $(CNN_WORKERS) --seed $(CNN_SEED) --device $(CNN_DEVICE) --max-train-samples $(CNN_MAX_TRAIN) --max-val-samples $(CNN_MAX_VAL) $(if $(filter 1 true TRUE yes YES,$(CNN_PRETRAINED)),--pretrained,) $(if $(filter 1 true TRUE yes YES,$(CNN_AMP)),--amp,)
	@echo "$(GREEN)Entrenamiento CNN finalizado. Salida: $(CNN_RUN_DIR)$(NC)"

cnn_smoke: ## Smoke test CNN rápido (1 época, pocas muestras, CPU)
	@echo "$(BLUE)Smoke test CNN (rápido)$(NC)"
	$(PYTHON) $(SCRIPT_CNN_TRAIN) --train-csv $(CNN_TRAIN_CSV) --val-csv $(CNN_VAL_CSV) --out-dir cnn/runs/smoke --model resnet18 --img-size 224 --epochs 1 --batch-size 4 --lr 1e-4 --weight-decay 1e-4 --loss mae --workers 0 --seed $(CNN_SEED) --device cpu --max-train-samples 64 --max-val-samples 32
	@echo "$(GREEN)Smoke test completado (cnn/runs/smoke)$(NC)"
