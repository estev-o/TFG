.PHONY: help clean run1 run1_debug run2 run2_debug run2_all run2_debug_individual run3 train_yolo apply_yolo normalize_out

# Variables
VENV = .venv
PYTHON = $(VENV)/bin/python
SCRIPT_FILTRO = scripts/01_script_filtro.py
SCRIPT_DETECT = scripts/02_recortar_algas.py
SCRIPT_YOLO = scripts/04_recortar_yolo.py
SCRIPT_TRAIN = scripts/05_entrenar_yolo.py
SCRIPT_APPLY = scripts/06_aplicar_yolo.py
SCRIPT_NORM = scripts/07_normalizar_recortes.py
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
NORM_DEBUG ?= out_img_norm_debug
NORM_OPEN_KERNEL ?= 3

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

# normalize_out args: NORM_INPUT NORM_CSV NORM_OUTPUT NORM_DEBUG NORM_OPEN_KERNEL
normalize_out: ## Genera máscaras binarias (alga=255, fondo=0) + paneles debug
	@echo "$(BLUE)Normalizando recortes en $(NORM_INPUT) usando $(NORM_CSV) (open_kernel=$(NORM_OPEN_KERNEL))...$(NC)"
	$(PYTHON) $(SCRIPT_NORM) --input_dir $(NORM_INPUT) --csv $(NORM_CSV) --output_dir $(NORM_OUTPUT) --debug_dir $(NORM_DEBUG) --open_kernel $(NORM_OPEN_KERNEL)
	@echo "$(GREEN)Normalización completa en $(NORM_OUTPUT)/ y debug en $(NORM_DEBUG)/$(NC)"
