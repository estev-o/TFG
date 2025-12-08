.PHONY: help clean run1 run2 run2_debug run2_all

# Variables
VENV = .venv
PYTHON = $(VENV)/bin/python
SCRIPT_FILTRO = script_filtro.py
SCRIPT_DETECT = recortar_algas.py
OUTPUT_DIR = out
N ?= 10

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
	@echo "$(GREEN)Limpio$(NC)"

run1: ## Ejecuta filtro (N=número de imágenes, default 10)
	@echo "$(BLUE)Ejecutando filtro en $(N) imágenes...$(NC)"
	$(PYTHON) $(SCRIPT_FILTRO) --num_samples $(N)
	@echo "$(GREEN)Filtro completado$(NC)"

run2: ## Recorta y guarda algas (N=número de imágenes, default 10)
	@echo "$(BLUE)Recortando algas en $(N) imágenes...$(NC)"
	$(PYTHON) $(SCRIPT_DETECT) --num_samples $(N)
	@echo "$(GREEN)Recorte completado$(NC)"

run2_debug: ## Recorta algas con debug (N=número de imágenes, default 10)
	@echo "$(BLUE)Recortando algas en $(N) imágenes (modo debug)...$(NC)"
	$(PYTHON) $(SCRIPT_DETECT) --num_samples $(N) --debug
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
	$(PYTHON) debug_individual.py $(CODIGO)
	@echo "$(GREEN)✓ Debug completado$(NC)"
