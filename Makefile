.PHONY: help clean run1 run2 run2_debug run2_all

# Variables
VENV = .venv
PYTHON = $(VENV)/bin/python
SCRIPT_FILTRO = script_filtro.py
SCRIPT_DETECT = detect_alga_center.py
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

run2: ## Ejecuta detección de algas (N=número de imágenes, default 10)
	@echo "$(BLUE)Detectando algas en $(N) imágenes...$(NC)"
	$(PYTHON) $(SCRIPT_DETECT) --num_samples $(N)
	@echo "$(GREEN)Detección completada$(NC)"

run2_debug: ## Ejecuta detección con debug (N=número de imágenes, default 10)
	@echo "$(BLUE)Detectando algas en $(N) imágenes (modo debug)...$(NC)"
	$(PYTHON) $(SCRIPT_DETECT) --num_samples $(N) --debug
	@echo "$(GREEN)Detección completada$(NC)"
	@echo "$(YELLOW)Ver carpetas *_debug en $(OUTPUT_DIR)/$(NC)"

run2_all: ## Ejecuta detección en TODAS las imágenes
	@echo "$(BLUE)Detectando algas en TODAS las imágenes...$(NC)"
	$(PYTHON) $(SCRIPT_DETECT)
	@echo "$(GREEN)✓ Detección completa$(NC)"
