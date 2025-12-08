#!/usr/bin/env python3
"""
Detección y recorte de algas - Versión producción (sin debug)
- Detecta bordes con Canny
- Filtra texto coloreado y zonas de regla
- Busca alga desde centro hacia afuera
- Recorta y redimensiona a cuadrado uniforme
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# Configuración reutilizable / constantes
IMAGE_PATTERNS = [
    '*.jfif', '*.JFIF', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG',
    '*.png', '*.PNG', '*.heic', '*.HEIC'
]
DILATE_KERNEL = np.ones((3, 3), np.uint8)
AREA_MINIMA_ALGA = 5000
OVERLAP_TEXTO_MAX = 0.3
AREA_MINIMA_REGLA = 3000
OVERLAP_REGLA_MAX = 0.5
REGLA_DILATE_ITERS = 4


def es_regla(aspect_ratio):
    """Determina si un contorno parece una regla por su aspect ratio."""
    return aspect_ratio > 3.0 or aspect_ratio < 0.33


def porcentaje_solape_contorno(contorno, area, mascara, imagen_shape):
    """Porcentaje del contorno que coincide con una máscara dada."""
    if mascara is None:
        return 0.0
    mask_contorno = np.zeros(imagen_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_contorno, [contorno], -1, 255, -1)

    overlap = cv2.bitwise_and(mask_contorno, mascara)
    overlap_area = np.count_nonzero(overlap)
    return overlap_area / area if area > 0 else 0


def porcentaje_texto_coloreado(contorno, area, mascara_texto, imagen_shape):
    """Porcentaje del contorno que coincide con la máscara de texto."""
    return porcentaje_solape_contorno(contorno, area, mascara_texto, imagen_shape)


def recolectar_imagenes(input_dir, num_samples=None):
    """Devuelve la lista de imágenes encontradas."""
    imagenes = []
    for pattern in IMAGE_PATTERNS:
        imagenes.extend(sorted(input_dir.glob(pattern)))

    if num_samples:
        imagenes = imagenes[:num_samples]

    return imagenes


def construir_mascara_regla(contornos_regla, imagen_shape):
    """Construye una máscara binaria de las reglas detectadas por forma."""
    if not contornos_regla:
        return None

    mask = np.zeros(imagen_shape[:2], dtype=np.uint8)

    for contorno in contornos_regla:
        area = cv2.contourArea(contorno)
        if area < AREA_MINIMA_REGLA:
            continue

        rect = cv2.minAreaRect(contorno)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(mask, [box], -1, 255, -1)

    if np.any(mask):
        mask = cv2.dilate(mask, DILATE_KERNEL, iterations=REGLA_DILATE_ITERS)

    return mask


def detectar_texto_coloreado(imagen_bgr):
    """Detecta texto verde y rojo usando HSV."""
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    
    # Verde
    lower_verde = np.array([82, 150, 40])
    upper_verde = np.array([92, 255, 130])
    
    # Rojo
    lower_rojo1 = np.array([0, 120, 100])
    upper_rojo1 = np.array([8, 255, 255])
    lower_rojo2 = np.array([172, 120, 100])
    upper_rojo2 = np.array([180, 255, 255])
    
    # Combinar máscaras
    mascara_verde = cv2.inRange(hsv, lower_verde, upper_verde)
    mascara_rojo1 = cv2.inRange(hsv, lower_rojo1, upper_rojo1)
    mascara_rojo2 = cv2.inRange(hsv, lower_rojo2, upper_rojo2)
    
    mascara_texto = cv2.bitwise_or(mascara_verde, mascara_rojo1)
    mascara_texto = cv2.bitwise_or(mascara_texto, mascara_rojo2)
    
    # Dilatar para capturar texto completo
    mascara_texto = cv2.dilate(mascara_texto, DILATE_KERNEL, iterations=2)
    
    return mascara_texto


def calcular_score_alga(contorno, imagen_bgr, gray, centro_x, centro_y, img_width, img_height):
    """Calcula score para determinar probabilidad de ser alga."""
    x, y, w, h = cv2.boundingRect(contorno)
    area = cv2.contourArea(contorno)
    
    score = 0.0
    
    # 1. Distancia al centro (peso principal)
    distancia_min = float('inf')
    for punto in contorno:
        px, py = punto[0]
        dist = np.sqrt((px - centro_x)**2 + (py - centro_y)**2)
        if dist < distancia_min:
            distancia_min = dist
    
    score += max(0, 150 - (distancia_min / 8))
    
    # 2. Formas
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    extent = area / (w * h) if (w * h) > 0 else 0
    perimetro = cv2.arcLength(contorno, True)
    compacidad = (4 * np.pi * area) / (perimetro * perimetro) if perimetro > 0 else 0
    
    hull = cv2.convexHull(contorno)
    hull_area = cv2.contourArea(hull)
    solidez = area / hull_area if hull_area > 0 else 0
    
    # 3. Filtros anti-texto
    margen_texto = 0.10
    margen_x = int(img_width * margen_texto)
    margen_y = int(img_height * margen_texto)
    
    en_borde = (x < margen_x or y < margen_y or 
                x + w > img_width - margen_x or y + h > img_height - margen_y)
    
    if en_borde and area < 10000:
        score -= 80
    
    if solidez < 0.5 and area < 15000:
        score -= 60
    
    if solidez > 0.7 and area < 3000:
        score -= 70
    
    if area < 2000 and (y < img_height * 0.15 or y + h > img_height * 0.85 or 
                        x < img_width * 0.15 or x + w > img_width * 0.85):
        score -= 100
    
    perimetro_area_ratio = perimetro / area if area > 0 else 0
    if perimetro_area_ratio > 0.5 and area < 20000:
        score -= 50
    
    # Filtro anti-regla
    margen_regla = 0.15
    margen_regla_x = int(img_width * margen_regla)
    margen_regla_y = int(img_height * margen_regla)
    
    cerca_lateral = (x < margen_regla_x or x + w > img_width - margen_regla_x)
    cerca_horizontal = (y < margen_regla_y or y + h > img_height - margen_regla_y)
    
    if (cerca_lateral or cerca_horizontal) and area < 8000:
        score -= 90
    
    # 4. Premios por forma de alga
    if compacidad > 0.05:
        score += 30
    
    if extent > 0.3:
        score += 20
    
    if solidez > 0.8:
        score += 25
    
    # 5. Premios por tamaño
    if area > 50000:
        score += 80
    elif area > 20000:
        score += 60
    elif area > 5000:
        score += 30
    elif area < 1000:
        score -= 30
    
    return score


def detectar_alga_desde_centro(imagen):
    """Detecta el alga buscando bordes cercanos al centro."""
    h, w = imagen.shape[:2]
    centro_x, centro_y = w // 2, h // 2
    
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    mascara_texto = detectar_texto_coloreado(imagen)
    
    edges = cv2.dilate(edges, DILATE_KERNEL, iterations=1)
    
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return None, None
    
    # Detectar regla por aspect ratio
    reglas = []
    contornos_no_regla = []
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < 1000:
            continue
            
        x, y, w_box, h_box = cv2.boundingRect(contorno)
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        
        if es_regla(aspect_ratio):
            reglas.append(contorno)
        else:
            contornos_no_regla.append(contorno)

    mascara_regla = construir_mascara_regla(reglas, imagen.shape)
    
    # Filtrar contornos
    contornos_filtrados = []
    
    for contorno in contornos_no_regla:
        area = cv2.contourArea(contorno)
        
        # Filtro área mínima
        if area < AREA_MINIMA_ALGA:
            continue

        # Filtro: dentro de la regla detectada por forma
        overlap_regla = porcentaje_solape_contorno(
            contorno, area, mascara_regla, imagen.shape
        )
        if overlap_regla > OVERLAP_REGLA_MAX:
            continue
        
        # Filtro texto coloreado
        overlap_ratio = porcentaje_texto_coloreado(
            contorno, area, mascara_texto, imagen.shape
        )
        
        if overlap_ratio > OVERLAP_TEXTO_MAX:
            continue
        
        contornos_filtrados.append(contorno)
    
    # Calcular scores
    scores = []
    for contorno in contornos_filtrados:
        score = calcular_score_alga(contorno, imagen, gray, centro_x, centro_y, w, h)
        scores.append((score, contorno))
    
    if not scores:
        return None, None
    
    # Ordenar por score
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Mejor contorno
    _, alga_contorno = scores[0]
    x, y, w_box, h_box = cv2.boundingRect(alga_contorno)
    
    return alga_contorno, (x, y, w_box, h_box)


def procesar_imagen(imagen_path):
    """Detectar alga y recortar."""
    imagen = cv2.imread(str(imagen_path))
    if imagen is None:
        return None, None, None, None
    
    contorno, bbox = detectar_alga_desde_centro(imagen)
    
    if bbox is None:
        return None, None, None, None
    
    x, y, w, h = bbox
    alga_recortada = imagen[y:y+h, x:x+w]
    nombre = imagen_path.stem
    
    return alga_recortada, w, h, nombre


def main():
    parser = argparse.ArgumentParser(description='Recortar algas para entrenamiento IA')
    parser.add_argument('--input_dir', type=str, 
                       default='dataset/Kelps_database_photos/Photos_kelps_database',
                       help='Directorio con imágenes')
    parser.add_argument('--output_dir', type=str, default='out',
                       help='Directorio de salida')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Número de imágenes a procesar')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"ERROR: No existe el directorio: {input_dir}")
        sys.exit(1)
    
    imagenes = recolectar_imagenes(input_dir, args.num_samples)

    if not imagenes:
        print(f"ERROR: No se encontraron imágenes")
        sys.exit(1)
    
    print(f"Procesando {len(imagenes)} imágenes...")
    print(f"Salida: {output_dir}/\n")
    
    temp_dir = output_dir / "temp_crops"
    temp_dir.mkdir(exist_ok=True)
    
    # Fase 1: Detectar y guardar temporales
    print("Fase 1: Detectando algas y guardando temporales...")
    metadatos = []
    tamaño_max = 0
    errores = 0
    
    for i, img_path in enumerate(imagenes, 1):
        print(f"  [{i}/{len(imagenes)}] {img_path.name}... ", end='', flush=True)
        
        try:
            alga_recortada, w, h, nombre = procesar_imagen(img_path)
        except Exception as e:
            print(f"Error: {e}")
            errores += 1
            continue
        
        if alga_recortada is None:
            print(f"No se detectó alga")
            errores += 1
        else:
            temp_path = temp_dir / f"{nombre}_temp.jpg"
            cv2.imwrite(str(temp_path), alga_recortada)
            
            metadatos.append({
                'nombre': nombre,
                'ancho': w,
                'alto': h,
                'temp_path': temp_path
            })
            lado_max = max(w, h)
            tamaño_max = max(tamaño_max, lado_max)
            print(f"OK ({w}x{h})")
    
    if tamaño_max == 0:
        print("\nNo se detectaron algas válidas")
        temp_dir.rmdir()
        sys.exit(1)
    
    print(f"\nTamaño máximo: {tamaño_max}x{tamaño_max}")
    print(f"Fase 2: Redimensionando {len(metadatos)} algas...\n")
    
    # Fase 2: Hacer cuadrado y redimensionar
    exitosos = 0
    for i, meta in enumerate(metadatos, 1):
        alga_temp = cv2.imread(str(meta['temp_path']))
        
        if alga_temp is None:
            continue
        
        h_temp, w_temp = alga_temp.shape[:2]
        lado_cuadrado = max(w_temp, h_temp)
        
        # Canvas cuadrado con fondo
        alga_cuadrada = np.full((lado_cuadrado, lado_cuadrado, 3), 127, dtype=np.uint8)
        
        offset_y = (lado_cuadrado - h_temp) // 2
        offset_x = (lado_cuadrado - w_temp) // 2
        alga_cuadrada[offset_y:offset_y+h_temp, offset_x:offset_x+w_temp] = alga_temp
        
        # Redimensionar
        alga_final = cv2.resize(alga_cuadrada, (tamaño_max, tamaño_max), 
                               interpolation=cv2.INTER_LANCZOS4)
        
        output_path = output_dir / f"{meta['nombre']}_alga.jpg"
        cv2.imwrite(str(output_path), alga_final)
        exitosos += 1
        
        del alga_temp
        del alga_final
        
        if i % 50 == 0:
            print(f"  [{i}/{len(metadatos)}] Procesadas...")
    
    # Limpiar temporales
    print(f"\n→ Limpiando temporales...")
    for meta in metadatos:
        meta['temp_path'].unlink(missing_ok=True)
    temp_dir.rmdir()
    
    print(f"\n✓ Completado: {exitosos} algas guardadas ({tamaño_max}x{tamaño_max})")
    if errores > 0:
        print(f"⚠ {errores} imágenes con errores")


if __name__ == '__main__':
    main()
