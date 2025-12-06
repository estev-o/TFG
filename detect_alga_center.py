#!/usr/bin/env python3
"""
Detección de alga basada en posición central y bordes.
- Detecta bordes con Canny
- Busca desde el centro de la imagen hacia afuera
- El primer borde cercano al centro es el alga
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

def detectar_numeros_rojos(imagen_bgr):
    """Detecta números rojos usando análisis de color HSV."""
    # Convertir a HSV
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    
    # Rango para rojo (el rojo está en dos rangos en HSV: 0-10 y 170-180)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Crear máscaras para ambos rangos
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mascara_rojo = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Dilatar para unir fragmentos de números
    kernel = np.ones((5, 5), np.uint8)
    mascara_rojo = cv2.dilate(mascara_rojo, kernel, iterations=2)
    
    return mascara_rojo


def calcular_score_alga(contorno, imagen_bgr, gray, centro_x, centro_y):
    """Calcula un score para determinar si el contorno es un alga.
    Score alto = más probable que sea alga.
    Sistema: Distancia centro + forma + tamaño
    """
    x, y, w, h = cv2.boundingRect(contorno)
    area = cv2.contourArea(contorno)
    
    score = 0.0
    detalles = []  # Para debug
    
    # 1. Distancia al centro (más cerca = mejor) - PESO PRINCIPAL
    distancia_min = float('inf')
    for punto in contorno:
        px, py = punto[0]
        dist = np.sqrt((px - centro_x)**2 + (py - centro_y)**2)
        if dist < distancia_min:
            distancia_min = dist
    # Normalizar distancia (máximo 150 puntos si está en el centro)
    score_distancia = max(0, 150 - (distancia_min / 8))
    score += score_distancia
    detalles.append(f"dist={score_distancia:.1f}")
    
    # 2. Calculamos formas
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    extent = area / (w * h) if (w * h) > 0 else 0
    perimetro = cv2.arcLength(contorno, True)
    compacidad = (4 * np.pi * area) / (perimetro * perimetro) if perimetro > 0 else 0
    
    # 3. FORMA: Algas suelen ser más compactas y con área moderada-grande
    if 1.5 <= aspect_ratio <= 6:
        score += 50
        detalles.append(f"AR_ok=+50(AR={aspect_ratio:.2f})")
    elif aspect_ratio > 10:
        score -= 100  # Muy alargado = regla
        detalles.append(f"AR_regla=-100(AR={aspect_ratio:.2f})")
    
    # Compacidad
    if compacidad > 0.05:  # Algas más compactas
        score += 30
        detalles.append(f"compacto=+30(c={compacidad:.3f})")
    
    # Extent
    if extent > 0.3:  # Llena bien su bbox
        score += 20
        detalles.append(f"extent=+20(e={extent:.2f})")
    
    # 4. TAMAÑO: Áreas grandes son más probables de ser alga
    if area > 50000:
        score += 80
        detalles.append(f"area_muy_grande=+80(a={area:.0f})")
    elif area > 20000:
        score += 60
        detalles.append(f"area_grande=+60(a={area:.0f})")
    elif area > 5000:
        score += 30
        detalles.append(f"area_media=+30(a={area:.0f})")
    elif area < 1000:
        score -= 30 
        detalles.append(f"area_pequeña=-30(a={area:.0f})")
    
    return score, detalles





def detectar_alga_desde_centro(imagen, debug=False, debug_dir=None):
    """Detecta el alga buscando bordes cercanos al centro."""
    
    h, w = imagen.shape[:2]
    centro_x, centro_y = w // 2, h // 2
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordes con Canny
    edges = cv2.Canny(blurred, 30, 100)
    
    # Detectar números rojos y eliminarlos de los bordes
    mascara_numeros = detectar_numeros_rojos(imagen)
    edges_sin_numeros = cv2.bitwise_and(edges, cv2.bitwise_not(mascara_numeros))
    
    # Dilatar bordes ligeramente para conectar fragmentos
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges_sin_numeros, kernel, iterations=1)
    
    # DEBUG: Guardar máscaras intermedias (solo 3 imágenes)
    if debug and debug_dir:
        blurred_dbg = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_canny = cv2.Canny(blurred_dbg, 30, 100)
        
        cv2.imwrite(str(debug_dir / '1_canny.jpg'), edges_canny)
        cv2.imwrite(str(debug_dir / '2_mascara_numeros.jpg'), mascara_numeros)
        cv2.imwrite(str(debug_dir / '3_edges_finales.jpg'), edges)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return None, None
    
    # Calcular score de cada contorno
    scores = []
    info_debug = []  # Para archivo de texto debug
    
    for idx, contorno in enumerate(contornos):
        area = cv2.contourArea(contorno)
        razon_filtrado = None
        
        # Filtrar contornos muy pequeños (ruido)
        if area < 1000:
            if debug:
                info_debug.append(f"Contorno {idx}: área_pequeña ({area:.0f}px)")
            continue
        
        # Eliminamos según aspect ratio (reglas)
        # CON ESTO, ELIMINAMOS LAS REGLAS HORIZONTALES Y VERTICALES
        x, y, w_box, h_box = cv2.boundingRect(contorno)
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        
        # Reglas horizontales: AR > 3.0
        if aspect_ratio > 3.0:
            if debug:
                info_debug.append(f"Contorno {idx} (área={area:.0f}, AR={aspect_ratio:.1f}): regla_horizontal_AR>3")
            continue
        
        # Reglas verticales: AR < 0.33
        if aspect_ratio < 0.33:
            if debug:
                info_debug.append(f"Contorno {idx} (área={area:.0f}, AR={aspect_ratio:.2f}): regla_vertical_AR<0.33")
            continue
        
        # Calcular score del contorno
        score, detalles = calcular_score_alga(contorno, imagen, gray, centro_x, centro_y)
        
        if debug:
            detalles_str = ", ".join(detalles)
            info_debug.append(f"Contorno {idx} (área={area:.0f}, AR={aspect_ratio:.2f}): score={score:.1f} [{detalles_str}]")
        
        scores.append((score, contorno, area))
    
    # DEBUG: Guardar info en archivo de texto
    if debug and debug_dir:
        with open(debug_dir / 'debug_info.txt', 'w') as f:
            f.write(f"Total contornos: {len(contornos)}\n")
            f.write(f"Contornos evaluados: {len(scores)}\n")
            f.write(f"Contornos filtrados: {len(contornos) - len(scores)}\n\n")
            
            if scores:
                scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
                f.write("TOP 10 CONTORNOS (por score):\n")
                f.write("=" * 60 + "\n")
                for i, (score, _, area) in enumerate(scores_sorted[:10]):
                    f.write(f"  #{i+1}: Score={score:.1f}, Área={area:.0f}\n")
                f.write("\n")
            
            f.write("DETALLE DE FILTRADO:\n")
            for line in info_debug:
                f.write(f"  {line}\n")
    
    if not scores:
        return None, None
    
    # Ordenar por score (mayor score = más probable que sea alga)
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # El contorno con mayor score es el alga
    _, alga_contorno, _ = scores[0]
    
    # Obtener bounding box
    x, y, w_box, h_box = cv2.boundingRect(alga_contorno)
    
    return alga_contorno, (x, y, w_box, h_box)


def procesar_imagen(imagen_path, output_dir, debug=False):
    """Detectar alga y guardar imagen con bbox."""
    
    # Leer imagen
    imagen = cv2.imread(str(imagen_path))
    if imagen is None:
        return None, "Error al leer imagen"
    
    # Crear directorio debug si es necesario
    debug_dir = None
    if debug:
        debug_dir = output_dir / f"{imagen_path.stem}_debug"
        debug_dir.mkdir(exist_ok=True)
    
    # Detectar alga
    contorno, bbox = detectar_alga_desde_centro(imagen, debug=debug, debug_dir=debug_dir)
    
    if bbox is None:
        return None, "No se detectó alga"
    
    # Calcular info del alga
    x, y, w, h = bbox
    centro_x, centro_y = imagen.shape[1] // 2, imagen.shape[0] // 2
    alga_centro_x = x + w // 2
    alga_centro_y = y + h // 2
    distancia = np.sqrt((alga_centro_x - centro_x)**2 + (alga_centro_y - centro_y)**2)
    
    # Solo en modo debug: guardar imagen con bounding box
    if debug and debug_dir:
        imagen_result = imagen.copy()
        
        # Añadir padding 10%
        padding = 0.1
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(imagen.shape[1], x + w + pad_x)
        y2 = min(imagen.shape[0], y + h + pad_y)
        
        # Dibujar bbox con padding
        cv2.rectangle(imagen_result, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        # Dibujar centro de imagen
        cv2.circle(imagen_result, (centro_x, centro_y), 10, (0, 0, 255), -1)
        
        # Añadir info
        cv2.putText(imagen_result, f"Dist al centro: {distancia:.0f}px", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imwrite(str(debug_dir / '4_alga_detectada.jpg'), imagen_result)
    
    # Retornar info del bbox (sin guardar imagen en modo normal)
    return f"Alga detectada: bbox=({x},{y},{w},{h}), dist_centro={distancia:.0f}px", None


def main():
    parser = argparse.ArgumentParser(description='Detectar alga desde centro con Canny')
    parser.add_argument('--input_dir', type=str, 
                       default='dataset/Kelps_database_photos/Photos_kelps_database',
                       help='Directorio con imágenes')
    parser.add_argument('--output_dir', type=str, default='out',
                       help='Directorio de salida')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Número de imágenes a procesar')
    parser.add_argument('--debug', action='store_true',
                       help='Activar modo debug (guarda máscaras intermedias)')
    
    args = parser.parse_args()
    
    # Configurar rutas
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"ERROR: No existe el directorio: {input_dir}")
        sys.exit(1)
    
    # Buscar imágenes
    patterns = ['*.jfif', '*.jpg', '*.jpeg', '*.png']
    imagenes = []
    for pattern in patterns:
        imagenes.extend(sorted(input_dir.glob(pattern)))
    
    if not imagenes:
        print(f"ERROR: No se encontraron imágenes")
        sys.exit(1)
    
    imagenes = imagenes[:args.num_samples]
    
    print(f"Procesando {len(imagenes)} imágenes...")
    print(f"Salida: {output_dir}/\n")
    
    # Procesar
    exitosos = 0
    errores = []
    
    for i, img_path in enumerate(imagenes, 1):
        print(f"[{i}/{len(imagenes)}] {img_path.name}... ", end='', flush=True)
        
        output_path, error = procesar_imagen(img_path, output_dir, debug=args.debug)
        
        if error:
            print(f"{error}")
            errores.append({'file': img_path.name, 'error': error})
        else:
            print(f"")
            exitosos += 1


if __name__ == '__main__':
    main()
