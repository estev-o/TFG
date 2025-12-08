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

def detectar_texto_coloreado(imagen_bgr):
    """Detecta texto verde y rojo (números/letras de dataset) usando HSV.
    Verde: HSL(174, 98, 24) = rgb(1, 119, 108)
    Rojo: HSL(354, 50, 42) = rgb(160, 53, 64)
    """
    # Convertir a HSV
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    
    # Rango para VERDE - MÁS ESTRICTO para evitar algas verdosas
    # Solo verde brillante/saturado como en letras
    # H=87 (174°/2), S muy alta (>150), V medio-bajo
    lower_verde = np.array([82, 150, 40])   # Más estricto en H y S
    upper_verde = np.array([92, 255, 130])  # Rango H más estrecho
    
    # Rango para ROJO - MÁS ESTRICTO
    # Solo rojo saturado como en letras
    # S muy alta (>120) para evitar marrones/oscuros
    lower_rojo1 = np.array([0, 120, 100])   # S más alta, V más alto
    upper_rojo1 = np.array([8, 255, 255])   # Rango H más estrecho
    lower_rojo2 = np.array([172, 120, 100]) # S más alta, V más alto
    upper_rojo2 = np.array([180, 255, 255])
    
    # Crear máscaras
    mascara_verde = cv2.inRange(hsv, lower_verde, upper_verde)
    mascara_rojo1 = cv2.inRange(hsv, lower_rojo1, upper_rojo1)
    mascara_rojo2 = cv2.inRange(hsv, lower_rojo2, upper_rojo2)
    
    # Combinar todas las máscaras
    mascara_texto = cv2.bitwise_or(mascara_verde, mascara_rojo1)
    mascara_texto = cv2.bitwise_or(mascara_texto, mascara_rojo2)
    
    # Dilatar ligeramente para capturar texto completo
    kernel = np.ones((3, 3), np.uint8)
    mascara_texto = cv2.dilate(mascara_texto, kernel, iterations=2)
    
    return mascara_texto


def tiene_fondo_dorado(contorno, imagen_bgr, umbral_porcentaje=30):
    """Detecta si el fondo del bbox (píxeles fuera del contorno) es dorado/marrón de regla.
    
    Args:
        contorno: El contorno a analizar
        imagen_bgr: Imagen original en BGR
        umbral_porcentaje: % de píxeles dorados necesarios para considerar fondo dorado
    
    Returns:
        True si el fondo es mayormente dorado (está en regla), False si no
    """
    # Obtener bbox
    x, y, w, h = cv2.boundingRect(contorno)
    
    # Asegurar que bbox está dentro de los límites
    x = max(0, x)
    y = max(0, y)
    x2 = min(imagen_bgr.shape[1], x + w)
    y2 = min(imagen_bgr.shape[0], y + h)
    
    # Recortar región bbox
    roi = imagen_bgr[y:y2, x:x2]
    
    if roi.size == 0:
        return False
    
    # Crear máscara del contorno en coordenadas locales del bbox
    mascara_contorno = np.zeros((y2-y, x2-x), dtype=np.uint8)
    contorno_local = contorno - np.array([x, y])  # Trasladar a coordenadas locales
    cv2.drawContours(mascara_contorno, [contorno_local], -1, 255, -1)
    
    # Máscara del fondo (invertir máscara del contorno)
    mascara_fondo = cv2.bitwise_not(mascara_contorno)
    
    # Contar píxeles de fondo
    pixeles_fondo = np.sum(mascara_fondo == 255)
    
    if pixeles_fondo < 10:  # Muy pocos píxeles de fondo para analizar
        return False
    
    # Convertir a HSV para detectar colores dorados/marrones
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Rangos para dorado/marrón de regla (basado en #f9d572, #d7ab51, #eabe60)
    # Conversión correcta OpenCV (H en 0-179):
    # #f9d572 → HSV(22, 138, 249)
    # #d7ab51 → HSV(20, 159, 215)
    # #eabe60 → HSV(20, 150, 234)
    # Rango amplio: H[18-25], S[130-165], V[200-255]
    lower_dorado = np.array([18, 130, 200])
    upper_dorado = np.array([25, 165, 255])
    
    # Detectar píxeles dorados
    mascara_dorado = cv2.inRange(roi_hsv, lower_dorado, upper_dorado)
    
    # Aplicar máscara de fondo: solo contar píxeles dorados que estén en el fondo
    mascara_dorado_fondo = cv2.bitwise_and(mascara_dorado, mascara_fondo)
    
    # Contar píxeles dorados en el fondo
    pixeles_dorados = np.sum(mascara_dorado_fondo == 255)
    
    # Porcentaje de fondo que es dorado
    porcentaje_dorado = (pixeles_dorados / pixeles_fondo) * 100
    
    return porcentaje_dorado > umbral_porcentaje


def calcular_score_alga(contorno, imagen_bgr, gray, centro_x, centro_y, img_width, img_height):
    """Calcula un score para determinar si el contorno es un alga.
    Score alto = más probable que sea alga.
    Sistema: Distancia centro + forma + tamaño + filtros anti-texto
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
    
    # Solidez (área del contorno / área del convex hull)
    hull = cv2.convexHull(contorno)
    hull_area = cv2.contourArea(hull)
    solidez = area / hull_area if hull_area > 0 else 0
    
    # 3. FILTRO ANTI-TEXTO: Detectar números/letras/logos
    # Los textos suelen estar en los bordes de la imagen
    margen_texto = 0.10  # 10% de margen en cada borde
    margen_x = int(img_width * margen_texto)
    margen_y = int(img_height * margen_texto)
    
    en_borde = (x < margen_x or 
                y < margen_y or 
                x + w > img_width - margen_x or 
                y + h > img_height - margen_y)
    
    if en_borde and area < 10000:
        score -= 80
        detalles.append(f"texto_borde=-80")
    
    # Textos tienen baja solidez (muchos "huecos" como Q, O, 0, 8)
    if solidez < 0.5 and area < 15000:
        score -= 60
        detalles.append(f"baja_solidez=-60(s={solidez:.2f})")
    
    # Letras circulares como Q, O tienen solidez alta pero área muy pequeña
    if solidez > 0.7 and area < 3000:
        score -= 70
        detalles.append(f"letra_circular=-70(s={solidez:.2f},a={area:.0f})")
    
    # Números/letras: áreas muy pequeñas cerca de reglas
    if area < 2000 and (y < img_height * 0.15 or y + h > img_height * 0.85 or 
                        x < img_width * 0.15 or x + w > img_width * 0.85):
        score -= 100
        detalles.append(f"numero_peq_borde=-100(a={area:.0f})")
    
    # Perímetro excesivo relativo al área = fragmentado/texto
    perimetro_area_ratio = perimetro / area if area > 0 else 0
    if perimetro_area_ratio > 0.5 and area < 20000:
        score -= 50
        detalles.append(f"perimetro_alto=-50(p/a={perimetro_area_ratio:.3f})")
    
    # FILTRO ANTI-REGLA: Detectar elementos alineados con reglas
    # Elementos en las reglas suelen estar en líneas verticales u horizontales
    # Checkeamos si está muy cerca de los bordes laterales (reglas verticales)
    # o superior/inferior (reglas horizontales)
    margen_regla = 0.15  # 15% de margen para reglas
    margen_regla_x = int(img_width * margen_regla)
    margen_regla_y = int(img_height * margen_regla)
    
    # Cerca de bordes laterales (reglas verticales) + área pequeña = número/logo en regla
    cerca_lateral = (x < margen_regla_x or x + w > img_width - margen_regla_x)
    cerca_horizontal = (y < margen_regla_y or y + h > img_height - margen_regla_y)
    
    if (cerca_lateral or cerca_horizontal) and area < 8000:
        score -= 90
        detalles.append(f"elemento_regla=-90")
    
    # 4. FORMA: Algas suelen ser más compactas
    # Compacidad
    if compacidad > 0.05:  # Algas más compactas
        score += 30
        detalles.append(f"compacto=+30(c={compacidad:.3f})")
    
    # Extent
    if extent > 0.3:  # Llena bien su bbox
        score += 20
        detalles.append(f"extent=+20(e={extent:.2f})")
    
    # Solidez alta = forma compacta (bueno para algas)
    if solidez > 0.8:
        score += 25
        detalles.append(f"solidez_alta=+25(s={solidez:.2f})")
    
    # 5. TAMAÑO: Áreas grandes son más probables de ser alga
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
    
    # Detectar texto coloreado (verde y rojo)
    mascara_texto = detectar_texto_coloreado(imagen)
    
    # Dilatar bordes ligeramente para conectar fragmentos
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # DEBUG: Guardar máscaras intermedias
    if debug and debug_dir:
        blurred_dbg = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_canny = cv2.Canny(blurred_dbg, 30, 100)
        
        cv2.imwrite(str(debug_dir / '1_canny.jpg'), edges_canny)
        cv2.imwrite(str(debug_dir / '2_mascara_texto.jpg'), mascara_texto)
        cv2.imwrite(str(debug_dir / '3_edges_finales.jpg'), edges)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return None, None
    
    # PASO 1: Detectar reglas por aspect ratio
    reglas = []
    contornos_no_reglas = []
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < 1000:
            continue
            
        x, y, w_box, h_box = cv2.boundingRect(contorno)
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        
        # Reglas: AR > 3.0 (horizontal) o AR < 0.33 (vertical)
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            reglas.append(contorno)
        else:
            contornos_no_reglas.append(contorno)
    
    # PASO 2: Filtrar contornos que están DENTRO de alguna regla o son TEXTO COLOREADO
    contornos_filtrados = []
    info_debug = []  # Para archivo de texto debug
    
    # Área mínima absoluta: algas reales son mucho más grandes que números/letras
    AREA_MINIMA_ALGA = 5000  # píxeles cuadrados
    
    for idx, contorno in enumerate(contornos_no_reglas):
        area = cv2.contourArea(contorno)
        
        # FILTRO 1: Área mínima - eliminar números/letras/logos pequeños
        if area < AREA_MINIMA_ALGA:
            if debug:
                info_debug.append(f"Contorno {idx} (área={area:.0f}): area_muy_pequeña (min={AREA_MINIMA_ALGA})")
            continue
        
        # Verificar si este contorno está dentro o MUY CERCA de alguna regla
        dentro_o_cerca_regla = False
        for regla in reglas:
            # Obtener un punto del contorno para testear (debe ser float)
            punto_test = (float(contorno[0][0][0]), float(contorno[0][0][1]))
            
            # cv2.pointPolygonTest con measureDist=True devuelve distancia con signo
            # >0 = dentro, <0 = fuera (distancia negativa)
            distancia = cv2.pointPolygonTest(regla, punto_test, measureDist=True)
            
            # Si está dentro (distancia >= 0) o MUY CERCA (< 30 píxeles de distancia)
            if distancia >= 0:  # Está dentro o en el borde
                dentro_o_cerca_regla = True
                if debug:
                    info_debug.append(f"Contorno {idx} (área={area:.0f}): dentro_de_regla")
                break
            elif distancia > -30:  # Fuera pero a menos de 30px
                dentro_o_cerca_regla = True
                if debug:
                    info_debug.append(f"Contorno {idx} (área={area:.0f}): cerca_de_regla (dist={abs(distancia):.1f}px)")
                break
        
        if dentro_o_cerca_regla:
            continue
        
        # Verificar si el FONDO del bbox es dorado (está en regla)
        es_fondo_dorado = tiene_fondo_dorado(contorno, imagen)
        if debug:
            x_dbg, y_dbg, w_dbg, h_dbg = cv2.boundingRect(contorno)
            info_debug.append(f"Contorno {idx} (área={area:.0f}, pos={x_dbg},{y_dbg}): fondo_dorado={'SÍ' if es_fondo_dorado else 'NO'}")
        
        if es_fondo_dorado:
            continue
        
        # Verificar si este contorno es TEXTO COLOREADO (verde/rojo)
        # Crear máscara del contorno y verificar overlap con máscara de texto
        mask_contorno = np.zeros(imagen.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_contorno, [contorno], -1, 255, -1)
        
        # Calcular overlap entre contorno y texto coloreado
        overlap = cv2.bitwise_and(mask_contorno, mascara_texto)
        overlap_area = np.count_nonzero(overlap)
        overlap_ratio = overlap_area / area if area > 0 else 0
        
        # Si más del 30% del contorno es texto coloreado, descartarlo
        if overlap_ratio > 0.3:
            if debug:
                info_debug.append(f"Contorno {idx} (área={area:.0f}): texto_coloreado (overlap={overlap_ratio:.1%})")
            continue
        
        contornos_filtrados.append((idx, contorno, area))
    
    # PASO 3: Calcular score de contornos válidos
    scores = []
    
    for idx, contorno, area in contornos_filtrados:
        x, y, w_box, h_box = cv2.boundingRect(contorno)
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        
        # Calcular score del contorno
        score, detalles = calcular_score_alga(contorno, imagen, gray, centro_x, centro_y, w, h)
        
        if debug:
            detalles_str = ", ".join(detalles)
            info_debug.append(f"Contorno {idx} (área={area:.0f}, AR={aspect_ratio:.2f}): score={score:.1f} [{detalles_str}]")
        
        scores.append((score, contorno, area))
    
    # DEBUG: Guardar info en archivo de texto
    if debug and debug_dir:
        with open(debug_dir / 'debug_info.txt', 'w') as f:
            f.write(f"Total contornos encontrados: {len(contornos)}\n")
            f.write(f"Reglas detectadas: {len(reglas)}\n")
            f.write(f"Contornos después de filtros: {len(contornos_filtrados)}\n")
            f.write(f"Contornos evaluados con score: {len(scores)}\n\n")
            
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
    """Detectar alga, recortar y guardar con aspect ratio cuadrado (1:1) para IA."""
    
    # Leer imagen
    imagen = cv2.imread(str(imagen_path))
    if imagen is None:
        return None, None, None, None
    
    # Crear directorio debug si es necesario
    debug_dir = None
    if debug:
        debug_dir = output_dir / f"{imagen_path.stem}_debug"
        debug_dir.mkdir(exist_ok=True)
    
    # Detectar alga
    contorno, bbox = detectar_alga_desde_centro(imagen, debug=debug, debug_dir=debug_dir)
    
    if bbox is None:
        return None, None, None, None
    
    # Calcular info del alga
    x, y, w, h = bbox
    centro_x, centro_y = imagen.shape[1] // 2, imagen.shape[0] // 2
    alga_centro_x = x + w // 2
    alga_centro_y = y + h // 2
    distancia = np.sqrt((alga_centro_x - centro_x)**2 + (alga_centro_y - centro_y)**2)
    
    # Recortar EXACTAMENTE el bounding box (sin padding ni cuadrado)
    alga_recortada = imagen[y:y+h, x:x+w]
    
    # Coordenadas para debug visual
    x1_cuadrado = x
    y1_cuadrado = y
    x2_cuadrado = x + w
    y2_cuadrado = y + h
    
    # Solo en modo debug: guardar imagen con bounding box
    if debug and debug_dir:
        imagen_result = imagen.copy()
        
        # Dibujar bbox cuadrado
        cv2.rectangle(imagen_result, (x1_cuadrado, y1_cuadrado), 
                     (x2_cuadrado, y2_cuadrado), (0, 255, 0), 5)
        
        # Dibujar centro de imagen
        cv2.circle(imagen_result, (centro_x, centro_y), 10, (0, 0, 255), -1)
        
        # Añadir info
        cv2.putText(imagen_result, f"Dist al centro: {distancia:.0f}px", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(imagen_result, f"Recorte: {alga_recortada.shape[1]}x{alga_recortada.shape[0]}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imwrite(str(debug_dir / '4_resultado_final.jpg'), imagen_result)
    
    # Retornar imagen recortada (rectangular, sin padding)
    nombre = imagen_path.stem
    
    return alga_recortada, w, h, nombre


def main():
    parser = argparse.ArgumentParser(description='Recortar algas con aspect ratio cuadrado para entrenamiento IA')
    parser.add_argument('--input_dir', type=str, 
                       default='dataset/Kelps_database_photos/Photos_kelps_database',
                       help='Directorio con imágenes')
    parser.add_argument('--output_dir', type=str, default='out',
                       help='Directorio de salida')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Número de imágenes a procesar (por defecto: todas)')
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
    
    # Buscar imágenes (case-insensitive para mayúsculas/minúsculas)
    patterns = ['*.jfif', '*.JFIF', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.heic', '*.HEIC']
    imagenes = []
    for pattern in patterns:
        imagenes.extend(sorted(input_dir.glob(pattern)))
    
    if not imagenes:
        print(f"ERROR: No se encontraron imágenes")
        sys.exit(1)
    
    # Limitar solo si num_samples está especificado
    if args.num_samples:
        imagenes = imagenes[:args.num_samples]
    
    print(f"Procesando {len(imagenes)} imágenes...")
    print(f"Salida: {output_dir}/\n")
    
    # Crear directorio temporal
    temp_dir = output_dir / "temp_crops"
    temp_dir.mkdir(exist_ok=True)
    
    # Primera pasada: detectar, recortar, guardar temporales y trackear tamaños
    print("→ Fase 1: Detectando algas, recortando y guardando temporales...")
    metadatos = []  # Solo guardamos (nombre, tamaño) en memoria
    tamaño_max = 0
    errores = 0
    
    for i, img_path in enumerate(imagenes, 1):
        print(f"  [{i}/{len(imagenes)}] {img_path.name}... ", end='', flush=True)
        
        try:
            alga_recortada, w, h, nombre = procesar_imagen(img_path, output_dir, debug=args.debug)
        except Exception as e:
            print(f"Error: {e}")
            errores += 1
            continue
        
        if alga_recortada is None:
            print(f"No se detectó alga")
            errores += 1
        else:
            # Guardar temporal en disco (libera memoria inmediatamente)
            temp_path = temp_dir / f"{nombre}_temp.jpg"
            cv2.imwrite(str(temp_path), alga_recortada)
            
            # Solo trackear metadatos (mínima memoria)
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
        print("\n✗ No se detectaron algas válidas en ninguna imagen")
        temp_dir.rmdir()
        sys.exit(1)
    
    print(f"\n→ Tamaño máximo detectado: {tamaño_max}x{tamaño_max}")
    print(f"→ Fase 2: Redimensionando {len(metadatos)} algas al tamaño máximo...\n")
    
    # Segunda pasada: leer temporales, hacer cuadrado con padding y redimensionar
    exitosos = 0
    for i, meta in enumerate(metadatos, 1):
        # Leer temporal (solo una imagen en memoria a la vez)
        alga_temp = cv2.imread(str(meta['temp_path']))
        
        if alga_temp is None:
            print(f"  ✗ Error leyendo temporal: {meta['nombre']}")
            continue
        
        h_temp, w_temp = alga_temp.shape[:2]
        lado_cuadrado = max(w_temp, h_temp)
        
        # Crear canvas cuadrado con fondo gris neutro
        alga_cuadrada = np.full((lado_cuadrado, lado_cuadrado, 3), 127, dtype=np.uint8)
        
        # Centrar alga en el canvas
        offset_y = (lado_cuadrado - h_temp) // 2
        offset_x = (lado_cuadrado - w_temp) // 2
        alga_cuadrada[offset_y:offset_y+h_temp, offset_x:offset_x+w_temp] = alga_temp
        
        # Redimensionar al tamaño máximo (todos iguales)
        alga_final = cv2.resize(
            alga_cuadrada, 
            (tamaño_max, tamaño_max), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Guardar final
        output_path = output_dir / f"{meta['nombre']}_alga.jpg"
        cv2.imwrite(str(output_path), alga_final)
        exitosos += 1
        
        # Liberar memoria de esta imagen
        del alga_temp
        del alga_final
        
        # Progress cada 50 imágenes
        if i % 50 == 0:
            print(f"  [{i}/{len(metadatos)}] Procesadas...")
    
    # Limpiar temporales
    print(f"\n→ Limpiando archivos temporales...")
    for meta in metadatos:
        meta['temp_path'].unlink(missing_ok=True)
    temp_dir.rmdir()
    
    print(f"\n✓ Recorte completado: {exitosos} algas guardadas ({tamaño_max}x{tamaño_max})")
    if errores > 0:
        print(f"⚠ {errores} imágenes con errores")


def debug_individual(codigo, input_dir='dataset/Kelps_database_photos/Photos_kelps_database', output_dir='out'):
    """Debug completo de una imagen individual mostrando todos los pasos."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🔍 DEBUG INDIVIDUAL: {codigo}")
    print(f"{'='*70}\n")
    
    # Buscar imagen por código
    if not input_dir.exists():
        print(f"✗ ERROR: No existe el directorio: {input_dir}")
        sys.exit(1)
    
    patterns = ['*.jfif', '*.JFIF', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.heic', '*.HEIC']
    imagen_path = None
    
    for pattern in patterns:
        for img_path in input_dir.glob(pattern):
            if img_path.stem == codigo:
                imagen_path = img_path
                break
        if imagen_path:
            break
    
    if not imagen_path:
        print(f"✗ ERROR: No se encontró imagen con código '{codigo}'")
        sys.exit(1)
    
    # Leer imagen
    imagen = cv2.imread(str(imagen_path))
    if imagen is None:
        print(f"✗ ERROR: No se pudo leer la imagen")
        sys.exit(1)
    
    h, w = imagen.shape[:2]
    print(f"📷 Dimensiones: {w}x{h}")
    print(f"📍 Centro imagen: ({w//2}, {h//2})\n")
    
    # Crear directorio debug
    debug_dir = output_dir / f"{codigo}_debug"
    debug_dir.mkdir(exist_ok=True, parents=True)
    print(f"💾 Guardando debug en: {debug_dir}/\n")
    
    # Guardar original
    cv2.imwrite(str(debug_dir / '0_original.jpg'), imagen)
    
    # Ejecutar detección con debug
    print("🔎 Ejecutando detección con debug completo...")
    contorno_alga, bbox_alga = detectar_alga_desde_centro(imagen, debug=True, debug_dir=debug_dir)
    
    if bbox_alga is None:
        print("\n✗ No se detectó alga válida")
        print(f"\n{'='*70}")
        print(f"📂 Archivos generados en: {debug_dir}/")
        print(f"{'='*70}\n")
        return
    
    # Leer debug_info.txt
    debug_info_path = debug_dir / 'debug_info.txt'
    if debug_info_path.exists():
        print(f"\n   📊 INFORMACIÓN DETALLADA DE FILTRADO:")
        print(f"   {'──'*33}")
        with open(debug_info_path, 'r') as f:
            for line in f:
                print(f"   {line.rstrip()}")
    
    # Visualización avanzada
    print(f"\n🎨 Generando visualización avanzada...")
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    mascara_texto = detectar_texto_coloreado(imagen)
    
    # Encontrar contornos para visualizar
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contornos_vis, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear imagen resultado
    imagen_result = imagen.copy()
    
    # Dibujar reglas en amarillo
    for contorno in contornos_vis:
        area = cv2.contourArea(contorno)
        if area < 1000:
            continue
        x_c, y_c, w_c, h_c = cv2.boundingRect(contorno)
        ar_c = w_c / h_c if h_c > 0 else 0
        if ar_c > 3.0 or ar_c < 0.33:
            cv2.drawContours(imagen_result, [contorno], -1, (0, 255, 255), 2)
    
    # Overlay de texto coloreado en magenta
    overlay = imagen_result.copy()
    overlay[mascara_texto > 0] = [255, 0, 255]
    imagen_result = cv2.addWeighted(imagen_result, 0.85, overlay, 0.15, 0)
    
    # Alga seleccionada en verde grueso
    x, y, w_box, h_box = bbox_alga
    cv2.drawContours(imagen_result, [contorno_alga], -1, (0, 255, 0), 8)
    
    # Centro de imagen en rojo
    centro_x, centro_y = w // 2, h // 2
    cv2.circle(imagen_result, (centro_x, centro_y), 20, (0, 0, 255), -1)
    
    # Leyenda
    y_text = 50
    cv2.putText(imagen_result, "LEYENDA:", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    y_text += 40
    cv2.putText(imagen_result, "- Verde: ALGA SELECCIONADA", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y_text += 35
    cv2.putText(imagen_result, "- Amarillo: Reglas detectadas", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y_text += 35
    cv2.putText(imagen_result, "- Magenta: Texto coloreado", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    y_text += 35
    cv2.putText(imagen_result, "- Rojo: Centro imagen", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Info del alga
    area_alga = cv2.contourArea(contorno_alga)
    ar_alga = w_box / h_box if h_box > 0 else 0
    y_text += 60
    cv2.putText(imagen_result, f"Alga: {w_box}x{h_box}, AR={ar_alga:.2f}", 
               (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # Top 3 scores
    if debug_info_path.exists():
        y_text += 50
        with open(debug_info_path, 'r') as f:
            lines = f.readlines()
            in_top = False
            count = 0
            for line in lines:
                if "TOP 10 CONTORNOS" in line:
                    in_top = True
                    continue
                if in_top and line.strip().startswith("#") and count < 3:
                    cv2.putText(imagen_result, line.strip()[:55], (10, y_text), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_text += 30
                    count += 1
    
    cv2.imwrite(str(debug_dir / '4_resultado_visual.jpg'), imagen_result)
    
    # Recortar alga
    print(f"\n✂️  Recortando alga...")
    alga_recortada = imagen[y:y+h_box, x:x+w_box]
    cv2.imwrite(str(debug_dir / '5_alga_recortada.jpg'), alga_recortada)
    print(f"   ✓ Recorte guardado: {w_box}x{h_box}")
    
    print(f"\n{'='*70}")
    print(f"✓ DEBUG COMPLETADO")
    print(f"Archivos generados en: {debug_dir}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Detectar si es modo debug individual o batch
    parser = argparse.ArgumentParser(description='Recortar algas con debug')
    parser.add_argument('codigo', nargs='?', type=str, 
                       help='Código de imagen para debug individual (ej: GI405)')
    parser.add_argument('--input_dir', type=str, 
                       default='dataset/Kelps_database_photos/Photos_kelps_database',
                       help='Directorio con imágenes')
    parser.add_argument('--output_dir', type=str, default='out',
                       help='Directorio de salida')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Número de imágenes a procesar')
    parser.add_argument('--debug', action='store_true',
                       help='Activar modo debug (guarda máscaras intermedias)')
    
    args = parser.parse_args()
    
    # Si se proporciona código, ejecutar debug individual
    if args.codigo:
        debug_individual(args.codigo, args.input_dir, args.output_dir)
    else:
        # Ejecutar batch con debug
        main()
