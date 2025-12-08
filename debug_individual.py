#!/usr/bin/env python3
"""
Debug individual de una imagen específica.
Muestra todo el proceso de detección paso a paso.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# Importar funciones del script principal
from recortar_algas import detectar_numeros_rojos, calcular_score_alga, detectar_alga_desde_centro, procesar_imagen


def debug_imagen_completo(imagen_path, output_dir):
    """Debug completo de una imagen mostrando todos los pasos."""
    
    print(f"\n{'='*70}")
    print(f"DEBUG INDIVIDUAL: {imagen_path.name}")
    print(f"{'='*70}\n")
    
    # Leer imagen
    imagen = cv2.imread(str(imagen_path))
    if imagen is None:
        print(f" ERROR: No se pudo leer la imagen")
        return
    
    h, w = imagen.shape[:2]
    print(f"Dimensiones: {w}x{h}")
    print(f"Centro imagen: ({w//2}, {h//2})\n")
    
    # Crear directorio debug
    debug_dir = output_dir / f"{imagen_path.stem}_debug"
    debug_dir.mkdir(exist_ok=True, parents=True)
    print(f"Guardando debug en: {debug_dir}/\n")
    
    # Guardar imagen original
    cv2.imwrite(str(debug_dir / '0_original.jpg'), imagen)
    print(f"Imagen original guardada\n")
    
    # Paso 1: Canny
    print("PASO 1: Detección de bordes (Canny)")
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_canny = cv2.Canny(blurred, 30, 100)
    cv2.imwrite(str(debug_dir / '1_canny.jpg'), edges_canny)
    print(f"    Bordes detectados con Canny(30, 100)")
    
    # Paso 2: Detectar números rojos
    print("\nPASO 2: Detección de números rojos")
    mascara_numeros = detectar_numeros_rojos(imagen)
    cv2.imwrite(str(debug_dir / '2_mascara_numeros.jpg'), mascara_numeros)
    num_pixeles_rojos = np.count_nonzero(mascara_numeros)
    print(f"    Píxeles rojos detectados: {num_pixeles_rojos}")
    
    # Paso 3: Eliminar números de bordes
    print("\nPASO 3: Eliminar números de los bordes")
    edges_sin_numeros = cv2.bitwise_and(edges_canny, cv2.bitwise_not(mascara_numeros))
    kernel = np.ones((3, 3), np.uint8)
    edges_finales = cv2.dilate(edges_sin_numeros, kernel, iterations=1)
    cv2.imwrite(str(debug_dir / '3_edges_finales.jpg'), edges_finales)
    print(f"    Bordes limpiados y dilatados")
    
    # Paso 4: Encontrar contornos
    print("\nPASO 4: Búsqueda de contornos")
    contornos, _ = cv2.findContours(edges_finales, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Total contornos encontrados: {len(contornos)}")
    
    if not contornos:
        print("    No se encontraron contornos")
        return
    
    # Paso 5: Filtrado y scoring
    print("\n PASO 5: Filtrado y scoring de contornos")
    print(f"   {'─'*66}")
    
    centro_x, centro_y = w // 2, h // 2
    scores = []
    contornos_filtrados = 0
    
    for idx, contorno in enumerate(contornos):
        area = cv2.contourArea(contorno)
        
        # Filtro: área mínima
        if area < 1000:
            contornos_filtrados += 1
            continue
        
        # Filtro: aspect ratio (reglas)
        x, y, w_box, h_box = cv2.boundingRect(contorno)
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        
        if aspect_ratio > 3.0:
            contornos_filtrados += 1
            print(f"   [Contorno {idx:2d}] Filtrado: regla horizontal (AR={aspect_ratio:.2f})")
            continue
        
        if aspect_ratio < 0.33:
            contornos_filtrados += 1
            print(f"   [Contorno {idx:2d}] Filtrado: regla vertical (AR={aspect_ratio:.2f})")
            continue
        
        # Calcular score
        score, detalles = calcular_score_alga(contorno, imagen, gray, centro_x, centro_y)
        scores.append((score, contorno, area, aspect_ratio, detalles))
    
    print(f"\n   Contornos filtrados: {contornos_filtrados}")
    print(f"   Contornos evaluados: {len(scores)}\n")
    
    if not scores:
        print("    No quedan contornos válidos después del filtrado")
        return
    
    # Ordenar por score
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Mostrar top 5
    print("   TOP 5 CONTORNOS:")
    print(f"   {'─'*66}")
    for i, (score, _, area, ar, detalles) in enumerate(scores[:5], 1):
        detalles_str = ", ".join(detalles)
        print(f"   #{i}. Score={score:6.1f} | Área={area:8.0f} | AR={ar:4.2f}")
        print(f"       [{detalles_str}]")
    
    # Paso 6: Visualizar resultado
    print(f"\n PASO 6: Alga seleccionada (mayor score)")
    mejor_score, mejor_contorno, mejor_area, mejor_ar, mejor_detalles = scores[0]
    print(f"   Score: {mejor_score:.1f}")
    print(f"   Área: {mejor_area:.0f} px²")
    print(f"   Aspect Ratio: {mejor_ar:.2f}")
    
    # Dibujar resultado
    imagen_result = imagen.copy()
    
    # Dibujar todos los contornos en gris
    for score, contorno, _, _, _ in scores[1:]:
        cv2.drawContours(imagen_result, [contorno], -1, (128, 128, 128), 2)
    
    # Dibujar el mejor en verde
    cv2.drawContours(imagen_result, [mejor_contorno], -1, (0, 255, 0), 5)
    
    # Bbox del alga seleccionada
    x, y, w_box, h_box = cv2.boundingRect(mejor_contorno)
    cv2.rectangle(imagen_result, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)
    
    # Centro de imagen
    cv2.circle(imagen_result, (centro_x, centro_y), 10, (0, 0, 255), -1)
    cv2.circle(imagen_result, (centro_x, centro_y), 15, (0, 0, 255), 2)
    
    # Info en imagen
    cv2.putText(imagen_result, f"Score: {mejor_score:.1f}", 
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(imagen_result, f"Area: {mejor_area:.0f}", 
               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(imagen_result, f"AR: {mejor_ar:.2f}", 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    cv2.imwrite(str(debug_dir / '4_resultado_final.jpg'), imagen_result)
    
    # Paso 7: Reorte
    print(f"\n PASO 7: Recortando alga")
    alga_recortada, lado_cuadrado, nombre, error = procesar_imagen(
        imagen_path, 
        output_dir.parent,  # Guardar en out/ directamente
        debug=False
    )
    
    if error:
        print(f"    Error en recorte: {error}")
    else:
        print(f"    Recorte guardado: {lado_cuadrado}x{lado_cuadrado}")
        
        # Guardar también en debug
        cv2.imwrite(str(debug_dir / '5_alga_recortada.jpg'), alga_recortada)
    
    print(f"\n{'='*70}")
    print(f" DEBUG COMPLETADO")
    print(f"Archivos generados en: {debug_dir}/")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Debug individual de una imagen específica')
    parser.add_argument('codigo', type=str, 
                       help='Código de la imagen (ej: GI405, GI406, etc.)')
    parser.add_argument('--input_dir', type=str, 
                       default='dataset/Kelps_database_photos/Photos_kelps_database',
                       help='Directorio con imágenes')
    parser.add_argument('--output_dir', type=str, default='out',
                       help='Directorio de salida para debug')
    
    args = parser.parse_args()
    
    # Buscar la imagen por código
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f" ERROR: No existe el directorio: {input_dir}")
        sys.exit(1)
    
    # Buscar archivo con ese código
    patterns = ['*.jfif', '*.JFIF', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.heic', '*.HEIC']
    imagen_encontrada = None
    
    for pattern in patterns:
        for img_path in input_dir.glob(pattern):
            if img_path.stem == args.codigo:
                imagen_encontrada = img_path
                break
        if imagen_encontrada:
            break
    
    if not imagen_encontrada:
        print(f" ERROR: No se encontró imagen con código '{args.codigo}'")
        print(f"\nBuscar en: {input_dir}/")
        print(f"Patrones: {', '.join(patterns)}")
        sys.exit(1)
    
    # Ejecutar debug
    debug_imagen_completo(imagen_encontrada, output_dir)


if __name__ == '__main__':
    main()
