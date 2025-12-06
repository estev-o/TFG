#!/usr/bin/env python3
"""
Utilidades para debug y visualización de detección de algas.
"""

import cv2
import numpy as np
from pathlib import Path


def guardar_mascaras_debug(debug_dir, gray, mascara_regla, mascara_numeros, mascara_combined, edges_finales, imagen_original=None, rectangulos_regla=None):
    """Guarda máscaras intermedias del proceso de detección."""
    # Detectar bordes Canny original
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_canny = cv2.Canny(blurred, 30, 100)
    
    cv2.imwrite(str(debug_dir / 'debug_1_edges_canny.jpg'), edges_canny)
    cv2.imwrite(str(debug_dir / 'debug_2_mascara_regla.jpg'), mascara_regla)
    cv2.imwrite(str(debug_dir / 'debug_3_mascara_numeros.jpg'), mascara_numeros)
    cv2.imwrite(str(debug_dir / 'debug_4_mascara_combined.jpg'), mascara_combined)
    cv2.imwrite(str(debug_dir / 'debug_5_edges_finales.jpg'), edges_finales)
    
    # Visualizar rectángulos de regla detectados sobre la imagen original
    if imagen_original is not None and rectangulos_regla is not None:
        img_rectangulos = imagen_original.copy()
        for x, y, w, h in rectangulos_regla:
            cv2.rectangle(img_rectangulos, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img_rectangulos, f"Regla {w}x{h}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / 'debug_7_rectangulos_regla.jpg'), img_rectangulos)


def visualizar_contornos_numerados(debug_dir, imagen, contornos):
    """Visualiza todos los contornos con números."""
    img_contornos = imagen.copy()
    cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)
    
    for idx, cnt in enumerate(contornos):
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(img_contornos, str(idx), (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imwrite(str(debug_dir / 'debug_6_contornos_numerados.jpg'), img_contornos)


def visualizar_rectangulos_regla(debug_dir, imagen, rectangulos_regla):
    """Visualiza los rectángulos de regla detectados."""
    img_rectangulos = imagen.copy()
    
    if rectangulos_regla:
        for i, (x, y, w, h) in enumerate(rectangulos_regla):
            cv2.rectangle(img_rectangulos, (x, y), (x + w, y + h), (0, 0, 255), 4)
            # Añadir etiqueta con dimensiones
            cv2.putText(img_rectangulos, f"#{i+1}: {w}x{h} px, AR={w/h:.1f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Añadir texto con cantidad
    cv2.putText(img_rectangulos, f"Rectangulos regla: {len(rectangulos_regla) if rectangulos_regla else 0}", 
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    cv2.imwrite(str(debug_dir / 'debug_7_rectangulos_regla.jpg'), img_rectangulos)


def guardar_info_filtrado(debug_dir, contornos, scores, filtros_aplicados, rectangulos_regla=None):
    """Guarda información detallada del proceso de filtrado."""
    debug_info = f"Total contornos: {len(contornos)}\n"
    debug_info += f"Contornos evaluados: {len(scores)}\n"
    debug_info += f"Contornos filtrados (hard): {len(contornos) - len(scores)}\n"
    if rectangulos_regla is not None:
        debug_info += f"Rectángulos de regla detectados: {len(rectangulos_regla)}\n"
        
        # Listar coordenadas y dimensiones de los rectángulos
        if len(rectangulos_regla) > 0:
            debug_info += "\nRectángulos de regla (x,y,w,h):\n"
            for i, (x, y, w, h) in enumerate(rectangulos_regla):
                ar = w/h if h > 0 else 0
                debug_info += f"  Rect {i}: ({x},{y}) size={w}x{h} AR={ar:.1f}\n"
    debug_info += "\n"
    
    # TOP 10 contornos con mayor score
    if scores:
        scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
        debug_info += "TOP 10 CONTORNOS (por score):\n"
        debug_info += "=" * 60 + "\n"
        for i, (score, contorno, area) in enumerate(scores_sorted[:10]):
            debug_info += f"  #{i+1}: Score={score:.1f}, Área={area:.0f}\n"
        debug_info += "\n"
    
    debug_info += "Todos los Scores:\n"
    for razon in filtros_aplicados:
        debug_info += f"  - {razon}\n"
    
    with open(debug_dir / 'debug_info.txt', 'w') as f:
        f.write(debug_info)


def analizar_caracteristicas_contorno(contorno, imagen_bgr, etiqueta=""):
    """Analiza y muestra todas las características de un contorno."""
    x, y, w, h = cv2.boundingRect(contorno)
    area = cv2.contourArea(contorno)
    
    # Extraer ROI
    x1 = max(0, x - 5)
    y1 = max(0, y - 5)
    x2 = min(imagen_bgr.shape[1], x + w + 5)
    y2 = min(imagen_bgr.shape[0], y + h + 5)
    roi_bgr = imagen_bgr[y1:y2, x1:x2]
    
    # Color HSV
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv_roi[:, :, 0])
    s_mean = np.mean(hsv_roi[:, :, 1])
    v_mean = np.mean(hsv_roi[:, :, 2])
    h_std = np.std(hsv_roi[:, :, 0])
    s_std = np.std(hsv_roi[:, :, 1])
    v_std = np.std(hsv_roi[:, :, 2])
    
    # Color BGR
    b_mean = np.mean(roi_bgr[:, :, 0])
    g_mean = np.mean(roi_bgr[:, :, 1])
    r_mean = np.mean(roi_bgr[:, :, 2])
    
    # Forma
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    extent = area / (w * h) if (w * h) > 0 else 0
    perimetro = cv2.arcLength(contorno, True)
    compacidad = (4 * np.pi * area) / (perimetro * perimetro) if perimetro > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"{etiqueta}")
    print(f"{'='*60}")
    print(f"ÁREA: {area:.0f} px")
    print(f"BBOX: {w}x{h} (aspect_ratio={aspect_ratio:.2f})")
    print(f"EXTENT: {extent:.3f}")
    print(f"COMPACIDAD: {compacidad:.3f}")
    print(f"\nCOLOR HSV:")
    print(f"  H (matiz): {h_mean:.1f} ± {h_std:.1f}")
    print(f"  S (saturación): {s_mean:.1f} ± {s_std:.1f}")
    print(f"  V (valor): {v_mean:.1f} ± {v_std:.1f}")
    print(f"\nCOLOR BGR:")
    print(f"  B: {b_mean:.1f}")
    print(f"  G: {g_mean:.1f}")
    print(f"  R: {r_mean:.1f}")
    print(f"{'='*60}\n")
    
    return {
        'area': area,
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'compacidad': compacidad,
        'h_mean': h_mean, 's_mean': s_mean, 'v_mean': v_mean,
        'h_std': h_std, 's_std': s_std, 'v_std': v_std,
        'b_mean': b_mean, 'g_mean': g_mean, 'r_mean': r_mean
    }


def analizar_contornos_especificos(analisis_list):
    """Analiza contornos específicos de varias imágenes.
    
    Args:
        analisis_list: Lista de tuplas (ruta_imagen, contorno_id, etiqueta)
    """
    print("\nMODO ANÁLISIS DE CARACTERÍSTICAS")
    print("Este modo analiza contornos específicos para ajustar los rangos de detección.\n")
    
    resultados = []
    
    for img_path, contorno_id, etiqueta in analisis_list:
        path = Path(img_path)
        if not path.exists():
            print(f"ERROR: No existe {img_path}")
            continue
        
        imagen = cv2.imread(str(path))
        if imagen is None:
            print(f"ERROR: No se pudo leer {img_path}")
            continue
        
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contorno_id >= len(contornos):
            print(f"ERROR: Contorno {contorno_id} no existe en {path.name} (solo hay {len(contornos)} contornos)")
            continue
        
        contorno = contornos[contorno_id]
        etiqueta_full = f"{path.name} - Contorno {contorno_id} [{etiqueta}]"
        resultado = analizar_caracteristicas_contorno(contorno, imagen, etiqueta_full)
        resultado['etiqueta'] = etiqueta
        resultado['imagen'] = path.name
        resultados.append(resultado)
    
    # Estadísticas agregadas
    if resultados:
        print("\n" + "="*60)
        print("RESUMEN ESTADÍSTICO")
        print("="*60)
        
        algas = [r for r in resultados if r['etiqueta'] == 'ALGA']
        reglas = [r for r in resultados if r['etiqueta'] == 'REGLA']
        
        if algas:
            print("\nALGAS:")
            print(f"  H: {np.mean([a['h_mean'] for a in algas]):.1f} ± {np.std([a['h_mean'] for a in algas]):.1f}")
            print(f"  S: {np.mean([a['s_mean'] for a in algas]):.1f} ± {np.std([a['s_mean'] for a in algas]):.1f}")
            print(f"  V: {np.mean([a['v_mean'] for a in algas]):.1f} ± {np.std([a['v_mean'] for a in algas]):.1f}")
            print(f"  Aspect Ratio: {np.mean([a['aspect_ratio'] for a in algas]):.2f} ± {np.std([a['aspect_ratio'] for a in algas]):.2f}")
        
        if reglas:
            print("\nREGLAS:")
            print(f"  H: {np.mean([r['h_mean'] for r in reglas]):.1f} ± {np.std([r['h_mean'] for r in reglas]):.1f}")
            print(f"  S: {np.mean([r['s_mean'] for r in reglas]):.1f} ± {np.std([r['s_mean'] for r in reglas]):.1f}")
            print(f"  V: {np.mean([r['v_mean'] for r in reglas]):.1f} ± {np.std([r['v_mean'] for r in reglas]):.1f}")
            print(f"  Aspect Ratio: {np.mean([r['aspect_ratio'] for r in reglas]):.2f} ± {np.std([r['aspect_ratio'] for r in reglas]):.2f}")
        
        print("="*60 + "\n")
    
    return resultados


# CÓDIGO ELIMINADO - Red de seguridad redundante para detectar overlap con números rojos
# Ya no se usa porque los números se eliminan de edges antes de encontrar contornos
# Se mantiene aquí por si se necesita en el futuro para debugging
"""
def check_overlap_numeros_rojos(contorno, mascara_numeros):
    '''Red de seguridad: verifica si un contorno solapa con números rojos.
    Los números ya fueron eliminados de edges, pero esto detecta solapamiento espacial.
    '''
    area = cv2.contourArea(contorno)
    mascara_contorno = np.zeros(mascara_numeros.shape, dtype=np.uint8)
    cv2.drawContours(mascara_contorno, [contorno], -1, 255, -1)
    overlap = cv2.bitwise_and(mascara_contorno, mascara_numeros)
    overlap_ratio = np.sum(overlap > 0) / (area + 1e-6)
    
    if overlap_ratio > 0.2:  # Más del 20% es rojo
        return -250, f"rojo=-250(overlap={overlap_ratio:.2f})"
    return 0, None
"""
