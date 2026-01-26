# Proyecto: Filtrado de Photos_kelp_database

Resumen de lo hecho
--------------------

He implementado y ejecutado un script (`script_filtro.py`) que procesa el archivo
`dataset/Kelps_database_photos/Photos_kelp_database.xlsx` y genera una versión
filtrada que contiene únicamente las columnas:

- `Photo_cod`
- `HPI`
- `IVR`

Filtros aplicados
-----------------

1. Se convierten `HPI` e `IVR` a valores numéricos; las entradas no numéricas se
	 eliminan (se convirtieron a NaN con coerción y luego se descartaron).
2. Se eliminan las filas donde ambos `HPI` e `IVR` son 0.

Archivos de salida
------------------

Los archivos generados están en `dataset/`:

- `kelp_photos_filtered.csv`
- `kelp_photos_filtered.xlsx`

Resultados (ejecución realizada)
--------------------------------

- Filas originales: 5490 (conteo antes del filtrado)
- Filas eliminadas por valores no numéricos en HPI/IVR: 4
- Filas eliminadas donde HPI e IVR eran ambos 0: 1364
- Filas resultantes: 4122

Cómo ejecutar
-------------

1. Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecutar el script desde la raíz del proyecto:

```bash
python script_filtro.py
```

Salidas esperadas:

- `dataset/kelp_photos_filtered.csv`
- `dataset/kelp_photos_filtered.xlsx`

Notas adicionales
-----------------

- El script busca las columnas de forma robusta (insensible a mayúsculas/espacios).
- Si una columna esperada no se encuentra, el script imprimirá las columnas
	detectadas y abortará.

# Normalización de Imágenes

Coinciden solo 3 769: sobran 1 215 fotos que no aparecen en el CSV, y faltan 354 códigos del CSV que no tienen foto en el directorio.

1. Recortar el alga
	1.1 Técnicas de CV
		1.1.1 Detectar qué es el alga (lo más centrado, colores de alga, tamaño irregular (no cuadrado o texto)...)
		1.1.2 Calcular su bounding box y recortar
		1.1.3 Recrear la imagen con tamaño que querramos y padding para mantener la relacción de aspecto

		1.1.4 Usamos YOLO entrenándolo con ejemplos bien recortados de el primer proceso, así podremos generalizar más facilmente que haciendo el coso directamente

		El primer entrenamiento sotisfactorio de YOLO8n ha dado resultados muy destacables frente a métodos de visión. Con tan solo alrededor de 1000 imágenes de test y 200 de validación. Hemos podido recortar
		4705 fotos + (37 no detectados + 155 errores)
	1.2 Técnicas de IA