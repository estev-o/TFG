#!/usr/bin/env python3
"""Leer `dataset/Kelps_database_photos/Photos_kelp_database.xlsx`, quedarse solo
con las columnas `Photo_cod`, `HPI` e `IVR` y guardar el resultado en CSV y XLSX.

Comportamiento:
- Comprueba que el fichero de entrada existe.
- Busca las columnas por nombre de forma robusta (insensible a mayúsculas/espacios).
- Si falta alguna columna, muestra las columnas disponibles y aborta.
"""
from pathlib import Path
import sys
import pandas as pd

BASE = Path(__file__).parent
INPUT = BASE / 'dataset' / 'Kelps_database_photos' / 'Photos_kelp_database.xlsx'
OUT_CSV = BASE / 'dataset' / 'kelp_photos_filtered.csv'
OUT_XLSX = BASE / 'dataset' / 'kelp_photos_filtered.xlsx'


def find_column(df_columns, target):
	"""Buscar una columna en df_columns de forma robusta.

	target: nombre esperado como 'Photo_cod', 'HPI' o 'IVR'
	Devuelve el nombre real de la columna si se encuentra, o None.
	"""
	# comprobación exacta primero
	for c in df_columns:
		if c == target:
			return c

	# caso-insensible y con/ sin guiones/espacios
	lowered = {c.lower(): c for c in df_columns}
	candidate_forms = [target, target.replace('_', ' '), target.lower(), target.upper(), target.capitalize()]
	for form in candidate_forms:
		if form.lower() in lowered:
			return lowered[form.lower()]

	# búsqueda por contains
	for c in df_columns:
		if target.lower() in c.lower():
			return c

	return None


def main():
	if not INPUT.exists():
		print(f'ERROR: archivo de entrada no encontrado: {INPUT}')
		sys.exit(2)

	try:
		df = pd.read_excel(INPUT, engine='openpyxl')
	except Exception as e:
		print('ERROR leyendo el Excel:', e)
		sys.exit(3)

	wanted = ['Photo_cod', 'HPI', 'IVR']
	mapped = {}
	for w in wanted:
		col = find_column(df.columns, w)
		if col is None:
			print('\nERROR: no se encontró columna para', w)
			print('Columnas disponibles:', list(df.columns))
			sys.exit(4)
		mapped[w] = col

	out = df[[mapped['Photo_cod'], mapped['HPI'], mapped['IVR']]].copy()
	out.columns = ['Photo_cod', 'HPI', 'IVR']

	# Forzar HPI e IVR a valores numéricos; valores no numéricos se convierten a NaN
	out['HPI'] = pd.to_numeric(out['HPI'], errors='coerce')
	out['IVR'] = pd.to_numeric(out['IVR'], errors='coerce')

	# Eliminar filas con valores no numéricos en HPI o IVR (NaN tras coerción)
	before_drop_non_numeric = len(out)
	out = out.dropna(subset=['HPI', 'IVR'])
	dropped_non_numeric = before_drop_non_numeric - len(out)

	# Eliminar filas donde ambos HPI e IVR son 0
	before_drop_both_zero = len(out)
	out = out[~((out['HPI'] == 0) & (out['IVR'] == 0))].copy()
	dropped_both_zero = before_drop_both_zero - len(out)

	out.to_csv(OUT_CSV, index=False)
	out.to_excel(OUT_XLSX, index=False)

	print(f'Salida escrita: {OUT_CSV}  ({len(out)} filas)')
	print(f'Salida escrita: {OUT_XLSX}  ({len(out)} filas)')
	print(f'Filas eliminadas por valores no numéricos: {dropped_non_numeric}')
	print(f'Filas eliminadas donde HPI e IVR eran ambos 0: {dropped_both_zero}')


if __name__ == '__main__':
	main()

