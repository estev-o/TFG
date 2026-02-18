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
import argparse
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT = REPO_ROOT / 'dataset' / 'Kelps_database_photos' / 'Photos_kelp_database.xlsx'
OUT_CSV = REPO_ROOT / 'dataset' / 'kelp_photos_filtered.csv'
OUT_XLSX = REPO_ROOT / 'dataset' / 'kelp_photos_filtered.xlsx'
PHOTOS_DIR = REPO_ROOT / 'dataset' / 'Kelps_database_photos' / 'Photos_kelps_database'
PHOTO_EXTS = {'.jfif', '.jpg', '.jpeg', '.png', '.heic'}


def parse_args():
	parser = argparse.ArgumentParser(
		description='Filtra el Excel y opcionalmente limpia fotos inexistentes en el CSV.'
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Solo muestra cuántas fotos se eliminarían, sin borrar nada.'
	)
	parser.add_argument(
		'--num_samples',
		type=int,
		default=None,
		help='Compatibilidad con Makefile; no se usa en este script.'
	)
	return parser.parse_args()


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


def limpiar_fotos(out, dry_run=False):
	codigos_validos = set()
	for code in out['Photo_cod']:
		if pd.isna(code):
			continue
		code_str = str(code).strip()
		if code_str:
			codigos_validos.add(code_str)

	if not PHOTOS_DIR.exists():
		print(f'AVISO: no se encontró el directorio de fotos: {PHOTOS_DIR}')
		return

	codigos_norm = {c.lower() for c in codigos_validos}
	codigos_stem = {Path(c).stem.lower() for c in codigos_validos}

	revisadas = 0
	ignorar = 0
	eliminadas = 0
	eliminables = 0

	for foto in PHOTOS_DIR.iterdir():
		if not foto.is_file():
			continue

		if foto.suffix.lower() not in PHOTO_EXTS:
			ignorar += 1
			continue

		revisadas += 1
		nombre = foto.name.lower()
		stem = foto.stem.lower()

		if nombre in codigos_norm or stem in codigos_stem:
			continue

		if dry_run:
			eliminables += 1
			continue

		try:
			foto.unlink()
			eliminadas += 1
		except Exception as e:
			print(f'No se pudo eliminar {foto.name}: {e}')

	quedarian = revisadas - (eliminables if dry_run else eliminadas)
	if dry_run:
		print(
			f'Revisión de {PHOTOS_DIR}: {revisadas} archivos de imagen, '
			f'{eliminables} se eliminarían, {quedarian} quedarían, '
			f'{ignorar} ignorados por extensión (dry-run, no se borró nada)'
		)
	else:
		print(
			f'Revisión de {PHOTOS_DIR}: {revisadas} archivos de imagen, '
			f'{eliminadas} eliminados, {quedarian} quedan, {ignorar} ignorados por extensión'
		)


def main():
	args = parse_args()
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

	# Mantener filas donde ambos HPI e IVR son 0 para entrenar también casos sin mordida
	both_zero_rows = int(((out['HPI'] == 0) & (out['IVR'] == 0)).sum())

	out.to_csv(OUT_CSV, index=False)
	out.to_excel(OUT_XLSX, index=False)

	print(f'Salida escrita: {OUT_CSV}  ({len(out)} filas)')
	print(f'Salida escrita: {OUT_XLSX}  ({len(out)} filas)')
	print(f'Filas eliminadas por valores no numéricos: {dropped_non_numeric}')
	print(f'Filas con HPI e IVR ambos 0 conservadas: {both_zero_rows}')

	limpiar_fotos(out, dry_run=args.dry_run)


if __name__ == '__main__':
	main()
