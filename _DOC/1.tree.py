import os
import json

def get_directory_structure(dir_path, exclude_folders=[], exclude_files=[]):
    result = {}

    def read_directory(current_path, obj):
        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)

            # Comprobar si la carpeta debe ser excluida
            if os.path.isdir(item_path) and any(exclude_path in item_path for exclude_path in exclude_folders):
                continue

            # Comprobar si el archivo debe ser excluido
            if not os.path.isdir(item_path) and item in exclude_files:
                continue

            if os.path.isdir(item_path):
                obj[item] = {}
                read_directory(item_path, obj[item])
            else:
                obj[item] = 'file'

    read_directory(dir_path, result)
    return result

# Ruta del directorio a analizar
directory_path = '../src'  # Puedes cambiar esto por cualquier directorio

# Lista de carpetas a excluir
exclude_folders = [
    'node_modules',
    'strategies',
    '__pycache__',
    '.pytest_cache',
    '_documentation',
    '_Guide',
    '.git',
    '_REST',
    'apillm',
    'venv',
    'origin',
    '.vscode',
    'tests',
    'dist',
    'test/subtest'  # Ejemplo de subcarpeta a excluir
]

# Lista de archivos a excluir
exclude_files = [
    'README.md',
    'package-lock.json',
    'jest.config.mjs',
    '__init__.py',
    '.env',
    '.gitignore',
    'requirements.txt'
]

# Obtener la estructura del directorio
directory_structure = get_directory_structure(directory_path, exclude_folders, exclude_files)

# Guardar el resultado en un archivo JSON
output_path = os.path.join('public', 'projectTree.json')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(directory_structure, f, indent=2, ensure_ascii=False)

print('La estructura del directorio se ha guardado en projectTree.json')
