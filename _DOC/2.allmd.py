import os
import json

def generate_documentation(structure, base_path=''):
    documentation = ""

    def process_structure(structure, current_path):
        nonlocal documentation
        for key, value in structure.items():
            item_path = os.path.join(current_path, key)
            if value == 'file' and (key.endswith('.py') or key.endswith('.json')):
                try:
                    with open(item_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    documentation += f"""
## Archivo: {key}
### Ruta Relativa: {item_path}

```{'python' if key.endswith('.py') else 'json'}
{content}
```

"""
                except Exception as e:
                    documentation += f"""
## Archivo: {key}
### Ruta Relativa: {item_path}

Error al leer el archivo: {str(e)}

"""
            elif isinstance(value, dict):
                process_structure(value, item_path)

    process_structure(structure, base_path)
    return documentation

# Leer la estructura del directorio desde projectTree.json
with open('public/projectTree.json', 'r', encoding='utf-8') as f:
    directory_structure = json.load(f)

# Generar la documentación
documentation = generate_documentation(directory_structure, '../api')

# Guardar la documentación en un archivo .md
os.makedirs('public/docs', exist_ok=True)
with open('public/docs/all.md', 'w', encoding='utf-8') as f:
    f.write(documentation)

print('La documentación del código se ha guardado en public/docs/all.md')