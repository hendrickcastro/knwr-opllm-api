import os
import json

# Leer la estructura del directorio desde projectTree.json
with open('./public/projectTree.json', 'r', encoding='utf-8') as f:
    directory_structure = json.load(f)

# Crear carpeta docs si no existe
output_dir = './public/docs'
os.makedirs(output_dir, exist_ok=True)

def generate_documentation(structure, base_path=''):
    def process_structure(structure, current_path):
        for key, value in structure.items():
            item_path = os.path.join(current_path, key)
            if value == 'file' and key.endswith('.py'):
                with open(item_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                markdown_content = f"""
## Archivo: {key}
### Ruta Relativa: {item_path}

```python
{content}
```
"""
                output_file_path = os.path.join(output_dir, f"api_{key.replace('.py', '')}.md")
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
            elif isinstance(value, dict):
                process_structure(value, item_path)

    process_structure(structure, base_path)

# Generar la documentación
generate_documentation(directory_structure, '../src')

print('La documentación del código se ha guardado en la carpeta docs')