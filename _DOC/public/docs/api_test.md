
## Archivo: test.py
### Ruta Relativa: ../src\tools\test.py

```python
from transformers import AutoModelForCausalLM
import torch

# Importar el tokenizador personalizado
from InternVL226B.tokenization_internlm2 import InternLM2Tokenizer
from InternVL226B.tokenization_internlm2_fast import InternLM2TokenizerFast

# Directorio donde se guardaron los archivos descargados
model_directory = "./InternVL226B/"

# Cargar el tokenizador y el modelo desde el directorio local
tokenizer = InternLM2Tokenizer.from_pretrained(model_directory)
# Alternativamente, si usas el tokenizador rápido
# tokenizer = InternLM2TokenizerFast.from_pretrained(model_directory)

model = AutoModelForCausalLM.from_pretrained(model_directory, trust_remote_code=True)

# Verificar la carga correcta
print("Modelo y tokenizador cargados correctamente.")

# Ejemplo de texto de entrada para el modelo
input_text = "Once upon a time"

# Tokenizar la entrada
inputs = tokenizer(input_text, return_tensors="pt")

# Generar texto con el modelo
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,  # Máxima longitud del texto generado
        num_return_sequences=1,  # Número de secuencias a generar
        no_repeat_ngram_size=2,  # Evita la repetición de n-gramas
        early_stopping=True
    )

# Decodificar y mostrar el texto generado
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Texto generado: {generated_text}")

```
