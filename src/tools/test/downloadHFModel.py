import requests
import os

# Autenticarte en Hugging Face
huggingface_token = "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"  # Reemplaza con tu token de Hugging Face
headers = {"Authorization": f"Bearer {huggingface_token}"}

# Lista de archivos a descargar
files_to_download = [
    "model.safetensors.index.json",
    "modeling_intern_vit.py",
    "modeling_internlm2.py",
    "modeling_internvl_chat.py",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenization_internlm2.py",
    "tokenization_internlm2_fast.py",
    "tokenizer.model",
    "tokenizer_config.json",
]

# Directorio base de Hugging Face
base_url = "https://huggingface.co/OpenGVLab/InternVL2-26B/resolve/main/"

# Directorio donde se guardarán los archivos descargados
download_directory = "./InternVL2-26B/"
os.makedirs(download_directory, exist_ok=True)

# Función para descargar un archivo
def download_file(file_url, destination):
    response = requests.get(file_url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {destination}")
    else:
        print(f"Failed to download {file_url}")

# Descargar cada archivo en la lista
for file_name in files_to_download:
    file_url = base_url + file_name
    destination = os.path.join(download_directory, file_name)
    download_file(file_url, destination)
