#!/bin/bash

# Crear la estructura de directorios
mkdir -p api_project/{core,models,prompts,chunks,agents,storage,api,tests}

# Crear archivos en el directorio raíz
touch api_project/{requirements.txt,Dockerfile,.env,README.md}

# Crear archivos en el directorio core
touch api_project/core/{__init__.py,config.py,utils.py}

# Crear archivos en el directorio models
touch api_project/models/{__init__.py,model_manager.py,embeddings.py}

# Crear archivos en el directorio prompts
touch api_project/prompts/{__init__.py,prompt_processor.py,prompt_handler.py}

# Crear archivos en el directorio chunks
touch api_project/chunks/{__init__.py,text_chunks.py,code_chunks.py,chunk_handler.py}

# Crear archivos en el directorio agents
touch api_project/agents/{__init__.py,autoagent.py}

# Crear archivos en el directorio storage
touch api_project/storage/{__init__.py,database.py,models.py}

# Crear archivos en el directorio api
touch api_project/api/{__init__.py,main.py,routes.py}

# Crear archivos en el directorio tests
touch api_project/tests/{__init__.py,test_model_manager.py,test_embeddings.py,test_prompt_processor.py,test_chunks.py,test_autoagent.py,test_database.py}

echo "Estructura del proyecto creada con éxito."