#!/bin/bash

# Verificar si se ha pasado el nombre del paquete como argumento
if [ -z "$1" ]; then
  echo "Por favor, proporciona el nombre del paquete como argumento."
  echo "Uso: $0 nombre_del_paquete"
  exit 1
fi

PAQUETE=$1

# Desinstalar el paquete
echo "Desinstalando el paquete $PAQUETE..."
pip uninstall -y $PAQUETE

# Eliminar caché de pip
echo "Eliminando caché de pip..."
rm -rf ~/.cache/pip

# Encontrar y eliminar directorios __pycache__
echo "Eliminando directorios __pycache__..."
find . -type d -name "__pycache__" -exec rm -r {} +

# Eliminar caché de pip específico del paquete
echo "Purgando caché de pip..."
pip cache purge

# Intentar eliminar caché del paquete específico (si existe)
echo "Eliminando caché específico del paquete $PAQUETE..."
rm -rf ~/.cache/$PAQUETE

# Confirmar eliminación
echo "Eliminación completa de $PAQUETE y su caché finalizada."

# Mensaje final
echo "Todos los pasos se han completado. Asegúrate de verificar manualmente si quedan archivos residuales."

exit 0