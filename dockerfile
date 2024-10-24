FROM python:3.8-slim

# Instalar dependencias del sistema necesarias para OpenCV y MTCNN
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que usará Flask
EXPOSE 5001

# Ejecutar la aplicación usando Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5001", "server:app"]
