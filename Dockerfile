FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Copia requerimientos
COPY requirements.txt ./

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia código
COPY app ./app

EXPOSE 8080

# Arranque
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
