# 📰 SmartNewsAPI – API de Inteligencia de Noticias  

## 📌 Descripción
SmartNewsAPI es un servicio backend diseñado para **ingerir artículos de noticias, limpiarlos, generar embeddings semánticos y permitir búsquedas vectoriales**.  
Está pensado para ser **escalable, reproducible y desplegable en la nube (GCP Cloud Run con Terraform + CI/CD)**.

La API ofrece:
- Indexación automática de artículos desde RSS.  
- Búsqueda semántica con embeddings.  
- Soporte para chunking tradicional y chunking inteligente (Gemini).  
- Vector Store con Qdrant.  

---

## ⚙️ Arquitectura General

### Ingesta de datos (RSS)
- Lectura de feeds RSS (BBC, Reuters, El País, etc.).  
- Descarga y limpieza de HTML con `trafilatura` + `BeautifulSoup`.  
- Filtrado de ruido (cookies, banners, publicidad, social media embeds).  

### Procesamiento de contenido
- Normalización de texto (`quality.py`).  
- Segmentación en chunks:  
  - **Simple** → por caracteres con solapamiento.  
  - **Agentic** → vía Gemini (Vertex AI) con JSON validado.  

### Generación de embeddings
- **LocalE5Provider** (SentenceTransformers en CPU).  
- **VertexEmbeddingProvider** (GCP Vertex AI: `text-embedding-004`, `gemini-embedding-001`, etc.).  
- Con lógica de reintentos con backoff y conversión de errores de cuota (429 → 503).  

### Vector Store
- Persistencia de embeddings en **Qdrant** (local o cloud).  
- Indexación de payload enriquecido (fuente, fecha, snippets, etc.).  

### API REST (FastAPI)
- `POST /index` → indexa noticias en la base vectorial.  
- `GET /search` → búsqueda semántica.  
- `GET /feeds` → lista de feeds configurados.  
- `GET /health` → health check.  

---

## 📡 Endpoints principales

### 🔹 `POST /index`
- Indexa artículos de RSS en la base vectorial.  
- Limpia, filtra y chunkifica el texto.  
- Genera embeddings y los guarda en Qdrant.  
- Responde con número de artículos y chunks indexados.  

### 🔹 `GET /search`
- Búsqueda semántica de artículos.  
- Convierte la query en embeddings.  
- Busca en Qdrant.  
- Devuelve artículos agrupados con snippets relevantes.  

### 🔹 `GET /feeds`
- Devuelve la lista de feeds RSS usados para la ingesta.  

### 🔹 `GET /health`
- Health check sencillo (`{"ok": true}`).  

---

## 🛡️ Manejo de cuotas y errores
- Cuando Vertex AI devuelve un **429 / ResourceExhausted / QuotaExceeded**, la API lo transforma en **503 Service Unavailable**.  
- Esto evita falsos **500 Internal Server Error** y comunica mejor la causa.  
- Implementado en `embed_in_batches` + `is_429_error`.  

---

## 🧩 Tecnologías usadas
- **Lenguaje:** Python 3 + FastAPI  
- **Embeddings:**  
  - Local con SentenceTransformers (E5)  
  - GCP Vertex AI (`text-embedding-004`, `gemini-embedding-001`)  
- **Chunking:**  
  - Simple (caracteres con solapamiento)  
  - Agentic (Gemini LLM en Vertex AI)  
- **Vector DB:** Qdrant  
- **Infraestructura:** Terraform + Cloud Run (GCP)  
- **CI/CD:** Automatizado con tests + build + deploy  
- **Tests:** Pytest para validar ingestion, embeddings y búsqueda  

---

## 📊 Flujo de Indexación
1. Descargar noticias de RSS.  
2. Limpiar HTML y boilerplate.  
3. Filtrar calidad (mínimo caracteres + score heurístico).  
4. Chunkificar.  
5. Generar embeddings en batch con backoff.  
6. Insertar en Qdrant con metadata.  

---

## 🚀 Despliegue
- **Contenerización:** Dockerfile para empaquetar la app.  
- **Infraestructura:** Terraform define Cloud Run, permisos.  
- **CI/CD:** Al hacer push → se ejecuta el build, y se despliega automáticamente.  

