# TouringGO - Embeddings Service

Microservicio encargado de generar representaciones vectoriales (*embeddings*) de textos para el sistema de búsqueda semántica de TouringGO.

Utiliza **Sentence Transformers** y el modelo `BAAI/bge-m3`, que permite generar embeddings multilingües de 1024 dimensiones.

## Responsabilidades

Este servicio centraliza la generación de embeddings para evitar que otros microservicios tengan que cargar y ejecutar el modelo directamente.

Sus principales responsabilidades son:

- Generar embeddings de descripciones de lugares.
- Generar embeddings de consultas de búsqueda.
- Normalizar los vectores generados para facilitar la comparación mediante similitud coseno.
- Proporcionar una API HTTP para que otros servicios puedan solicitar embeddings.

El microservicio `touringgo-lugares` consume esta API para generar y utilizar embeddings en sus búsquedas semánticas.

## Tecnologías

- Python 3.11
- FastAPI
- Uvicorn
- Sentence Transformers
- PyTorch
- Modelo: `BAAI/bge-m3`

## Estructura del proyecto

```text
touringgo-embeddings/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── embedding_service.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Modelo de embeddings

Se utiliza el modelo `BAAI/bge-m3`.

Características principales:

- 1024 dimensiones por embedding.
- Soporte multilingüe, incluido español.
- Adecuado para búsqueda semántica.
- Permite comparar consultas y documentos mediante similitud coseno.

Los embeddings se generan normalizados mediante `normalize_embeddings=True`, por lo que la similitud coseno puede calcularse mediante el producto punto entre dos vectores.

## API

La API expone los siguientes endpoints:

### GET /health

Permite comprobar el estado del servicio y verificar que el modelo se haya cargado correctamente.

Ejemplo de respuesta:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /embeddings/documents

Genera embeddings para uno o varios textos que representan documentos.

Se utiliza principalmente para generar los embeddings de los lugares del catálogo.

**Request:**

```json
{
  "texts": [
    "El Obelisco es uno de los monumentos más emblemáticos de Buenos Aires.",
    "El Jardín Japonés es un espacio verde de la ciudad."
  ]
}
```

**Response:**

```json
{
  "embeddings": [
    [0.0123, -0.0345, 0.0567],
    [0.0234, 0.0456, -0.0678]
  ],
  "dimensions": 1024,
  "count": 2
}
```

Los vectores del ejemplo están abreviados. Cada embedding contiene 1024 valores.

### POST /embeddings/queries

Genera embeddings para consultas de búsqueda.

Se utiliza cuando el usuario realiza una búsqueda semántica de lugares.

**Request:**

```json
{
  "texts": [
    "Museos de arte y exposiciones culturales en Buenos Aires."
  ]
}
```

**Response:**

Tiene la misma estructura que el endpoint de documentos.

### Documentación interactiva

FastAPI proporciona documentación interactiva mediante Swagger UI:

```text
/docs
```

También se encuentra disponible la especificación OpenAPI:

```text
/openapi.json
```

## Ejecución local

Se necesita Python 3.11 o superior.

### 1. Crear un entorno virtual

```bash
python -m venv .venv
```

Activarlo en Windows:

```powershell
.venv\Scripts\Activate.ps1
```

En Linux:

```bash
source .venv/bin/activate
```

### 2. Instalar dependencias

Instalar primero la versión CPU de PyTorch:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
```

Instalar el resto de las dependencias:

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el servicio

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

La API estará disponible en:

```text
http://localhost:8000
```

La primera ejecución puede requerir descargar el modelo desde Hugging Face.

## Ejecución con Docker

El repositorio incluye un Dockerfile para construir la imagen del servicio.

Construir la imagen:

```bash
docker build -t touringgo-embeddings .
```

Ejecutar el contenedor:

```bash
docker run -d \
  --name touringgo-embeddings \
  -p 8004:8000 \
  -v embeddings-model-cache:/models \
  touringgo-embeddings
```

El puerto interno del contenedor es `8000`. En este ejemplo se publica como `8004` en el host para evitar conflictos con otros servicios de TouringGO.

El volumen `embeddings-model-cache` permite conservar los archivos descargados del modelo entre recreaciones del contenedor.

La API quedará disponible en:

```text
http://localhost:8004
```

## Integración con TouringGO

El servicio está diseñado para ejecutarse dentro de la red Docker de TouringGO.

El microservicio `touringgo-lugares` debe utilizar la siguiente dirección para comunicarse con él:

```text
http://embeddings-api:8000
```

Configuración mediante variables de entorno:

```yaml
Embeddings__BaseUrl: "http://embeddings-api:8000"
```

No es necesario publicar el puerto del servicio al host para permitir esta comunicación. Los contenedores pueden comunicarse mediante el nombre del servicio dentro de la red de Docker Compose.

## Configuración

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `MODEL_NAME` | Identificador del modelo de Hugging Face | `BAAI/bge-m3` |
| `MODEL_PATH` | Ruta local de un modelo previamente descargado | No definida |
| `HF_HOME` | Directorio utilizado por Hugging Face para almacenar modelos y caché | Valor por defecto de Hugging Face |

Cuando no se especifica `MODEL_PATH`, se utiliza el identificador de `MODEL_NAME` (por defecto `BAAI/bge-m3`) y se descarga el modelo si no está disponible en la caché.

## Consideraciones

- El modelo se carga una sola vez durante el inicio de la aplicación.
- Las solicitudes pueden contener varios textos, permitiendo generar embeddings por lotes.
- Los embeddings de documentos y consultas se generan sobre el texto tal cual (bge-m3 no requiere prefijos).
- Los vectores generados tienen 1024 dimensiones y están normalizados.
- La inferencia se realiza mediante CPU en la configuración Docker prevista.
- El servicio no almacena los embeddings de los lugares: su persistencia y utilización corresponden a `touringgo-lugares`.

## Proyecto

TouringGO es un asistente de planificación turística para la Ciudad Autónoma de Buenos Aires (CABA), desarrollado como proyecto académico de Ingeniería en Informática.