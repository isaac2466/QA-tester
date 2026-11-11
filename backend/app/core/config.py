import os
from pathlib import Path

# Lee backend/.env (contraseña de MySQL, etc.). Ese archivo no se sube a git.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

# Modelo DINOv2. La variante "with-registers" produce mapas de atención mucho más
# limpios (sin los artefactos de alta norma del DINOv2 original).
DINO_MODEL = os.getenv("DINO_MODEL", "facebook/dinov2-with-registers-small")

# Ancho (en px) al que se reescala la captura antes de pasarla por DINOv2.
# Debe ser múltiplo de 14 (tamaño de parche). 448 = 32 parches.
DINO_INPUT_WIDTH = int(os.getenv("DINO_INPUT_WIDTH", "448"))
DINO_MAX_HEIGHT = int(os.getenv("DINO_MAX_HEIGHT", "1344"))

# Límite de tamaño de las imágenes subidas (en lado mayor) para el análisis por píxel.
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "4096"))

# Modelo de visión + lenguaje local (Ollama) que verifica los hallazgos y revisa el diseño.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# "instruct" responde directo; la versión normal "piensa" en voz alta y gasta minutos extra
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b-instruct")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "600"))  # en CPU puede tardar varios minutos
VLM_MAX_SIDE = int(os.getenv("VLM_MAX_SIDE", "1024"))  # lado mayor de la imagen que ve el modelo

CORS_ORIGINS =os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "qa_tester"),
}
AUTH_TOKEN_DAYS = int(os.getenv("AUTH_TOKEN_DAYS", "7"))
