import os
from pathlib import Path

# Modelo DINOv2. La variante "with-registers" produce mapas de atención mucho más
# limpios (sin los artefactos de alta norma del DINOv2 original).
DINO_MODEL = os.getenv("DINO_MODEL", "facebook/dinov2-with-registers-small")

# Ancho (en px) al que se reescala la captura antes de pasarla por DINOv2.
# Debe ser múltiplo de 14 (tamaño de parche). 448 = 32 parches.
DINO_INPUT_WIDTH = int(os.getenv("DINO_INPUT_WIDTH", "448"))
DINO_MAX_HEIGHT = int(os.getenv("DINO_MAX_HEIGHT", "1344"))

# Límite de tamaño de las imágenes subidas (en lado mayor) para el análisis por píxel.
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "4096"))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
DATABASE_PATH = os.getenv("DATABASE_PATH", str(Path(__file__).resolve().parents[2] / "data" / "users.db"))
