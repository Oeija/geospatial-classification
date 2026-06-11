"""Satellite image fetching service via Mapbox Static Images API."""

import logging
import os
from io import BytesIO
from pathlib import Path

import httpx
from dotenv import load_dotenv
from PIL import Image

from geospatial_classification.data.transforms import get_inference_transform

# Load .env from api/ directory
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
if not MAPBOX_TOKEN:
    raise RuntimeError("MAPBOX_ACCESS_TOKEN not set in environment")

STATIC_IMAGE_URL = (
    "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
    "/{lng},{lat},{zoom},0,0/{width}x{height}@2x"
    "?access_token={token}"
)


class SatelliteService:
    """Fetch satellite tiles from Mapbox and convert to model-ready tensors."""

    def __init__(self) -> None:
        self.token = MAPBOX_TOKEN
        self.transform = get_inference_transform((64, 64))

    def fetch_image(self, lat: float, lng: float, zoom: int = 16) -> Image.Image:
        """Fetch a 64×64 satellite tile from Mapbox.

        Returns a PIL Image in RGB mode.
        """
        url = STATIC_IMAGE_URL.format(
            lng=lng,
            lat=lat,
            zoom=zoom,
            width=64,
            height=64,
            token=self.token,
        )

        logger.info("Fetching satellite image: lat=%s, lng=%s, zoom=%s", lat, lng, zoom)
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")
        logger.info("Fetched image size: %s", image.size)
        return image

    def get_image_url(self, lat: float, lng: float, zoom: int = 16) -> str:
        """Return the direct Mapbox image URL (for frontend display)."""
        return STATIC_IMAGE_URL.format(
            lng=lng,
            lat=lat,
            zoom=zoom,
            width=64,
            height=64,
            token=self.token,
        )

    def preprocess(self, image: Image.Image) -> dict:
        """Convert PIL Image to model-ready tensor and metadata.

        Returns:
            {"tensor": torch.Tensor, "image_url": str}
        """
        tensor = self.transform(image)
        return {
            "tensor": tensor,
        }
