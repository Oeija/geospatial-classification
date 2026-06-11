"""Prediction API routes."""

import logging

from fastapi import APIRouter, HTTPException

from api.app.schemas import (
    AreaPredictionRequest,
    AreaPredictionResponse,
    PointPredictionRequest,
    PointPredictionResponse,
)
from api.app.services import ModelService, SatelliteService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])

model_service = ModelService()
satellite_service = SatelliteService()


@router.post("/point", response_model=PointPredictionResponse)
async def predict_point(request: PointPredictionRequest) -> dict:
    """Classify a single point on the map as agricultural or non-agricultural land.

    - Fetches a 64×64 satellite tile from Mapbox
    - Runs CNN-ViT inference on GPU
    - Returns prediction with confidence scores
    """
    try:
        # 1. Fetch satellite image
        image = satellite_service.fetch_image(request.lat, request.lng, request.zoom)

        # 2. Preprocess
        preprocessed = satellite_service.preprocess(image)
        tensor = preprocessed["tensor"]

        # 3. Inference
        result = model_service.predict(tensor)

        # 4. Build image URL for frontend
        image_url = satellite_service.get_image_url(request.lat, request.lng, request.zoom)

        return {
            "lat": request.lat,
            "lng": request.lng,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "image_url": image_url,
        }

    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@router.post("/area", response_model=AreaPredictionResponse)
async def predict_area(request: AreaPredictionRequest) -> dict:
    """Classify an area (polygon) on the map as agricultural or non-agricultural land.

    - Generates a grid of points within the polygon
    - Fetches satellite tiles for each point
    - Runs CNN-ViT inference on GPU for each point
    - Returns aggregated statistics with per-point results
    """
    try:
        # 1. Generate grid points within polygon
        grid_points = _generate_grid_points(request.polygon, grid_size=request.grid_size)

        if not grid_points:
            raise HTTPException(status_code=400, detail="No grid points generated from polygon")

        # 2. Run inference on each grid point
        results = []
        for point in grid_points:
            try:
                image = satellite_service.fetch_image(point["lat"], point["lng"], request.zoom)
                preprocessed = satellite_service.preprocess(image)
                tensor = preprocessed["tensor"]
                result = model_service.predict(tensor)
                image_url = satellite_service.get_image_url(
                    point["lat"], point["lng"], request.zoom
                )

                results.append(
                    {
                        "lat": point["lat"],
                        "lng": point["lng"],
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "probabilities": result["probabilities"],
                        "image_url": image_url,
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Failed to predict point (%s, %s): %s",
                    point["lat"],
                    point["lng"],
                    exc,
                )
                continue

        if not results:
            raise HTTPException(status_code=500, detail="All grid points failed to predict")

        # 3. Calculate statistics
        agri_count = sum(1 for r in results if r["prediction"] == "agri")
        non_agri_count = len(results) - agri_count
        agri_percentage = (agri_count / len(results)) * 100 if results else 0
        avg_confidence = sum(r["confidence"] for r in results) / len(results)

        # 4. Calculate bounding box (polygon is list of [lng, lat])
        lats = [p[1] for p in request.polygon]
        lngs = [p[0] for p in request.polygon]
        bounding_box = {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lng": min(lngs),
            "max_lng": max(lngs),
        }

        return {
            "total_points": len(results),
            "agri_points": agri_count,
            "non_agri_points": non_agri_count,
            "agri_percentage": round(agri_percentage, 2),
            "avg_confidence": round(avg_confidence, 4),
            "grid_points": results,
            "bounding_box": bounding_box,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Area prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Area prediction failed: {exc}")


def _generate_grid_points(polygon: list[list[float]], grid_size: int = 5) -> list[dict]:
    """Generate evenly spaced grid points within a polygon.

    Args:
        polygon: List of [lng, lat] coordinates
        grid_size: Number of grid points per side

    Returns:
        List of {"lat": float, "lng": float} dictionaries
    """
    # Extract coordinates
    lngs = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]

    min_lng, max_lng = min(lngs), max(lngs)
    min_lat, max_lat = min(lats), max(lats)

    # Generate grid
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            lng = min_lng + (max_lng - min_lng) * (i / (grid_size - 1))
            lat = min_lat + (max_lat - min_lat) * (j / (grid_size - 1))

            # Simple point-in-polygon check (ray casting)
            if _point_in_polygon(lng, lat, polygon):
                points.append({"lat": lat, "lng": lng})

    return points


def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Ray casting algorithm to check if point is inside polygon."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0][0], polygon[0][1]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n][0], polygon[i % n][1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
