"""Sentinel-2 image download from Google Earth Engine."""
import ee


def get_bounding_box(lat, lon, buffer_km):
    """Create a bounding box around a point.

    Args:
        lat: Center latitude.
        lon: Center longitude.
        buffer_km: Buffer distance in kilometers.

    Returns:
        ee.Geometry.Rectangle bounding box.
    """
    buffer_deg = buffer_km / 111.0
    return ee.Geometry.Rectangle(
        [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
    )


def download_sentinel2(lat, lon, buffer_km=15, max_cloud_cover=10, days=60):
    """Download the most recent Sentinel-2 image for a location.

    Args:
        lat: Center latitude.
        lon: Center longitude.
        buffer_km: Area size in kilometers.
        max_cloud_cover: Maximum cloud cover percentage.
        days: Number of days to search back.

    Returns:
        ee.Image with bands B4, B3, B2, B8.
    """
    bbox = get_bounding_box(lat, lon, buffer_km)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bbox)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
        .filterDate(
            ee.Date("2024-01-01"),
            ee.Date("2026-12-31"),
        )
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    image = collection.first().clip(bbox)

    print(f"Image found: {image.get('system:index').getInfo()}")
    print(f"Cloud cover: {image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()}%")

    return image.select(["B4", "B3", "B2", "B8"])
