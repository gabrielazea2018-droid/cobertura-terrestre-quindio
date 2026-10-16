"""Interactive map visualization with Folium."""
import folium
import numpy as np
from folium.plugins import FloatImage


def create_interactive_map(lat, lon, label_map, class_colors, stats, output_path):
    """Create an interactive HTML map with classification results.

    Args:
        lat: Center latitude.
        lon: Center longitude.
        label_map: 2D array of class indices.
        class_colors: List of hex color strings.
        stats: Coverage statistics dictionary.
        output_path: Path to save the HTML file.
    """
    # Create base map centered on the study area
    m = folium.Map(location=[lat, lon], zoom_start=12, tiles="OpenStreetMap")

    # Add satellite base layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Create colored overlay from label map
    overlay = np.zeros((*label_map.shape, 4), dtype=np.uint8)

    color_map = {
        0: (34, 139, 34, 180),    # Forest - forest green
        1: (144, 238, 144, 180),  # Grassland - light green
        2: (255, 215, 0, 180),    # Agriculture - gold
        3: (128, 128, 128, 180),  # Urban - gray
        4: (30, 144, 255, 180),   # Water - dodger blue
        5: (139, 69, 19, 180),    # Bare soil - saddle brown
    }

    for class_idx, color in color_map.items():
        mask = label_map == class_idx
        overlay[mask] = color

    # Calculate bounds for the overlay
    buffer_deg = 15 / 111.0
    bounds = [
        [lat - buffer_deg, lon - buffer_deg],
        [lat + buffer_deg, lon + buffer_deg],
    ]

    # Add classified overlay
    folium.raster_layers.ImageOverlay(
        image=overlay,
        bounds=bounds,
        opacity=0.7,
        name="Land Cover Classification",
    ).add_to(m)

    # Build legend HTML
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;margin:3px;">'
        f'<span style="width:20px;height:20px;background:{color};margin-right:8px;border:1px solid #333;"></span>'
        f'<span>{name}</span></div>'
        for name, color in class_colors.items()
    )

    stats_rows = "".join(
        f'<tr><td style="padding:4px 12px;">{name}</td>'
        f'<td style="padding:4px 12px;text-align:right;">{info["percentage"]}%</td></tr>'
        for name, info in stats.items()
    )

    legend_html = f"""
    <div style="position:fixed;bottom:50px;left:50px;z-index:1000;background:white;
                padding:15px;border-radius:8px;box-shadow:0 0 15px rgba(0,0,0,0.3);
                font-family:Arial;max-width:250px;">
        <h4 style="margin:0 0 10px 0;">Land Cover - Armenia, Quindio</h4>
        <div style="margin-bottom:10px;">{legend_items}</div>
        <table style="border-top:1px solid #ccc;width:100%;font-size:12px;">
            <tr style="border-bottom:1px solid #eee;">
                <th style="padding:4px;text-align:left;">Class</th>
                <th style="padding:4px;text-align:right;">%</th>
            </tr>
            {stats_rows}
        </table>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map
    m.save(output_path)
    print(f"Map saved to: {output_path}")
    return m
