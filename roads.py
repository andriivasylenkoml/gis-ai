import numpy as np
import geopandas as gpd

from shapely.geometry import Point, LineString


def create_jittered_line(points, jitter: float = 8.0):
    """
    Adds small random displacement to intermediate points
    to make synthetic roads look less perfectly artificial.

    Coordinates are in meters.
    """

    jittered = []

    for idx, (x, y) in enumerate(points):
        # Keep endpoints more stable
        if idx == 0 or idx == len(points) - 1:
            jittered.append((x, y))
        else:
            jittered.append((
                x + np.random.uniform(-jitter, jitter),
                y + np.random.uniform(-jitter, jitter)
            ))

    return LineString(jittered)


def generate_roads_geojson(
    output_path: str = "roads.geojson",
    output_csv: str = "buildings.csv",
    center_lon: float = 11.57,
    center_lat: float = 48.13,
    metric_crs: str = "EPSG:3857",
    random_seed: int = 42,
    
):
    """
    Generate synthetic roads.geojson.

    Output fields:
    - road_id
    - highway
    - geometry

    Geometry is generated in a metric CRS and saved in EPSG:4326,
    which is the standard coordinate system for GeoJSON.
    """

    np.random.seed(random_seed)

    # ------------------------------------------------------------
    # 1. Convert center point from EPSG:4326 to metric CRS
    # ------------------------------------------------------------

    center_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(center_lon, center_lat)],
        crs="EPSG:4326"
    ).to_crs(metric_crs)

    center_x = center_gdf.geometry.iloc[0].x
    center_y = center_gdf.geometry.iloc[0].y

    roads = []
    road_id = 1

    # ------------------------------------------------------------
    # 2. Primary roads
    #    Long roads crossing the whole area.
    # ------------------------------------------------------------

    primary_roads = [
        [
            (center_x - 1800, center_y - 900),
            (center_x - 900, center_y - 400),
            (center_x, center_y),
            (center_x + 900, center_y + 350),
            (center_x + 1800, center_y + 900)
        ],
        [
            (center_x - 1700, center_y + 850),
            (center_x - 800, center_y + 400),
            (center_x + 100, center_y + 100),
            (center_x + 850, center_y - 350),
            (center_x + 1700, center_y - 850)
        ]
    ]

    for road_points in primary_roads:
        roads.append({
            "road_id": road_id,
            "highway": "primary",
            "geometry": create_jittered_line(road_points, jitter=20)
        })
        road_id += 1

    # ------------------------------------------------------------
    # 3. Secondary roads
    #    Medium roads forming a rough city grid.
    # ------------------------------------------------------------

    grid_extent = 1500
    spacing_secondary = 500

    # Vertical secondary roads
    for x_offset in range(-grid_extent, grid_extent + 1, spacing_secondary):
        points = [
            (center_x + x_offset, center_y - grid_extent),
            (center_x + x_offset + np.random.uniform(-50, 50), center_y),
            (center_x + x_offset, center_y + grid_extent)
        ]

        roads.append({
            "road_id": road_id,
            "highway": "secondary",
            "geometry": create_jittered_line(points, jitter=18)
        })
        road_id += 1

    # Horizontal secondary roads
    for y_offset in range(-grid_extent, grid_extent + 1, spacing_secondary):
        points = [
            (center_x - grid_extent, center_y + y_offset),
            (center_x, center_y + y_offset + np.random.uniform(-50, 50)),
            (center_x + grid_extent, center_y + y_offset)
        ]

        roads.append({
            "road_id": road_id,
            "highway": "secondary",
            "geometry": create_jittered_line(points, jitter=18)
        })
        road_id += 1

    # ------------------------------------------------------------
    # 4. Residential roads
    #    Smaller local streets between secondary roads.
    # ------------------------------------------------------------

    spacing_residential = 250

    # Vertical residential roads
    for x_offset in range(-grid_extent + 125, grid_extent, spacing_residential):
        if x_offset % spacing_secondary == 0:
            continue

        y_start = center_y - grid_extent + np.random.uniform(-80, 80)
        y_end = center_y + grid_extent + np.random.uniform(-80, 80)

        points = [
            (center_x + x_offset, y_start),
            (center_x + x_offset + np.random.uniform(-30, 30), center_y),
            (center_x + x_offset, y_end)
        ]

        roads.append({
            "road_id": road_id,
            "highway": "residential",
            "geometry": create_jittered_line(points, jitter=10)
        })
        road_id += 1

    # Horizontal residential roads
    for y_offset in range(-grid_extent + 125, grid_extent, spacing_residential):
        if y_offset % spacing_secondary == 0:
            continue

        x_start = center_x - grid_extent + np.random.uniform(-80, 80)
        x_end = center_x + grid_extent + np.random.uniform(-80, 80)

        points = [
            (x_start, center_y + y_offset),
            (center_x, center_y + y_offset + np.random.uniform(-30, 30)),
            (x_end, center_y + y_offset)
        ]

        roads.append({
            "road_id": road_id,
            "highway": "residential",
            "geometry": create_jittered_line(points, jitter=10)
        })
        road_id += 1

    # ------------------------------------------------------------
    # 5. Service roads
    #    Short roads near buildings, yards, industrial zones.
    # ------------------------------------------------------------

    n_service_roads = 30

    for _ in range(n_service_roads):
        start_x = center_x + np.random.uniform(-grid_extent, grid_extent)
        start_y = center_y + np.random.uniform(-grid_extent, grid_extent)

        length = np.random.uniform(80, 250)
        angle = np.random.uniform(0, 2 * np.pi)

        end_x = start_x + length * np.cos(angle)
        end_y = start_y + length * np.sin(angle)

        middle_x = (start_x + end_x) / 2 + np.random.uniform(-20, 20)
        middle_y = (start_y + end_y) / 2 + np.random.uniform(-20, 20)

        points = [
            (start_x, start_y),
            (middle_x, middle_y),
            (end_x, end_y)
        ]

        roads.append({
            "road_id": road_id,
            "highway": "service",
            "geometry": create_jittered_line(points, jitter=5)
        })
        road_id += 1

    # ------------------------------------------------------------
    # 6. Create GeoDataFrame in metric CRS
    # ------------------------------------------------------------

    roads_metric = gpd.GeoDataFrame(
        roads,
        geometry="geometry",
        crs=metric_crs
    )

    # Remove empty or invalid geometries if any appear
    roads_metric = roads_metric[
        roads_metric.geometry.notnull() & ~roads_metric.geometry.is_empty
    ].copy()

    # ------------------------------------------------------------
    # 7. Convert to EPSG:4326 for GeoJSON
    # ------------------------------------------------------------

    roads_wgs84 = roads_metric.to_crs("EPSG:4326")

    # ------------------------------------------------------------
    # 8. Save GeoJSON
    # ------------------------------------------------------------

    roads_wgs84.to_file(output_path, driver="GeoJSON")
    
    csv_df = roads_wgs84.copy()
    csv_df["geometry"] = csv_df.geometry.to_wkt()

    csv_df.to_csv(output_csv, index=False)

    print(f"Created: {output_path}")
    print(roads_wgs84.head())
    print(f"Total roads: {len(roads_wgs84)}")

    return roads_wgs84


if __name__ == "__main__":
    generate_roads_geojson(
        output_path="roads.geojson",
        center_lon=11.57,
        center_lat=48.13,
        metric_crs="EPSG:3857",
        output_csv="roads.csv",
        random_seed=42
    )