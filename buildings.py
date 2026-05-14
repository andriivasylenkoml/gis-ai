import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, box
from shapely.affinity import rotate, translate


def generate_building_footprint(
    x: float,
    y: float,
    building_type: str,
    output_csv: str = "buildings.csv",
):
    """
    Generate a synthetic building footprint in metric coordinates.
    The footprint is generated in meters and later converted to EPSG:4326.
    """

    if building_type == "residential":
        width = np.random.uniform(8, 28)
        depth = np.random.uniform(8, 35)

    elif building_type == "commercial":
        width = np.random.uniform(18, 60)
        depth = np.random.uniform(18, 65)

    elif building_type == "industrial":
        width = np.random.uniform(35, 130)
        depth = np.random.uniform(25, 110)

    elif building_type == "public":
        width = np.random.uniform(20, 80)
        depth = np.random.uniform(20, 85)

    elif building_type == "cultural":
        width = np.random.uniform(15, 55)
        depth = np.random.uniform(15, 65)

    else:
        width = np.random.uniform(10, 40)
        depth = np.random.uniform(10, 40)

    # Basic rectangular footprint centered at (0, 0)
    footprint = box(
        -width / 2,
        -depth / 2,
        width / 2,
        depth / 2
    )

    # Random rotation for more realistic urban geometry
    angle = np.random.uniform(-25, 25)
    footprint = rotate(footprint, angle, origin=(0, 0))

    # Move to actual location
    footprint = translate(footprint, xoff=x, yoff=y)

    return footprint


def generate_floors(building_type: str) -> int:
    """
    Generate realistic number of floors depending on building type.
    """

    if building_type == "residential":
        return int(np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            p=[0.10, 0.18, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02]
        ))

    if building_type == "commercial":
        return int(np.random.choice(
            [1, 2, 3, 4, 5, 6, 8, 10, 12],
            p=[0.08, 0.14, 0.18, 0.18, 0.14, 0.10, 0.08, 0.06, 0.04]
        ))

    if building_type == "industrial":
        return int(np.random.choice(
            [1, 2, 3],
            p=[0.70, 0.23, 0.07]
        ))

    if building_type == "public":
        return int(np.random.choice(
            [1, 2, 3, 4, 5, 6],
            p=[0.12, 0.22, 0.25, 0.18, 0.13, 0.10]
        ))

    if building_type == "cultural":
        return int(np.random.choice(
            [1, 2, 3, 4],
            p=[0.25, 0.35, 0.25, 0.15]
        ))

    return int(np.random.randint(1, 5))


def generate_damage_level(zoning_category: str) -> float:
    """
    Generate damage level d_i in range [0, 1].
    Most buildings have low/medium damage, fewer are destroyed.
    """

    damage_class = np.random.choice(
        ["low", "medium", "high", "destroyed"],
        p=[0.45, 0.30, 0.18, 0.07]
    )

    if damage_class == "low":
        damage = np.random.uniform(0.05, 0.30)
    elif damage_class == "medium":
        damage = np.random.uniform(0.30, 0.60)
    elif damage_class == "high":
        damage = np.random.uniform(0.60, 0.85)
    else:
        damage = np.random.uniform(0.85, 1.00)

    # Historical buildings are usually treated more carefully in scenarios.
    # This avoids too many unrealistic "destroyed historical building" cases.
    if zoning_category == "historical":
        damage = min(damage, np.random.uniform(0.05, 0.80))

    return round(float(damage), 3)


def generate_buildings_geojson(
    output_path: str = "buildings.geojson",
    n_buildings: int = 100,
    center_lon: float = 11.57,
    center_lat: float = 48.13,
    metric_crs: str = "EPSG:3857",
    output_csv: str = "buildings.csv",
    random_seed: int = 42
):
    """
    Generate synthetic buildings.geojson.

    Output fields:
    - building_id
    - geometry
    - floors
    - damage_level
    - building_type
    - zoning_category

    Geometry is saved in EPSG:4326, which is standard for GeoJSON.
    """

    np.random.seed(random_seed)

    building_types = [
        "residential",
        "commercial",
        "industrial",
        "public",
        "cultural"
    ]

    building_type_probabilities = [
        0.55,
        0.18,
        0.12,
        0.10,
        0.05
    ]

    zoning_by_type = {
        "residential": ["residential", "mixed_use"],
        "commercial": ["commercial", "mixed_use"],
        "industrial": ["industrial"],
        "public": ["public", "mixed_use"],
        "cultural": ["historical", "public"]
    }

    # Convert center point from EPSG:4326 to metric CRS
    center_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(center_lon, center_lat)],
        crs="EPSG:4326"
    ).to_crs(metric_crs)

    center_x = center_gdf.geometry.iloc[0].x
    center_y = center_gdf.geometry.iloc[0].y

    # Generate loose grid to reduce overlap
    grid_size = int(np.ceil(np.sqrt(n_buildings)))
    spacing = 120  # meters between approximate building centers

    rows = []
    geometries = []

    building_id = 1

    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            if building_id > n_buildings:
                break

            # Urban-like grid with random jitter
            x = (
                center_x
                + (col_idx - grid_size / 2) * spacing
                + np.random.uniform(-30, 30)
            )

            y = (
                center_y
                + (row_idx - grid_size / 2) * spacing
                + np.random.uniform(-30, 30)
            )

            building_type = np.random.choice(
                building_types,
                p=building_type_probabilities
            )

            zoning_category = np.random.choice(
                zoning_by_type[building_type]
            )

            floors = generate_floors(building_type)

            damage_level = generate_damage_level(zoning_category)

            geometry = generate_building_footprint(
                x=x,
                y=y,
                building_type=building_type
            )

            rows.append({
                "building_id": building_id,
                "floors": floors,
                "damage_level": damage_level,
                "building_type": building_type,
                "zoning_category": zoning_category
            })

            geometries.append(geometry)

            building_id += 1

    # Create GeoDataFrame in metric CRS
    buildings_metric = gpd.GeoDataFrame(
        rows,
        geometry=geometries,
        crs=metric_crs
    )

    # Fix invalid geometries if any appear
    buildings_metric["geometry"] = buildings_metric.geometry.make_valid()

    # Convert to EPSG:4326 for GeoJSON output
    buildings_wgs84 = buildings_metric.to_crs("EPSG:4326")

    # Save to GeoJSON
    buildings_wgs84.to_file(output_path, driver="GeoJSON")

    csv_df = buildings_wgs84.copy()
    csv_df["geometry"] = csv_df.geometry.to_wkt()

    csv_df.to_csv(output_csv, index=False)

    print(f"Created: {output_path}")
    print(buildings_wgs84.head())

    return buildings_wgs84


if __name__ == "__main__":
    generate_buildings_geojson(
        output_path="buildings.geojson",
        n_buildings=100,
        center_lon=11.57,
        center_lat=48.13,
        metric_crs="EPSG:3857",
        output_csv="buildings.csv",
        random_seed=42
    )
