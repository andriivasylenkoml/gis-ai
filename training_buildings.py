import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, box
from shapely.affinity import rotate, translate


def generate_quality_synthetic_buildings(
    n_buildings: int = 500,
    center_lon: float = 11.57,
    center_lat: float = 48.13,
    metric_crs: str = "EPSG:3857",
    output_csv: str = "training_projects.csv",
    output_geojson: str = "training_buildings.geojson",
    random_seed: int = 42
):
    """
    Generate quality synthetic building data.

    Output CSV columns:
    - building_id
    - geometry as WKT
    - floors
    - damage_level
    - building_type
    - zoning_category

    Output GeoJSON:
    - same data, but with real geometry column.
    """

    np.random.seed(random_seed)

    # ------------------------------------------------------------
    # 1. Define realistic categories
    # ------------------------------------------------------------

    building_types = [
        "residential",
        "commercial",
        "industrial",
        "public",
        "cultural"
    ]

    building_type_probabilities = [
        0.55,  # residential
        0.18,  # commercial
        0.12,  # industrial
        0.10,  # public
        0.05   # cultural
    ]

    zoning_by_type = {
        "residential": ["residential", "mixed_use"],
        "commercial": ["commercial", "mixed_use"],
        "industrial": ["industrial"],
        "public": ["public", "mixed_use"],
        "cultural": ["historical", "public"]
    }

    # ------------------------------------------------------------
    # 2. Convert city center to metric CRS
    # ------------------------------------------------------------

    center_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(center_lon, center_lat)],
        crs="EPSG:4326"
    ).to_crs(metric_crs)

    center_x = center_gdf.geometry.iloc[0].x
    center_y = center_gdf.geometry.iloc[0].y

    # ------------------------------------------------------------
    # 3. Generate buildings on a loose grid
    #    This avoids excessive overlaps.
    # ------------------------------------------------------------

    grid_size = int(np.ceil(np.sqrt(n_buildings)))
    spacing = 90  # meters between building centers

    rows = []
    geometries_metric = []

    building_id = 1

    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            if building_id > n_buildings:
                break

            # Random city-like displacement
            x = center_x + (col_idx - grid_size / 2) * spacing + np.random.uniform(-20, 20)
            y = center_y + (row_idx - grid_size / 2) * spacing + np.random.uniform(-20, 20)

            # ----------------------------------------------------
            # 4. Generate building type
            # ----------------------------------------------------

            building_type = np.random.choice(
                building_types,
                p=building_type_probabilities
            )

            zoning_category = np.random.choice(zoning_by_type[building_type])

            # ----------------------------------------------------
            # 5. Generate realistic footprint size and floors
            # ----------------------------------------------------

            if building_type == "residential":
                width = np.random.uniform(8, 28)
                depth = np.random.uniform(8, 35)
                floors = np.random.choice(
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    p=[0.12, 0.20, 0.22, 0.18, 0.12, 0.08, 0.05, 0.03]
                )

            elif building_type == "commercial":
                width = np.random.uniform(18, 55)
                depth = np.random.uniform(18, 60)
                floors = np.random.choice(
                    [1, 2, 3, 4, 5, 6, 8, 10],
                    p=[0.08, 0.16, 0.20, 0.20, 0.14, 0.10, 0.07, 0.05]
                )

            elif building_type == "industrial":
                width = np.random.uniform(35, 120)
                depth = np.random.uniform(25, 100)
                floors = np.random.choice(
                    [1, 2, 3],
                    p=[0.65, 0.25, 0.10]
                )

            elif building_type == "public":
                width = np.random.uniform(20, 75)
                depth = np.random.uniform(20, 80)
                floors = np.random.choice(
                    [1, 2, 3, 4, 5],
                    p=[0.15, 0.25, 0.25, 0.20, 0.15]
                )

            else:  # cultural
                width = np.random.uniform(15, 50)
                depth = np.random.uniform(15, 60)
                floors = np.random.choice(
                    [1, 2, 3, 4],
                    p=[0.25, 0.35, 0.25, 0.15]
                )

            # ----------------------------------------------------
            # 6. Generate rectangular footprint in meters
            # ----------------------------------------------------

            geom = box(
                -width / 2,
                -depth / 2,
                width / 2,
                depth / 2
            )

            angle = np.random.uniform(-20, 20)
            geom = rotate(geom, angle, origin=(0, 0))
            geom = translate(geom, xoff=x, yoff=y)

            # ----------------------------------------------------
            # 7. Generate damage level with realistic distribution
            # ----------------------------------------------------

            # Most buildings have low or medium damage.
            # Some have severe damage.
            damage_group = np.random.choice(
                ["low", "medium", "high", "destroyed"],
                p=[0.45, 0.30, 0.18, 0.07]
            )

            if damage_group == "low":
                damage_level = np.random.uniform(0.05, 0.30)
            elif damage_group == "medium":
                damage_level = np.random.uniform(0.30, 0.60)
            elif damage_group == "high":
                damage_level = np.random.uniform(0.60, 0.85)
            else:
                damage_level = np.random.uniform(0.85, 1.00)

            # Historical/cultural buildings may be assigned slightly lower demolition-like damage
            # to avoid too many impossible cases.
            if zoning_category == "historical":
                damage_level = min(damage_level, np.random.uniform(0.05, 0.80))

            rows.append({
                "building_id": building_id,
                "floors": int(floors),
                "damage_level": round(float(damage_level), 3),
                "building_type": building_type,
                "zoning_category": zoning_category
            })

            geometries_metric.append(geom)
            building_id += 1

    # ------------------------------------------------------------
    # 8. Create GeoDataFrame in metric CRS
    # ------------------------------------------------------------

    gdf_metric = gpd.GeoDataFrame(
        rows,
        geometry=geometries_metric,
        crs=metric_crs
    )

    # Convert to EPSG:4326 for GeoJSON compatibility
    gdf_wgs84 = gdf_metric.to_crs("EPSG:4326")

    # ------------------------------------------------------------
    # 9. Save GeoJSON
    # ------------------------------------------------------------

    gdf_wgs84.to_file(output_geojson, driver="GeoJSON")

    # ------------------------------------------------------------
    # 10. Save CSV with geometry as WKT
    # ------------------------------------------------------------

    csv_df = gdf_wgs84.copy()
    csv_df["geometry"] = csv_df.geometry.to_wkt()

    csv_df.to_csv(output_csv, index=False)

    print(f"Created CSV: {output_csv}")
    print(f"Created GeoJSON: {output_geojson}")
    print(csv_df.head())

    return csv_df, gdf_wgs84


if __name__ == "__main__":
    generate_quality_synthetic_buildings(
        n_buildings=500,
        center_lon=11.57,
        center_lat=48.13,
        metric_crs="EPSG:3857",
        output_csv="training_projects.csv",
        output_geojson="training_buildings.geojson",
        random_seed=42
    )