import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, box, LineString
from shapely.affinity import rotate, translate


# ============================================================
# 0. Настройки
# ============================================================

ACTIONS = ["repair", "rebuild", "demolish", "defer"]

RESOURCE_COEFFICIENTS = {
    "repair": {
        "labor_hours_per_m2": 4.0,
        "concrete_m3_per_m2": 0.03,
        "steel_tons_per_m2": 0.002,
        "equipment_hours_per_m2": 0.02,
    },
    "rebuild": {
        "labor_hours_per_m2": 12.0,
        "concrete_m3_per_m2": 0.25,
        "steel_tons_per_m2": 0.015,
        "equipment_hours_per_m2": 0.10,
    },
    "demolish": {
        "labor_hours_per_m2": 2.0,
        "concrete_m3_per_m2": 0.00,
        "steel_tons_per_m2": 0.000,
        "equipment_hours_per_m2": 0.08,
    },
    "defer": {
        "labor_hours_per_m2": 0.0,
        "concrete_m3_per_m2": 0.0,
        "steel_tons_per_m2": 0.0,
        "equipment_hours_per_m2": 0.0,
    },
}


# ============================================================
# 1. Генерация дорог
# ============================================================

def generate_roads_geojson(
    output_path: str = "roads.geojson",
    center_lon: float = 11.57,
    center_lat: float = 48.13,
    metric_crs: str = "EPSG:3857",
    random_seed: int = 42
) -> gpd.GeoDataFrame:
    """
    Генерирует roads.geojson.

    Поля:
    - road_id
    - highway
    - geometry
    """

    np.random.seed(random_seed)

    center = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(center_lon, center_lat)],
        crs="EPSG:4326"
    ).to_crs(metric_crs)

    center_x = center.geometry.iloc[0].x
    center_y = center.geometry.iloc[0].y

    roads = []
    road_id = 1

    # Главные дороги
    main_roads = [
        LineString([
            (center_x - 2500, center_y - 1200),
            (center_x, center_y),
            (center_x + 2500, center_y + 1200),
        ]),
        LineString([
            (center_x - 2500, center_y + 1200),
            (center_x, center_y + 100),
            (center_x + 2500, center_y - 1200),
        ]),
    ]

    for geom in main_roads:
        roads.append({
            "road_id": road_id,
            "highway": "primary",
            "geometry": geom
        })
        road_id += 1

    # Второстепенная сетка
    extent = 2200
    spacing = 500

    for offset in range(-extent, extent + 1, spacing):
        roads.append({
            "road_id": road_id,
            "highway": "secondary",
            "geometry": LineString([
                (center_x + offset, center_y - extent),
                (center_x + offset + np.random.uniform(-60, 60), center_y),
                (center_x + offset, center_y + extent),
            ])
        })
        road_id += 1

        roads.append({
            "road_id": road_id,
            "highway": "secondary",
            "geometry": LineString([
                (center_x - extent, center_y + offset),
                (center_x, center_y + offset + np.random.uniform(-60, 60)),
                (center_x + extent, center_y + offset),
            ])
        })
        road_id += 1

    # Локальные дороги
    for _ in range(35):
        x = center_x + np.random.uniform(-extent, extent)
        y = center_y + np.random.uniform(-extent, extent)

        length = np.random.uniform(120, 350)
        angle = np.random.uniform(0, 2 * np.pi)

        geom = LineString([
            (x, y),
            (
                x + length * np.cos(angle),
                y + length * np.sin(angle)
            )
        ])

        roads.append({
            "road_id": road_id,
            "highway": "residential",
            "geometry": geom
        })
        road_id += 1

    roads_metric = gpd.GeoDataFrame(
        roads,
        geometry="geometry",
        crs=metric_crs
    )

    roads_wgs84 = roads_metric.to_crs("EPSG:4326")
    roads_wgs84.to_file(output_path, driver="GeoJSON")

    print(f"Created: {output_path}")
    print(f"Roads count: {len(roads_wgs84)}")

    return roads_wgs84


# ============================================================
# 2. Генерация базовых зданий
# ============================================================

def generate_building_base_geojson(
    output_path: str,
    n_buildings: int,
    center_lon: float,
    center_lat: float,
    metric_crs: str = "EPSG:3857",
    random_seed: int = 42,
    start_building_id: int = 1
) -> gpd.GeoDataFrame:
    """
    Генерирует GeoJSON со зданиями.

    Поля:
    - building_id
    - geometry
    - floors
    - damage_level
    - building_type
    - zoning_category
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

    center = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(center_lon, center_lat)],
        crs="EPSG:4326"
    ).to_crs(metric_crs)

    center_x = center.geometry.iloc[0].x
    center_y = center.geometry.iloc[0].y

    grid_size = int(np.ceil(np.sqrt(n_buildings)))
    spacing = 110

    rows = []
    geometries = []

    building_id = start_building_id

    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            if len(rows) >= n_buildings:
                break

            x = (
                center_x
                + (col_idx - grid_size / 2) * spacing
                + np.random.uniform(-25, 25)
            )

            y = (
                center_y
                + (row_idx - grid_size / 2) * spacing
                + np.random.uniform(-25, 25)
            )

            building_type = np.random.choice(
                building_types,
                p=building_type_probabilities
            )

            zoning_category = np.random.choice(zoning_by_type[building_type])

            width, depth, floors = generate_size_and_floors(building_type)

            geom = box(
                -width / 2,
                -depth / 2,
                width / 2,
                depth / 2
            )

            geom = rotate(
                geom,
                np.random.uniform(-25, 25),
                origin=(0, 0)
            )

            geom = translate(
                geom,
                xoff=x,
                yoff=y
            )

            damage_level = generate_damage_level(zoning_category)

            rows.append({
                "building_id": building_id,
                "floors": int(floors),
                "damage_level": round(float(damage_level), 3),
                "building_type": building_type,
                "zoning_category": zoning_category
            })

            geometries.append(geom)
            building_id += 1

    gdf_metric = gpd.GeoDataFrame(
        rows,
        geometry=geometries,
        crs=metric_crs
    )

    gdf_metric["geometry"] = gdf_metric.geometry.make_valid()

    gdf_wgs84 = gdf_metric.to_crs("EPSG:4326")
    gdf_wgs84.to_file(output_path, driver="GeoJSON")

    print(f"Created: {output_path}")
    print(f"Buildings count: {len(gdf_wgs84)}")

    return gdf_wgs84


def generate_size_and_floors(building_type: str) -> tuple[float, float, int]:
    """
    Генерирует размер пятна застройки и этажность.
    """

    if building_type == "residential":
        width = np.random.uniform(8, 32)
        depth = np.random.uniform(8, 38)
        floors = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            p=[0.10, 0.18, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02]
        )

    elif building_type == "commercial":
        width = np.random.uniform(18, 65)
        depth = np.random.uniform(18, 70)
        floors = np.random.choice(
            [1, 2, 3, 4, 5, 6, 8, 10, 12],
            p=[0.08, 0.14, 0.18, 0.18, 0.14, 0.10, 0.08, 0.06, 0.04]
        )

    elif building_type == "industrial":
        width = np.random.uniform(35, 140)
        depth = np.random.uniform(25, 120)
        floors = np.random.choice(
            [1, 2, 3],
            p=[0.70, 0.23, 0.07]
        )

    elif building_type == "public":
        width = np.random.uniform(20, 85)
        depth = np.random.uniform(20, 90)
        floors = np.random.choice(
            [1, 2, 3, 4, 5, 6],
            p=[0.12, 0.22, 0.25, 0.18, 0.13, 0.10]
        )

    else:  # cultural
        width = np.random.uniform(15, 60)
        depth = np.random.uniform(15, 70)
        floors = np.random.choice(
            [1, 2, 3, 4],
            p=[0.25, 0.35, 0.25, 0.15]
        )

    return float(width), float(depth), int(floors)


def generate_damage_level(zoning_category: str) -> float:
    """
    Генерирует damage_level в диапазоне [0, 1].
    """

    damage_group = np.random.choice(
        ["low", "medium", "high", "destroyed"],
        p=[0.45, 0.30, 0.18, 0.07]
    )

    if damage_group == "low":
        damage = np.random.uniform(0.05, 0.30)
    elif damage_group == "medium":
        damage = np.random.uniform(0.30, 0.60)
    elif damage_group == "high":
        damage = np.random.uniform(0.60, 0.85)
    else:
        damage = np.random.uniform(0.85, 1.00)

    if zoning_category == "historical":
        damage = min(damage, np.random.uniform(0.05, 0.80))

    return float(np.clip(damage, 0.0, 1.0))


# ============================================================
# 3. GIS-признаки: area_m2, perimeter_m, access_index
# ============================================================

def add_gis_features(
    buildings_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame | None = None,
    metric_crs: str = "EPSG:3857"
) -> gpd.GeoDataFrame:
    """
    Добавляет к зданиям:
    - area_m2
    - perimeter_m
    - distance_to_road_m
    - access_index

    Важно: access_index нормализуется в диапазоне [0, 1].
    """

    buildings = buildings_gdf.to_crs(metric_crs).copy()

    buildings["geometry"] = buildings.geometry.make_valid()
    buildings["area_m2"] = buildings.geometry.area
    buildings["perimeter_m"] = buildings.geometry.length

    if roads_gdf is not None:
        roads = roads_gdf.to_crs(metric_crs).copy()
        roads["geometry"] = roads.geometry.make_valid()

        try:
            roads_union = roads.geometry.union_all()
        except AttributeError:
            roads_union = roads.geometry.unary_union

        buildings["distance_to_road_m"] = buildings.geometry.distance(roads_union)

        d_min = buildings["distance_to_road_m"].min()
        d_max = buildings["distance_to_road_m"].max()

        if d_max > d_min:
            buildings["access_index"] = 1 - (
                (buildings["distance_to_road_m"] - d_min) / (d_max - d_min)
            )
        else:
            buildings["access_index"] = 1.0
    else:
        buildings["distance_to_road_m"] = 0.0
        buildings["access_index"] = 1.0

    buildings["access_index"] = buildings["access_index"].clip(0, 1)

    return buildings


# ============================================================
# 4. Ресурсы
# ============================================================

def add_resource_needs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет:
    - labor_hours
    - concrete_m3
    - steel_tons
    - equipment_hours
    """

    df = df.copy()

    labor_hours = []
    concrete_m3 = []
    steel_tons = []
    equipment_hours = []

    for _, row in df.iterrows():
        action = row["action"]
        area = float(row["area_m2"])
        floors = int(row["floors"])
        damage = float(row["damage_level"])
        access = float(row["access_index"])

        coeff = RESOURCE_COEFFICIENTS[action]

        damage_factor = 1 + 0.6 * damage
        floors_factor = 1 + 0.025 * floors
        access_factor = 1 + 0.35 * (1 - access)

        factor = damage_factor * floors_factor * access_factor

        labor_hours.append(area * coeff["labor_hours_per_m2"] * factor)
        concrete_m3.append(area * coeff["concrete_m3_per_m2"] * factor)
        steel_tons.append(area * coeff["steel_tons_per_m2"] * factor)
        equipment_hours.append(area * coeff["equipment_hours_per_m2"] * factor)

    df["labor_hours"] = np.round(labor_hours, 3)
    df["concrete_m3"] = np.round(concrete_m3, 3)
    df["steel_tons"] = np.round(steel_tons, 3)
    df["equipment_hours"] = np.round(equipment_hours, 3)

    return df


# ============================================================
# 5. Стоимость и время
# ============================================================

def estimate_cost_and_duration(row: pd.Series) -> tuple[float, float]:
    """
    Синтетическая функция для создания исторических cost и duration_days.
    """

    action = row["action"]
    area = float(row["area_m2"])
    floors = int(row["floors"])
    damage = float(row["damage_level"])
    access = float(row["access_index"])
    zoning = str(row["zoning_category"])
    btype = str(row["building_type"])

    if action == "defer":
        cost = area * 10 * (1 + 0.2 * (1 - access))
        duration = 1.0
        return round(cost, 2), round(duration, 1)

    base_cost_per_m2 = {
        "repair": 250,
        "rebuild": 950,
        "demolish": 120,
    }[action]

    base_days_per_m2 = {
        "repair": 0.20,
        "rebuild": 0.65,
        "demolish": 0.12,
    }[action]

    zoning_factor = {
        "residential": 1.00,
        "mixed_use": 1.10,
        "commercial": 1.20,
        "industrial": 1.15,
        "public": 1.25,
        "historical": 1.50,
    }.get(zoning, 1.00)

    building_factor = {
        "residential": 1.00,
        "commercial": 1.15,
        "industrial": 1.25,
        "public": 1.20,
        "cultural": 1.40,
    }.get(btype, 1.00)

    damage_factor = 1 + 0.9 * damage
    access_factor = 1 + 0.4 * (1 - access)
    floors_factor = 1 + 0.025 * floors

    noise_cost = np.random.normal(1.0, 0.07)
    noise_time = np.random.normal(1.0, 0.09)

    cost = (
        area
        * base_cost_per_m2
        * zoning_factor
        * building_factor
        * damage_factor
        * access_factor
        * floors_factor
        * noise_cost
    )

    duration = (
        area
        * base_days_per_m2
        * damage_factor
        * access_factor
        * floors_factor
        * noise_time
    )

    return round(max(cost, 0), 2), round(max(duration, 1), 1)


# ============================================================
# 6. Training CSV
# ============================================================

def create_training_projects_csv(
    training_buildings_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    output_csv: str = "training_projects.csv",
    metric_crs: str = "EPSG:3857",
    include_defer: bool = True
) -> pd.DataFrame:
    """
    Создаёт training_projects.csv для CatBoost.

    CSV содержит признаки + известные targets:
    - area_m2
    - floors
    - damage_level
    - access_index
    - zoning_category
    - building_type
    - action
    - labor_hours
    - concrete_m3
    - steel_tons
    - equipment_hours
    - cost
    - duration_days
    """

    buildings = add_gis_features(
        training_buildings_gdf,
        roads_gdf=roads_gdf,
        metric_crs=metric_crs
    )

    actions = ACTIONS if include_defer else ["repair", "rebuild", "demolish"]

    rows = []

    for _, b in buildings.iterrows():
        for action in actions:
            rows.append({
                "building_id": b["building_id"],
                "area_m2": round(float(b["area_m2"]), 3),
                "perimeter_m": round(float(b["perimeter_m"]), 3),
                "floors": int(b["floors"]),
                "damage_level": round(float(b["damage_level"]), 3),
                "access_index": round(float(b["access_index"]), 3),
                "zoning_category": str(b["zoning_category"]),
                "building_type": str(b["building_type"]),
                "action": action
            })

    df = pd.DataFrame(rows)
    df = add_resource_needs(df)

    costs = []
    durations = []

    for _, row in df.iterrows():
        cost, duration = estimate_cost_and_duration(row)
        costs.append(cost)
        durations.append(duration)

    df["cost"] = costs
    df["duration_days"] = durations

    ordered_columns = [
        "building_id",
        "area_m2",
        "perimeter_m",
        "floors",
        "damage_level",
        "access_index",
        "zoning_category",
        "building_type",
        "action",
        "labor_hours",
        "concrete_m3",
        "steel_tons",
        "equipment_hours",
        "cost",
        "duration_days",
    ]

    df = df[ordered_columns]

    df.to_csv(output_csv, index=False)

    print(f"Created: {output_csv}")
    print(f"Training rows: {len(df)}")
    print(df.head())

    return df


# ============================================================
# 7. Главный запуск
# ============================================================

def main():
    """
    Создаёт 4 файла:

    1. training_projects.csv
       Обучающая таблица для CatBoost.

    2. training_buildings.geojson
       GIS-слой обучающих зданий для проверки.

    3. buildings.geojson
       Новые здания для прогноза.

    4. roads.geojson
       Дороги для расчёта access_index.
    """

    metric_crs = "EPSG:3857"

    # Одна дорожная сеть, чтобы признаки access_index были сопоставимы.
    roads = generate_roads_geojson(
        output_path="roads.geojson",
        center_lon=11.57,
        center_lat=48.13,
        metric_crs=metric_crs,
        random_seed=10
    )

    # Исторические здания для обучения.
    training_buildings = generate_building_base_geojson(
        output_path="training_buildings.geojson",
        n_buildings=500,
        center_lon=11.565,
        center_lat=48.125,
        metric_crs=metric_crs,
        random_seed=42,
        start_building_id=1
    )

    # Новые здания для прогноза.
    prediction_buildings = generate_building_base_geojson(
        output_path="buildings.geojson",
        n_buildings=100,
        center_lon=11.575,
        center_lat=48.135,
        metric_crs=metric_crs,
        random_seed=77,
        start_building_id=10_001
    )

    create_training_projects_csv(
        training_buildings_gdf=training_buildings,
        roads_gdf=roads,
        output_csv="training_projects.csv",
        metric_crs=metric_crs,

        # Если твоя основная модель использует defer, оставь True.
        # Если хочешь строго repair/rebuild/demolish, поставь False.
        include_defer=True
    )

    print("\nDone.")
    print("Generated files:")
    print("- training_projects.csv")
    print("- training_buildings.geojson")
    print("- buildings.geojson")
    print("- roads.geojson")


if __name__ == "__main__":
    main()