import warnings
warnings.filterwarnings("ignore")

import buildings
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib

from shapely import wkt
from shapely.geometry import Point, Polygon, LineString, box
from shapely.affinity import rotate, translate

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ortools.sat.python import cp_model


# 0. Основные настройки модели

ACTIONS = ["repair", "rebuild", "demolish", "defer"]

NUMERIC_FEATURES = [
    "area_m2",
    "perimeter_m",
    "floors",
    "damage_level",
    "access_index",
    "labor_hours",
    "concrete_m3",
    "steel_tons",
    "equipment_hours",
]

CAT_FEATURES = [
    "building_type",
    "zoning_category",
    "action",
]

FEATURES = NUMERIC_FEATURES + CAT_FEATURES

TARGET_COST = "cost"
TARGET_TIME = "duration_days"


# 1. Чтение CSV с geometry в WKT

def read_training_projects_table(csv_path: str) -> pd.DataFrame:
    """
    Читает training_projects.csv в новом формате.

    Этот CSV уже готов для CatBoost и не содержит geometry.
    Он должен содержать одинаковые признаки с прогнозной таблицей,
    а также target-поля:
    - cost
    - duration_days
    """

    df = pd.read_csv(csv_path)

    required = [
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

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"В training_projects.csv отсутствуют поля: {missing}"
        )

    df["area_m2"] = df["area_m2"].astype(float)
    df["perimeter_m"] = df["perimeter_m"].astype(float)
    df["floors"] = df["floors"].fillna(1).astype(int)
    df["damage_level"] = df["damage_level"].astype(float).clip(0, 1)
    df["access_index"] = df["access_index"].astype(float).clip(0, 1)

    df["zoning_category"] = df["zoning_category"].fillna("unknown").astype(str)
    df["building_type"] = df["building_type"].fillna("unknown").astype(str)
    df["action"] = df["action"].fillna("unknown").astype(str)

    df["labor_hours"] = df["labor_hours"].astype(float)
    df["concrete_m3"] = df["concrete_m3"].astype(float)
    df["steel_tons"] = df["steel_tons"].astype(float)
    df["equipment_hours"] = df["equipment_hours"].astype(float)

    df["cost"] = df["cost"].astype(float)
    df["duration_days"] = df["duration_days"].astype(float)

    return df


# 2. Подготовка GIS-признаков для buildings.geojson + roads.geojson

def prepare_gis_features(
    buildings_path: str,
    roads_path: str | None = None,
    metric_crs: str = "EPSG:3857"
) -> gpd.GeoDataFrame:
    """
    Подготовка новых зданий для прогноза.

    Вход:
    - buildings.geojson:
        building_id, geometry, floors, damage_level, building_type, zoning_category
    - roads.geojson:
        road_id, highway, geometry

    Выход:
    - area_m2
    - perimeter_m
    - distance_to_road_m
    - access_index
    """

    buildings = gpd.read_file(buildings_path)

    if buildings.crs is None:
        raise ValueError("У buildings.geojson отсутствует CRS.")

    buildings = buildings.to_crs(metric_crs)

    buildings["geometry"] = buildings.geometry.make_valid()
    buildings["area_m2"] = buildings.geometry.area
    buildings["perimeter_m"] = buildings.geometry.length

    if "building_id" not in buildings.columns:
        buildings["building_id"] = range(1, len(buildings) + 1)

    if "floors" not in buildings.columns:
        buildings["floors"] = 1

    if "damage_level" not in buildings.columns:
        buildings["damage_level"] = 0.0

    if "building_type" not in buildings.columns:
        buildings["building_type"] = "unknown"

    if "zoning_category" not in buildings.columns:
        buildings["zoning_category"] = "unknown"

    buildings["floors"] = buildings["floors"].fillna(1).astype(int)
    buildings["damage_level"] = buildings["damage_level"].astype(float).clip(0, 1)
    buildings["building_type"] = buildings["building_type"].fillna("unknown").astype(str)
    buildings["zoning_category"] = buildings["zoning_category"].fillna("unknown").astype(str)

    if roads_path is not None:
        roads = gpd.read_file(roads_path)

        if roads.crs is None:
            raise ValueError("У roads.geojson отсутствует CRS.")

        roads = roads.to_crs(metric_crs)
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


# 3. Допустимость действий q_ij

def is_action_allowed(row: pd.Series, action: str) -> int:
    """
    Реализация q_ij.

    q_ij = 1, если действие допустимо.
    q_ij = 0, если действие недопустимо.

    Действие defer означает "отложить восстановление".
    Оно всегда допустимо, потому что не требует немедленных работ.
    """

    if action == "defer":
        return 1

    damage = float(row["damage_level"])
    zoning = str(row.get("zoning_category", "unknown"))
    building_type = str(row.get("building_type", "unknown"))

    # Исторические здания нельзя демонтировать
    if zoning == "historical" and action == "demolish":
        return 0

    # Сильно повреждённое здание нельзя просто ремонтировать
    if damage >= 0.80 and action == "repair":
        return 0

    # Почти неповреждённое здание не нужно демонтировать
    if damage <= 0.50 and action == "demolish":
        return 0

    # Культурные объекты лучше не демонтировать
    if building_type == "cultural" and action == "demolish":
        return 0

    return 1


# 4. Создание пар "здание — действие"

def build_building_action_table(buildings: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт строки для всех пар:

    (b_i, a_j)

    То есть для каждого здания создаются варианты:
    - repair
    - rebuild
    - demolish

    Затем добавляется q_ij.
    """

    rows = []

    for _, b in buildings.iterrows():
        for action in ACTIONS:
            row = {
                "building_id": b["building_id"],
                "area_m2": float(b["area_m2"]),
                "perimeter_m": float(b["perimeter_m"]),
                "floors": int(b["floors"]),
                "damage_level": float(b["damage_level"]),
                "access_index": float(b["access_index"]),
                "building_type": str(b["building_type"]),
                "zoning_category": str(b["zoning_category"]),
                "action": action,
            }

            row["q_ij"] = is_action_allowed(pd.Series(row), action)
            rows.append(row)

    return pd.DataFrame(rows)


# 5. Ресурсная модель rho_ijr

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


def add_resource_needs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет rho_ijr:
    - labor_hours
    - concrete_m3
    - steel_tons
    - equipment_hours

    Ресурсы считаются отдельно.
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

        labor_hours.append(
            area * coeff["labor_hours_per_m2"] * factor
        )

        concrete_m3.append(
            area * coeff["concrete_m3_per_m2"] * factor
        )

        steel_tons.append(
            area * coeff["steel_tons_per_m2"] * factor
        )

        equipment_hours.append(
            area * coeff["equipment_hours_per_m2"] * factor
        )

    df["labor_hours"] = np.round(labor_hours, 3)
    df["concrete_m3"] = np.round(concrete_m3, 3)
    df["steel_tons"] = np.round(steel_tons, 3)
    df["equipment_hours"] = np.round(equipment_hours, 3)

    return df


# 6. Если training_projects.csv не имеет action/cost/duration_days
#    создаём синтетическую обучающую выборку

def synthetic_cost_duration(row: pd.Series) -> tuple[float, float]:
    """
    Синтетическая функция для создания cost и duration_days.

    Это не реальная оценка стоимости.
    Используется для прототипа, лабораторной и проверки pipeline.

    action = defer означает, что восстановление откладывается.
    Для defer задаётся малая административная стоимость и минимальное время.
    """

    action = row["action"]
    area = float(row["area_m2"])
    floors = int(row["floors"])
    damage = float(row["damage_level"])
    access = float(row["access_index"])
    zoning = row["zoning_category"]
    btype = row["building_type"]

    if action == "defer":
        access_factor = 1 + 0.2 * (1 - access)
        cost = area * 10 * access_factor
        duration_days = 1.0
        return round(max(cost, 0), 2), round(duration_days, 1)

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
        "unknown": 1.00,
    }.get(zoning, 1.00)

    building_factor = {
        "residential": 1.00,
        "commercial": 1.15,
        "industrial": 1.25,
        "public": 1.20,
        "cultural": 1.40,
        "unknown": 1.00,
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

    duration_days = (
        area
        * base_days_per_m2
        * damage_factor
        * access_factor
        * floors_factor
        * noise_time
    )

    return round(max(cost, 0), 2), round(max(duration_days, 1), 1)


def make_training_dataset(
    base_buildings: pd.DataFrame,
    expand_all_actions: bool = True,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Делает полноценный training_df для CatBoost.

    Если base_buildings уже имеет action, cost, duration_days,
    можно использовать его напрямую.

    Если этих колонок нет, функция создаёт синтетические:
    - action
    - cost
    - duration_days
    """

    np.random.seed(random_seed)

    df = base_buildings.copy()

    has_targets = all(col in df.columns for col in ["action", "cost", "duration_days"])

    if has_targets:
        if "q_ij" not in df.columns:
            df["q_ij"] = df.apply(lambda r: is_action_allowed(r, r["action"]), axis=1)

        df = add_resource_needs(df)
        return df

    # Если action/cost/duration_days отсутствуют:
    # создаём контрфактические варианты для всех действий.
    if expand_all_actions:
        df = build_building_action_table(df)
    else:
        rows = []

        for _, row in df.iterrows():
            d = float(row["damage_level"])

            if d < 0.5:
                action = "repair"
            elif d < 0.85:
                action = "rebuild"
            else:
                action = np.random.choice(["rebuild", "demolish"])

            new_row = row.to_dict()
            new_row["action"] = action
            new_row["q_ij"] = is_action_allowed(pd.Series(new_row), action)
            rows.append(new_row)

        df = pd.DataFrame(rows)

    df = df[df["q_ij"] == 1].copy()
    df = add_resource_needs(df)

    costs = []
    durations = []

    for _, row in df.iterrows():
        cost, duration = synthetic_cost_duration(row)
        costs.append(cost)
        durations.append(duration)

    df["cost"] = costs
    df["duration_days"] = durations

    return df


# 7. Обучение CatBoost

def train_catboost_models(training_df: pd.DataFrame):
    """
    Обучает две модели:

    cost_model:
        признаки -> cost

    time_model:
        признаки -> duration_days
    """

    required = FEATURES + [TARGET_COST, TARGET_TIME]

    missing = [col for col in required if col not in training_df.columns]
    if missing:
        raise ValueError(f"В training_df отсутствуют поля: {missing}")

    df = training_df.copy()

    for col in CAT_FEATURES:
        df[col] = df[col].astype(str).fillna("unknown")

    X = df[FEATURES]
    y_cost = df[TARGET_COST]
    y_time = df[TARGET_TIME]

    X_train, X_test, y_cost_train, y_cost_test, y_time_train, y_time_test = train_test_split(
        X,
        y_cost,
        y_time,
        test_size=0.2,
        random_state=42
    )

    cost_model = CatBoostRegressor(
        iterations=800,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    time_model = CatBoostRegressor(
        iterations=800,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    cost_model.fit(
        X_train,
        y_cost_train,
        cat_features=CAT_FEATURES,
        eval_set=(X_test, y_cost_test),
        use_best_model=True
    )

    time_model.fit(
        X_train,
        y_time_train,
        cat_features=CAT_FEATURES,
        eval_set=(X_test, y_time_test),
        use_best_model=True
    )

    cost_pred = cost_model.predict(X_test)
    time_pred = time_model.predict(X_test)

    metrics = {
        "cost_MAE": mean_absolute_error(y_cost_test, cost_pred),
        "cost_RMSE": mean_squared_error(y_cost_test, cost_pred) ** 0.5,
        "time_MAE": mean_absolute_error(y_time_test, time_pred),
        "time_RMSE": mean_squared_error(y_time_test, time_pred) ** 0.5,
    }

    return cost_model, time_model, metrics


# 8. Прогноз C_hat_ij и T_hat_ij

def predict_cost_time(
    building_action_df: pd.DataFrame,
    cost_model: CatBoostRegressor,
    time_model: CatBoostRegressor,
    defer_cost_per_m2: float = 10.0
) -> pd.DataFrame:
    """
    Добавляет:
    - C_hat_ij
    - T_hat_ij

    Для defer задаётся:
    - малая административная стоимость;
    - нулевое технологическое время;
    - нулевые ресурсы.
    """

    df = building_action_df.copy()

    df = add_resource_needs(df)

    for col in CAT_FEATURES:
        df[col] = df[col].astype(str).fillna("unknown")

    X = df[FEATURES]

    df["C_hat_ij"] = cost_model.predict(X)
    df["T_hat_ij"] = time_model.predict(X)

    df["C_hat_ij"] = df["C_hat_ij"].clip(lower=0)
    df["T_hat_ij"] = df["T_hat_ij"].clip(lower=0)

    defer_mask = df["action"] == "defer"

    if defer_mask.any():
        access_factor = 1 + 0.2 * (1 - df.loc[defer_mask, "access_index"])

        df.loc[defer_mask, "C_hat_ij"] = (
            df.loc[defer_mask, "area_m2"]
            * defer_cost_per_m2
            * access_factor
        ).clip(lower=0)

        df.loc[defer_mask, "T_hat_ij"] = 0.0

        df.loc[defer_mask, "labor_hours"] = 0.0
        df.loc[defer_mask, "concrete_m3"] = 0.0
        df.loc[defer_mask, "steel_tons"] = 0.0
        df.loc[defer_mask, "equipment_hours"] = 0.0

    return df

def calculate_defer_priority(row: pd.Series) -> float:
    """
    Социальный приоритет здания.

    Чем выше значение, тем хуже откладывать восстановление здания.

    Используются:
    - damage_level: степень повреждения;
    - building_type: тип здания;
    - zoning_category: категория зоны.
    """

    damage = float(row.get("damage_level", 0.0))
    building_type = str(row.get("building_type", "unknown"))
    zoning = str(row.get("zoning_category", "unknown"))

    building_type_priority = {
        "public": 1.0,
        "residential": 0.9,
        "cultural": 0.8,
        "commercial": 0.5,
        "industrial": 0.4,
        "unknown": 0.3,
    }.get(building_type, 0.3)

    zoning_priority = {
        "public": 1.0,
        "residential": 0.9,
        "mixed_use": 0.75,
        "historical": 0.8,
        "commercial": 0.5,
        "industrial": 0.4,
        "unknown": 0.3,
    }.get(zoning, 0.3)

    priority_score = (
        0.6 * damage
        + 0.3 * building_type_priority
        + 0.1 * zoning_priority
    )

    return float(np.clip(priority_score, 0.0, 1.0))


def add_social_defer_penalty(
    df: pd.DataFrame,
    base_defer_penalty: float = 100_000
) -> pd.DataFrame:
    """
    Добавляет индивидуальный социальный штраф за defer.

    defer_penalty_i = base_defer_penalty * defer_priority

    Штраф применяется только к строкам, где action == "defer".
    Для repair/rebuild/demolish штраф равен 0.
    """

    df = df.copy()

    df["defer_priority"] = df.apply(calculate_defer_priority, axis=1)

    df["defer_penalty_i"] = 0.0

    defer_mask = df["action"] == "defer"

    df.loc[defer_mask, "defer_penalty_i"] = (
        base_defer_penalty * df.loc[defer_mask, "defer_priority"]
    )

    return df

# 9. Сценарная модель S

def scenario_rule(row: pd.Series) -> str:
    """
    Базовое правило сценария:

    S(b_i) =
    repair, если d_i < 0.5
    rebuild, если 0.5 <= d_i < 0.9
    demolish, если d_i >= 0.9
    """

    d = float(row["damage_level"])

    if d >= 0.90:
        return "demolish"
    if d >= 0.50:
        return "rebuild"
    return "repair"


def choose_safe_scenario_action(group: pd.DataFrame) -> str:
    """
    Выбирает действие сценария, но проверяет q_ij.

    Если базовое правило выбрало запрещённое действие,
    выбирается ближайший допустимый fallback.
    """

    base_row = group.iloc[0]
    preferred_action = scenario_rule(base_row)

    allowed = group[group["q_ij"] == 1].copy()
    allowed_actions = set(allowed["action"].tolist())

    if preferred_action in allowed_actions:
        return preferred_action

    # Fallback-порядок
    fallback_order = ["rebuild", "repair", "demolish", "defer"]

    for action in fallback_order:
        if action in allowed_actions:
            return action

    raise ValueError(
        f"Для здания {base_row['building_id']} нет допустимого действия."
    )


def evaluate_scenario(pred_df: pd.DataFrame) -> dict:
    """
    Оценивает заданный сценарий S.

    Теперь сценарий не падает, если выбранное действие запрещено.
    В таком случае выбирается допустимая альтернатива.
    """

    df = pred_df.copy()

    scenario_actions = []

    for building_id, group in df.groupby("building_id"):
        action = choose_safe_scenario_action(group)

        scenario_actions.append({
            "building_id": building_id,
            "scenario_action": action
        })

    scenario_actions = pd.DataFrame(scenario_actions)

    df = df.merge(
        scenario_actions,
        on="building_id",
        how="left"
    )

    df["x_scenario"] = (df["action"] == df["scenario_action"]).astype(int)

    selected = df[df["x_scenario"] == 1].copy()

    invalid = selected[selected["q_ij"] == 0]

    if len(invalid) > 0:
        raise ValueError(
            "Сценарий всё ещё содержит недопустимые действия. "
            "Проверь q_ij и choose_safe_scenario_action()."
        )

    resources = {
        "labor_hours": selected["labor_hours"].sum(),
        "concrete_m3": selected["concrete_m3"].sum(),
        "steel_tons": selected["steel_tons"].sum(),
        "equipment_hours": selected["equipment_hours"].sum(),
    }

    C_total = selected["C_hat_ij"].sum()

    active_selected = selected[selected["action"] != "defer"].copy()

    if active_selected.empty:
        T_tech = 0.0
    else:
        T_tech = active_selected["T_hat_ij"].max()

    return {
        "selected": selected,
        "C_total": C_total,
        "T_tech": T_tech,
        "resources": resources,
    }


def diagnose_constraints(
    pred_df: pd.DataFrame,
    budget_max: float | None = None,
    resource_limits: dict | None = None
) -> dict:
    """
    Диагностика ограничений перед запуском OR-Tools.

    Показывает:
    - минимальную стоимость, если выбирать самое дешёвое действие;
    - минимальные ресурсы;
    - достаточно ли бюджета и ресурсных лимитов.
    """

    df = pred_df.copy()

    allowed = df[df["q_ij"] == 1].copy()

    if allowed.empty:
        raise ValueError("Нет ни одного допустимого действия q_ij = 1.")

    buildings = sorted(df["building_id"].unique())

    allowed_counts = allowed.groupby("building_id")["action"].count()

    missing_buildings = [
        b for b in buildings
        if b not in allowed_counts.index or allowed_counts.loc[b] == 0
    ]

    if missing_buildings:
        raise ValueError(
            f"Для этих зданий нет допустимых действий: {missing_buildings}"
        )

    cheapest_any = (
        allowed
        .sort_values("C_hat_ij")
        .groupby("building_id")
        .head(1)
        .copy()
    )

    active_allowed = allowed[allowed["action"] != "defer"].copy()

    if not active_allowed.empty:
        cheapest_active = (
            active_allowed
            .sort_values("C_hat_ij")
            .groupby("building_id")
            .head(1)
            .copy()
        )
    else:
        cheapest_active = pd.DataFrame()

    resource_columns = [
        "labor_hours",
        "concrete_m3",
        "steel_tons",
        "equipment_hours",
    ]

    min_cost_any = cheapest_any["C_hat_ij"].sum()

    print("\n=== Диагностика ограничений ===")
    print(f"Количество зданий: {len(buildings)}")
    print(f"Минимальная стоимость с учётом defer: {min_cost_any:,.2f}")

    if budget_max is not None:
        print(f"Заданный бюджет: {budget_max:,.2f}")

        if min_cost_any > budget_max:
            print("Проблема: даже план с defer дороже бюджета.")
        else:
            print("Бюджет допускает хотя бы один план с defer.")

    print("\nМинимальные ресурсы с учётом defer:")

    min_resources_any = {}

    for r in resource_columns:
        value = cheapest_any[r].sum()
        min_resources_any[r] = value
        print(f"{r}: {value:,.3f}")

        if resource_limits is not None and r in resource_limits:
            limit = resource_limits[r]

            if value > limit:
                print(f"  Проблема: лимит {r} слишком мал. Лимит = {limit:,.3f}")
            else:
                print(f"  Лимит {r} допустим.")

    if not cheapest_active.empty:
        min_cost_active = cheapest_active["C_hat_ij"].sum()

        print("\nМинимальная стоимость без defer:")
        print(f"{min_cost_active:,.2f}")

    return {
        "cheapest_any": cheapest_any,
        "cheapest_active": cheapest_active,
        "min_cost_any": min_cost_any,
        "min_resources_any": min_resources_any,
    }

# 10. Оптимизационная модель OR-Tools

def optimize_actions(
    pred_df: pd.DataFrame,
    budget_max: float | None = None,
    resource_limits: dict | None = None,
    resource_capacity_per_day: dict | None = None,
    alpha_cost: float = 1.0,
    beta_time: float = 10.0,
    min_active_buildings: int | None = None,
    max_deferred_buildings: int | None = None,
    max_deferred_ratio: float | None = None,
    defer_penalty: float = 0.0,
    max_time_seconds: int = 60
) -> dict:
    """
    Решает задачу:

    min alpha * C_total + beta * T_total

    Условия:
    - sum_j x_ij = 1
    - x_ij <= q_ij
    - бюджет, если задан
    - ресурсные лимиты, если заданы
    - T_total = max(T_tech, T_resource)
    """

    df = pred_df.copy().reset_index(drop=True)

    # Добавляем индивидуальный социальный штраф за defer.
    # defer_penalty теперь трактуется как базовый штраф,
    # который умножается на социальный приоритет здания.
    if "defer_priority" not in df.columns or "defer_penalty_i" not in df.columns:
        df = add_social_defer_penalty(
            df,
            base_defer_penalty=defer_penalty
        )

    buildings = sorted(df["building_id"].unique())
    actions = sorted(df["action"].unique())

    buildings = sorted(df["building_id"].unique())
    actions = sorted(df["action"].unique())

    model = cp_model.CpModel()

    COST_SCALE = 100
    TIME_SCALE = 10
    RESOURCE_SCALE = 100

    x = {}

    # Переменные x_ij

    for _, row in df.iterrows():
        i = row["building_id"]
        j = row["action"]

        var = model.NewBoolVar(f"x_{i}_{j}")
        x[(i, j)] = var

        if int(row["q_ij"]) == 0:
            model.Add(var == 0)

    # Каждому зданию ровно одно действие

    for i in buildings:
        vars_for_i = [x[(i, j)] for j in actions if (i, j) in x]
        model.Add(sum(vars_for_i) == 1)

    # Ограничения на количество отложенных зданий defer

    defer_penalty_terms = []
    defer_vars = []

    if "defer" in actions:
        defer_vars = [
            x[(i, "defer")]
            for i in buildings
            if (i, "defer") in x
        ]

        for _, row in df[df["action"] == "defer"].iterrows():
            i = row["building_id"]
            j = row["action"]

            penalty_int = int(round(float(row["defer_penalty_i"]) * COST_SCALE))
            defer_penalty_terms.append(penalty_int * x[(i, j)])

    deferred_count = sum(defer_vars)
    active_count = len(buildings) - deferred_count

    if min_active_buildings is not None:
        model.Add(active_count >= int(min_active_buildings))

    if max_deferred_buildings is not None:
        model.Add(deferred_count <= int(max_deferred_buildings))

    if max_deferred_ratio is not None:
        max_deferred_from_ratio = int(np.floor(len(buildings) * max_deferred_ratio))
        model.Add(deferred_count <= max_deferred_from_ratio)

    defer_penalty_term = sum(defer_penalty_terms)
    
    
    # Стоимость

    total_cost_terms = []

    for _, row in df.iterrows():
        i = row["building_id"]
        j = row["action"]

        cost_int = int(round(float(row["C_hat_ij"]) * COST_SCALE))
        total_cost_terms.append(cost_int * x[(i, j)])

    total_cost = sum(total_cost_terms)

    if budget_max is not None:
        model.Add(total_cost <= int(round(budget_max * COST_SCALE)))

    # Ресурсные ограничения

    resource_columns = [
        "labor_hours",
        "concrete_m3",
        "steel_tons",
        "equipment_hours",
    ]

    if resource_limits is not None:
        for r, limit in resource_limits.items():
            if r not in resource_columns:
                raise ValueError(f"Неизвестный ресурс: {r}")

            terms = []

            for _, row in df.iterrows():
                i = row["building_id"]
                j = row["action"]

                value_int = int(round(float(row[r]) * RESOURCE_SCALE))
                terms.append(value_int * x[(i, j)])

            model.Add(sum(terms) <= int(round(limit * RESOURCE_SCALE)))

    # T_tech = max selected T_hat_ij только для активных действий
    # defer не должен увеличивать технологическое время проекта.

    active_df = df[df["action"] != "defer"].copy()

    if active_df.empty:
        max_duration_int = 0
    else:
        max_duration_int = int(round(float(active_df["T_hat_ij"].max()) * TIME_SCALE)) + 1

    T_tech = model.NewIntVar(0, max_duration_int, "T_tech")

    for _, row in df.iterrows():
        i = row["building_id"]
        j = row["action"]

        if j == "defer":
            continue

        duration_int = int(round(float(row["T_hat_ij"]) * TIME_SCALE))

        model.Add(T_tech >= duration_int).OnlyEnforceIf(x[(i, j)])

    # T_resource = max_r U_r / kappa_r

    T_resource = model.NewIntVar(0, 10**9, "T_resource")

    if resource_capacity_per_day is not None:
        for r, capacity in resource_capacity_per_day.items():
            if r not in resource_columns:
                raise ValueError(f"Неизвестный ресурс: {r}")

            if capacity <= 0:
                raise ValueError(f"capacity для ресурса {r} должен быть > 0")

            resource_terms = []

            for _, row in df.iterrows():
                i = row["building_id"]
                j = row["action"]

                value_int = int(round(float(row[r]) * RESOURCE_SCALE))
                resource_terms.append(value_int * x[(i, j)])

            U_r = sum(resource_terms)

            capacity_int = int(round(capacity * RESOURCE_SCALE))

            # T_resource / TIME_SCALE >= U_r / capacity
            # T_resource * capacity_int >= U_r * TIME_SCALE
            model.Add(T_resource * capacity_int >= U_r * TIME_SCALE)
    else:
        model.Add(T_resource == 0)

    # T_total = max(T_tech, T_resource)

    T_total = model.NewIntVar(0, 10**9, "T_total")

    model.Add(T_total >= T_tech)
    model.Add(T_total >= T_resource)

    # Целевая функция

    alpha_int = int(round(alpha_cost * 1000))
    beta_int = int(round(beta_time * 1000))

    model.Minimize(
        alpha_int * total_cost
        + beta_int * T_total
        + alpha_int * defer_penalty_term
    )

    # Решение

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        status_name = solver.StatusName(status)
        raise RuntimeError(
            f"Допустимое решение не найдено. "
            f"Статус OR-Tools: {status_name}. "
            f"Проверь budget_max, resource_limits, q_ij, "
            f"min_active_buildings и max_deferred_ratio."
        )

    selected_rows = []

    for _, row in df.iterrows():
        i = row["building_id"]
        j = row["action"]

        if solver.Value(x[(i, j)]) == 1:
            selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)

    deferred_count_value = int((selected["action"] == "defer").sum())
    active_count_value = int((selected["action"] != "defer").sum())

    resources_real = {
        r: float(selected[r].sum())
        for r in resource_columns
    }

    # --------------------------------------------------------
    # Реальный пересчёт T_resource после решения
    # --------------------------------------------------------
    # Внутренние переменные OR-Tools нужны для оптимизации.
    # Для отчёта лучше пересчитывать время вручную по выбранным действиям.
    # --------------------------------------------------------

    if resource_capacity_per_day is not None:
        resource_times = []

        for r, capacity in resource_capacity_per_day.items():
            if r in resources_real and capacity > 0:
                resource_times.append(resources_real[r] / capacity)

        T_resource_real = max(resource_times) if resource_times else 0.0
    else:
        T_resource_real = 0.0

    # --------------------------------------------------------
    # Реальный T_tech: только активные действия
    # --------------------------------------------------------

    active_selected = selected[selected["action"] != "defer"].copy()

    if active_selected.empty:
        T_tech_real = 0.0
    else:
        T_tech_real = float(active_selected["T_hat_ij"].max())

    T_total_real = max(T_tech_real, T_resource_real)
    
    defer_penalty_total_value = float(
        selected.loc[selected["action"] == "defer", "defer_penalty_i"].sum()
    )
    
    result = {
        "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        "selected": selected,
        "C_total": solver.Value(total_cost) / COST_SCALE,
        "defer_penalty_total": defer_penalty_total_value,
        "objective_value": solver.ObjectiveValue(),

        # Значения из OR-Tools
        "T_tech_solver": solver.Value(T_tech) / TIME_SCALE,
        "T_resource_solver": solver.Value(T_resource) / TIME_SCALE,
        "T_total_solver": solver.Value(T_total) / TIME_SCALE,

        # Реальные отчётные значения
        "T_tech": T_tech_real,
        "T_resource": T_resource_real,
        "T_total": T_total_real,

        "deferred_count": deferred_count_value,
        "active_count": active_count_value,
        "avg_defer_priority": float(
            selected.loc[selected["action"] == "defer", "defer_priority"].mean()
        ) if deferred_count_value > 0 else 0.0,
        
        "resources": resources_real,
    }

    return result


# 11. Сохранение и загрузка моделей

def save_models(cost_model, time_model):
    joblib.dump(cost_model, "cost_model_catboost.pkl")
    joblib.dump(time_model, "time_model_catboost.pkl")


def load_models():
    cost_model = joblib.load("cost_model_catboost.pkl")
    time_model = joblib.load("time_model_catboost.pkl")

    return cost_model, time_model


def save_dataframe_csv(
    df: pd.DataFrame,
    output_path: str,
    drop_geometry: bool = True
) -> None:
    """
    Безопасно сохраняет DataFrame/GeoDataFrame в CSV.

    Если есть geometry, она либо удаляется, либо переводится в WKT.
    """

    out = df.copy()

    if "geometry" in out.columns:
        if drop_geometry:
            out = out.drop(columns=["geometry"])
        else:
            out["geometry"] = out["geometry"].apply(
                lambda geom: geom.wkt if geom is not None else None
            )

    out.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(out)}")


def save_pipeline_outputs(
    intermediate_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    opt_result: dict,
    output_dir: str = "."
) -> None:
    """
    Сохраняет промежуточные и итоговые таблицы pipeline.

    Создаёт:
    - intermediate_prediction_table.csv
    - predictions_full_table.csv
    - selected_budget_plan.csv
    - selected_active_budget_plan.csv
    - selected_deferred_plan.csv
    - optimization_summary.csv
    """

    # --------------------------------------------------------
    # 1. Промежуточная таблица до CatBoost
    # --------------------------------------------------------

    intermediate_columns = [
        "building_id",
        "area_m2",
        "perimeter_m",
        "floors",
        "damage_level",
        "access_index",
        "zoning_category",
        "building_type",
        "action",
        "q_ij",
        "labor_hours",
        "concrete_m3",
        "steel_tons",
        "equipment_hours",
    ]

    existing_intermediate_columns = [
        col for col in intermediate_columns
        if col in intermediate_df.columns
    ]

    save_dataframe_csv(
        intermediate_df[existing_intermediate_columns],
        f"{output_dir}/intermediate_prediction_table.csv"
    )

    # --------------------------------------------------------
    # 2. Полная таблица прогнозов после CatBoost
    # --------------------------------------------------------

    prediction_columns = [
        "building_id",
        "area_m2",
        "perimeter_m",
        "floors",
        "damage_level",
        "access_index",
        "zoning_category",
        "building_type",
        "action",
        "q_ij",
        "defer_priority",
        "defer_penalty_i",
        "labor_hours",
        "concrete_m3",
        "steel_tons",
        "equipment_hours",
        "C_hat_ij",
        "T_hat_ij",
    ]

    existing_prediction_columns = [
        col for col in prediction_columns
        if col in pred_df.columns
    ]

    save_dataframe_csv(
        pred_df[existing_prediction_columns],
        f"{output_dir}/predictions_full_table.csv"
    )

    # --------------------------------------------------------
    # 3. Итоговый выбранный план OR-Tools
    # --------------------------------------------------------

    selected = opt_result["selected"].copy()

    selected_columns = [
        "building_id",
        "action",
        "q_ij",
        "area_m2",
        "perimeter_m",
        "floors",
        "damage_level",
        "access_index",
        "zoning_category",
        "building_type",
        "defer_priority",
        "defer_penalty_i",
        "C_hat_ij",
        "T_hat_ij",
        "labor_hours",
        "concrete_m3",
        "steel_tons",
        "equipment_hours",
    ]

    existing_selected_columns = [
        col for col in selected_columns
        if col in selected.columns
    ]

    selected = selected[existing_selected_columns].copy()

    selected["is_deferred"] = selected["action"].eq("defer").astype(int)
    selected["is_active"] = selected["action"].ne("defer").astype(int)

    save_dataframe_csv(
        selected,
        f"{output_dir}/selected_budget_plan.csv"
    )

    # --------------------------------------------------------
    # 4. Только активные здания, которые реально входят в бюджет работ
    # --------------------------------------------------------

    selected_active = selected[selected["action"] != "defer"].copy()

    save_dataframe_csv(
        selected_active,
        f"{output_dir}/selected_active_budget_plan.csv"
    )

    # --------------------------------------------------------
    # 5. Только отложенные здания
    # --------------------------------------------------------

    selected_deferred = selected[selected["action"] == "defer"].copy()

    save_dataframe_csv(
        selected_deferred,
        f"{output_dir}/selected_deferred_plan.csv"
    )

    # --------------------------------------------------------
    # 6. Краткая сводка оптимизации
    # --------------------------------------------------------

    summary = pd.DataFrame([{
        "status": opt_result["status"],
        "C_total": opt_result["C_total"],
        "defer_penalty_total": opt_result["defer_penalty_total"],
        "objective_value": opt_result["objective_value"],
        "T_tech": opt_result["T_tech"],
        "T_resource": opt_result["T_resource"],
        "T_total": opt_result["T_total"],
        "deferred_count": opt_result["deferred_count"],
        "active_count": opt_result["active_count"],
        "avg_defer_priority": opt_result["avg_defer_priority"],
        "labor_hours": opt_result["resources"].get("labor_hours", 0),
        "concrete_m3": opt_result["resources"].get("concrete_m3", 0),
        "steel_tons": opt_result["resources"].get("steel_tons", 0),
        "equipment_hours": opt_result["resources"].get("equipment_hours", 0),
    }])

    save_dataframe_csv(
        summary,
        f"{output_dir}/optimization_summary.csv"
    )


# 12. Полный пример запуска

def run_full_pipeline():
    """
    Полный pipeline:

    1. Читает training_projects.csv
    2. Обучает CatBoost
    3. Читает buildings.geojson + roads.geojson
    4. Создаёт пары building-action
    5. Прогнозирует C_hat_ij и T_hat_ij
    6. Оценивает сценарий
    7. Оптимизирует план через OR-Tools
    """

    # 1. Training data

    training_df = read_training_projects_table(
        csv_path="training_projects.csv"
    )

    print("Training dataframe:")
    print(training_df.head())

    # 2. Train CatBoost

    cost_model, time_model, metrics = train_catboost_models(training_df)

    print("\nCatBoost metrics:")
    print(metrics)

    save_models(cost_model, time_model)

    # 3. Prepare GIS data for prediction

    buildings = prepare_gis_features(
        buildings_path="buildings.geojson",
        roads_path="roads.geojson",
        metric_crs="EPSG:3857"
    )

    building_action_df = build_building_action_table(buildings)

    # --------------------------------------------------------
    # Промежуточная таблица:
    # GeoPandas + action + resources
    # но ещё без C_hat_ij и T_hat_ij
    # --------------------------------------------------------

    intermediate_df = add_resource_needs(building_action_df)

    save_dataframe_csv(
        intermediate_df[[
            "building_id",
            "area_m2",
            "perimeter_m",
            "floors",
            "damage_level",
            "access_index",
            "zoning_category",
            "building_type",
            "action",
            "q_ij",
            "labor_hours",
            "concrete_m3",
            "steel_tons",
            "equipment_hours",
        ]],
        "intermediate_prediction_table.csv"
    )

    # 4. Predict cost and time

    pred_df = predict_cost_time(
        building_action_df=building_action_df,
        cost_model=cost_model,
        time_model=time_model
    )

    pred_df = add_social_defer_penalty(
        pred_df,
        base_defer_penalty=200_000
    )

    # Сохраняем полную таблицу прогнозов
    save_dataframe_csv(
        pred_df[[
            "building_id",
            "area_m2",
            "perimeter_m",
            "floors",
            "damage_level",
            "access_index",
            "zoning_category",
            "building_type",
            "action",
            "q_ij",
            "defer_priority",
            "defer_penalty_i",
            "labor_hours",
            "concrete_m3",
            "steel_tons",
            "equipment_hours",
            "C_hat_ij",
            "T_hat_ij",
        ]],
        "predictions_full_table.csv"
    )

    print("\nPredictions:")
    print(pred_df[[
        "building_id",
        "action",
        "q_ij",
        "damage_level",
        "building_type",
        "zoning_category",
        "defer_priority",
        "defer_penalty_i",
        "C_hat_ij",
        "T_hat_ij",
        "labor_hours",
        "concrete_m3",
        "steel_tons",
        "equipment_hours"
    ]].head(10))

    # 5. Scenario evaluation

    scenario_result = evaluate_scenario(pred_df)

    print("\nScenario result:")
    print("C_total:", scenario_result["C_total"])
    print("T_tech:", scenario_result["T_tech"])
    print("Resources:", scenario_result["resources"])

    # 6. Optimization

    resource_limits = {
        "labor_hours": 120_000,
        "concrete_m3": 15_000,
        "steel_tons": 1_000,
        "equipment_hours": 10_000,
    }

    diagnose_constraints(
        pred_df=pred_df,
        budget_max=5_000_000,
        resource_limits=resource_limits
    )

    opt_result = optimize_actions(
        pred_df=pred_df,
        budget_max=5_000_000,
        resource_limits=resource_limits,
        resource_capacity_per_day={
            "labor_hours": 800,
            "concrete_m3": 120,
            "steel_tons": 20,
            "equipment_hours": 80,
        },
        alpha_cost=1.0,
        beta_time=1.0,
        min_active_buildings=1,
        max_deferred_ratio=0.90,
        defer_penalty=200_000,
        max_time_seconds=60
    )

    # --------------------------------------------------------
    # Сохраняем итоговые таблицы оптимизации
    # --------------------------------------------------------

    save_pipeline_outputs(
        intermediate_df=intermediate_df,
        pred_df=pred_df,
        opt_result=opt_result,
        output_dir="."
    )

    print("\nOptimization result:")
    print("Status:", opt_result["status"])
    print("C_total:", opt_result["C_total"])
    print("Defer penalty total:", opt_result["defer_penalty_total"])
    print("Average defer priority:", opt_result["avg_defer_priority"])
    print("Objective value:", opt_result["objective_value"])

    print("\nSolver time values:")
    print("T_tech_solver:", opt_result["T_tech_solver"])
    print("T_resource_solver:", opt_result["T_resource_solver"])
    print("T_total_solver:", opt_result["T_total_solver"])

    print("\nReal recalculated time values:")
    print("T_tech:", opt_result["T_tech"])
    print("T_resource:", opt_result["T_resource"])
    print("T_total:", opt_result["T_total"])

    print("\nAction counts:")
    print("Deferred buildings:", opt_result["deferred_count"])
    print("Active buildings:", opt_result["active_count"])

    print("\nResources:")
    print(opt_result["resources"])

    print("\nSelected actions:")
    print(opt_result["selected"][[
        "building_id",
        "action",
        "damage_level",
        "building_type",
        "zoning_category",
        "defer_priority",
        "defer_penalty_i",
        "C_hat_ij",
        "T_hat_ij",
        "labor_hours",
        "concrete_m3",
        "steel_tons",
        "equipment_hours"
    ]])

    return {
        "training_df": training_df,
        "pred_df": pred_df,
        "scenario_result": scenario_result,
        "opt_result": opt_result,
    }


if __name__ == "__main__":
    run_full_pipeline()