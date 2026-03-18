import sys
import os
import tempfile
import time
import uuid
from functools import reduce
from pathlib import Path
from typing import Optional, Sequence, TypedDict

import pandas as pd
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from Featurestore import Entity, Feature, FeatureStoreBase


class OdsToDsConfig(TypedDict):
    col_ls: list[str]
    pred_col_ls: list[str]
    targ_col_ls: list[str]
    hist_win_size: int
    forcast_win_size: int

path = "/home/ma-user/work/feature_store"
if path not in sys.path:
    sys.path.append(path)

DEFAULT_DUCKDB_TEMP_DIR = Path(
    os.environ.get("PIPELINE_DUCKDB_TEMP_DIR", tempfile.gettempdir())
).resolve()
DEFAULT_DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)

FeatureStoreBase.set_config(
    memory_limit="24GB",
    threads=12,
    temp_directory=str(DEFAULT_DUCKDB_TEMP_DIR),
)
FS_CONFIG = FeatureStoreBase()


def get_entity(
    dataframe: pd.DataFrame, station_ls: Optional[Sequence[str]] = None
) -> Entity:
    """获取数据集实体对象。

    通过时间戳窗口和场站标识符定义样本名称，用于后续数据索引。

    Args:
        dataframe: pandas 数据帧
        station_ls: 需要处理的场站名称列表。若为 None，则默认处理目录下所有场站。

    Returns:
        Entity object: 包含唯一标识符 (timestamp_win, station) 的实体集合。
    """
    start_time = time.time()

    table_name = f"df_datatable_{uuid.uuid4().hex[:6]}"
    FS_CONFIG.con.register(table_name, dataframe)

    if station_ls is None:
        sql_source_e = f"""
        SELECT
            timestamp_win,
            station
        FROM {table_name}
        """
    else:
        station_str = "', '".join(station_ls)
        sql_source_e = f"""
        SELECT
            timestamp_win,
            station
        FROM {table_name}
        WHERE station in ('{station_str}')
        """

    entity = Entity(entity_ids=["station", "timestamp_win"], sql_source=sql_source_e)
    print(entity)
    end_time = time.time()
    print(f"Entity 创建时间: {end_time - start_time} s.")

    return entity


def get_hist_features(
    dataframe: pd.DataFrame,
    entity: Entity,
    col_ls: Sequence[str],
    win_size: int = 672,
) -> list[Feature]:
    """提取历史时序特征序列。

    根据指定的实体索引和列名列表，从 Parquet 文件中检索固定窗口长度的历史观测数据。
    该函数是特征工程的核心，确保生成的特征名称与输入 `col_ls` 保持严格对应。

    Args:
        dataframe (pd.DataFrame): ODS层的数据帧
        entity (Entity): 实体索引对象，包含基础标识符（如 timestamp_win, station）。
        col_ls (list[str]): 需要提取的特征列名列表（对应 58 个变量的文件名）。
        win_size (int, optional): 历史回溯窗口步长。默认 672 (若间隔 15min，则代表过去 7 天)。

    Returns:
        list[Feature]: 包含时序特征对象的列表，每个 Feature 对象与 `col_ls` 中的列名一一对应。
    """
    window_hist7d_repr = f"ROWS BETWEEN {win_size - 1} PRECEDING AND CURRENT ROW"
    window_setting = window_hist7d_repr

    table_name = f"df_datatable_{uuid.uuid4().hex[:6]}"
    FS_CONFIG.con.register(table_name, dataframe)

    feature_ls: list[Feature] = []
    for col_name in tqdm(col_ls):
        start_time = time.time()

        sql_source_feat = f"""
        WITH raw_data AS (
            SELECT
                timestamp_win,
                station,
                {col_name}
            FROM {table_name}
        )
        SELECT
            station,
            timestamp_win,
            list({col_name}) OVER(
                PARTITION BY station
                ORDER BY timestamp_win
                {window_setting}
            ) AS {col_name}
        FROM raw_data
        """

        feat_tmp = Feature(
            entity=entity,
            feature_name=col_name,
            sql_source=sql_source_feat,
            description=col_name + " 历史X天特征",
        )

        feat_tmp.describe()
        end_time = time.time()
        print(f"Feature {col_name} spends time = {end_time - start_time} s.")
        feature_ls.append(feat_tmp)

    return feature_ls


def get_pred_features(
    dataframe: pd.DataFrame, entity: Entity, pred_col_ls: Sequence[str]
) -> list[Feature]:
    """获取预报序列特征。

    Args:
        dataframe (pd.DataFrame): ODS层数据帧。
        entity (Entity): 实体索引对象，包含基础标识符（如 timestamp_win, station）。
        pred_col_ls (list[str]): 需要提取的预报特征列名列表。

    Returns:
        list[Feature]: 包含时序特征对象的列表，每个 Feature 对象与 `pred_col_ls` 中的列名一一对应。
    """
    table_name = f"df_datatable_{uuid.uuid4().hex[:6]}"
    FS_CONFIG.con.register(table_name, dataframe)

    feature_ls: list[Feature] = []
    for col_name in tqdm(pred_col_ls):
        start_time = time.time()

        sql_source_feat = f"""
            SELECT
                timestamp_win,
                station,
                {col_name}
            FROM {table_name}
        """

        feat_tmp = Feature(
            entity=entity,
            feature_name=col_name,
            sql_source=sql_source_feat,
            description=col_name + " 预报未来X天特征",
        )

        feat_tmp.describe()
        end_time = time.time()
        print(f"Feature {col_name} spends time = {end_time - start_time} s.")
        feature_ls.append(feat_tmp)

    return feature_ls


def get_target_feature(
    dataframe: pd.DataFrame,
    entity: Entity,
    targ_col_ls: Sequence[str],
    win_size: int = 192,
) -> list[Feature]:
    """获取 target Feature。

    Args:
        dataframe (pd.DataFrame): ODS层数据帧。
        entity (Entity): 实体索引对象，包含基础标识符（如 timestamp_win, station）。
        targ_col_ls (list[str]): 需要提取的 Target 特征列名列表。
        win_size (int): 需要预测的 Target 窗口尺寸。

    Returns:
        list[Feature]: 包含时序特征对象的列表，每个 Feature 的名称是 `targ_col_ls` 中的列名加上 `_future` 后缀。
    """
    window_future2d_repr = f"ROWS BETWEEN 1 FOLLOWING AND {win_size} FOLLOWING"
    window_setting = window_future2d_repr

    table_name = f"df_datatable_{uuid.uuid4().hex[:6]}"
    FS_CONFIG.con.register(table_name, dataframe)

    feature_ls: list[Feature] = []
    for col_name in tqdm(targ_col_ls):
        start_time = time.time()

        sql_source_feat = f"""
        WITH raw_data AS (
            SELECT
                timestamp_win,
                station,
                {col_name}
            FROM {table_name}
        )
        SELECT
            station,
            timestamp_win,
            list({col_name}) OVER(
                PARTITION BY station
                ORDER BY timestamp_win
                {window_setting}
            ) AS {col_name + '_future'}
        FROM raw_data
        """

        feat_tmp = Feature(
            entity=entity,
            feature_name=col_name + "_future",
            sql_source=sql_source_feat,
            description=col_name + "_future" + " 真实未来X天特征",
        )

        feat_tmp.describe()
        end_time = time.time()
        print(f"Feature {col_name + '_future'} spends time = {end_time - start_time} s.")
        feature_ls.append(feat_tmp)

    return feature_ls


def efficient_merge_features(feature_ls: Sequence[Feature]) -> pd.DataFrame:
    """高效将同 Entity 对象的 Feature 合并为 pandas.DataFrame 的 DS 层宽表。"""
    if not feature_ls:
        raise ValueError("feature_ls cannot be empty.")

    entity_cols = feature_ls[0].entity.entity_ids

    df_ls: list[pd.DataFrame] = []
    for feat in tqdm(feature_ls, desc="merging features..."):
        df_tmp = feat.relation.df()
        df_ls.append(df_tmp)

    ds_table = reduce(
        lambda left, right: pd.merge(left, right, on=entity_cols, how="inner"),
        df_ls,
    )

    return ds_table


def convert_ods_to_ds(
    ods_dataframe: pd.DataFrame, plant_ids: Sequence[str], config: OdsToDsConfig
) -> pd.DataFrame:
    """将 ODS 层表转化为 DS 层表。"""
    station_ls = plant_ids

    if not station_ls:
        raise ValueError("plant_ids cannot be empty.")

    ds_table_ls: list[pd.DataFrame] = []
    for station in station_ls:
        entity = get_entity(
            ods_dataframe,
            station_ls=[station],
        )  # 虽然 station_ls 支持列表，但是为了分区保存 DS 层特征，每次依然传入 1 个站点

        col_ls = config["col_ls"]
        hist_features = get_hist_features(
            ods_dataframe,
            entity,
            col_ls,
            win_size=config["hist_win_size"],
        )

        pred_col_ls = config["pred_col_ls"]
        pred_features = get_pred_features(ods_dataframe, entity, pred_col_ls)

        targ_col_ls = config["targ_col_ls"]
        target_features = get_target_feature(
            ods_dataframe,
            entity,
            targ_col_ls,
            win_size=config["forcast_win_size"],
        )  # 南网打榜需要改为 4天*96点=384

        all_features = hist_features + pred_features + target_features
        ds_table_tmp = efficient_merge_features(feature_ls=all_features)

        ds_table_ls.append(ds_table_tmp)

    ds_table = pd.concat(ds_table_ls)

    return ds_table


def load_station_frame(data_dir: str, station_id: str) -> pd.DataFrame:
    """读取单站 parquet 数据并补齐统一字段。"""
    station_path = Path(data_dir) / f"station={station_id}"
    dataframe = pd.read_parquet(station_path)
    dataframe["timestamp_win"] = dataframe["__index_level_0__"]
    dataframe["station"] = station_id
    return dataframe


if __name__ == "__main__":
    total_start_time = time.time()

    data_dir = "/data/pg_data/"
    station_ls: list[str] = []

    if not station_ls:
        raise ValueError("请先在 station_ls 中填入至少一个场站 ID。")

    data_frames = [load_station_frame(data_dir, station_id) for station_id in station_ls]
    df_data = pd.concat(data_frames, ignore_index=True)
    config: OdsToDsConfig = {
        "col_ls": [
            "observe_power",
            "GHI_real",
            "GHI_SOLARGIS",
            "TEMP_SOLARGIS",
            "WS_SOLARGIS",
            "WD_SOLARGIS",
            "ssrd_pos_1",
            "ssrd_pos_2",
            "ssrd_pos_3",
            "ssrd_pos_4",
            "ssrd_pos_5",
            "ssrd_pos_6",
            "ssrd_pos_7",
            "ssrd_pos_8",
            "ssrd_pos_9",
            "t2m_pos_1",
            "t2m_pos_2",
            "t2m_pos_3",
            "t2m_pos_4",
            "t2m_pos_5",
            "t2m_pos_6",
            "t2m_pos_7",
            "t2m_pos_8",
            "t2m_pos_9",
        ],
        "pred_col_ls": [
            "GHI_SOLARGIS_predict",
            "TEMP_SOLARGIS_predict",
            "WD_SOLARGIS_predict",
            "WS_SOLARGIS_predict",
            "ssrd_pos_1_predict",
            "ssrd_pos_2_predict",
            "ssrd_pos_3_predict",
            "ssrd_pos_4_predict",
            "ssrd_pos_5_predict",
            "ssrd_pos_6_predict",
            "ssrd_pos_7_predict",
            "ssrd_pos_8_predict",
            "ssrd_pos_9_predict",
            "t2m_pos_1_predict",
            "t2m_pos_2_predict",
            "t2m_pos_3_predict",
            "t2m_pos_4_predict",
            "t2m_pos_5_predict",
            "t2m_pos_6_predict",
            "t2m_pos_7_predict",
            "t2m_pos_8_predict",
            "t2m_pos_9_predict",
        ],
        "targ_col_ls": ["observe_power"],
        "hist_win_size": 672,
        "forcast_win_size": 384,
    }

    ds_table = convert_ods_to_ds(
        ods_dataframe=df_data,
        plant_ids=station_ls,
        config=config,
    )

    print("done!")
    total_end_time = time.time()
    print(f"总耗时：{total_end_time - total_start_time} s")
