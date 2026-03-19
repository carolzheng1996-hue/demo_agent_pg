import os

import numpy as np
import pandas as pd

PAST_LEN = 24 * 4 * 7
FUTURE_LEN = 24 * 4 * 2
PER_DAY = 24 * 4
STATION_COL = 'station'
TIME_COL = 'timestamp_win'


def fill_nan(arr, points_per_day=PER_DAY):
    """
    填充一维数组中的 NaN 值。
    - 如果是第一天的 NaN，用均值填充
    - 如果是后续天的 NaN，用前一天同一时间点的值填充

    :param arr: 一维 numpy 数组，长度为天数 * points_per_day
    :param points_per_day: 每天的数据点数，默认 PER_DAY
    :return: ndarray
    """
    arr = arr.copy()

    # 如果全是 NaN，直接返回零数组
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)

    n_days = len(arr) // points_per_day
    mean_val = np.nanmean(arr)  # 整体均值

    for day in range(n_days):
        for i in range(points_per_day):
            idx = day * points_per_day + i
            if np.isnan(arr[idx]):
                if day == 0:
                    arr[idx] = mean_val
                else:
                    arr[idx] = arr[(day - 1) * points_per_day + i]
    return arr


def apply_fill_nan(arr):
    arr = np.asarray(arr)
    return fill_nan(arr)


def pad_array_tail(arr, target_len=PAST_LEN):
    """
    检查数组长度，不足则在末尾补0
    :param arr: 数组
    :param target_len: 目标长度
    :return: 数组
    """
    if arr is None:
        return np.zeros(target_len, dtype=float)
    arr = np.asarray(arr)
    if arr.shape[0] < target_len:
        # 在后面补零
        pad_width = target_len - arr.shape[0]
        arr = np.pad(arr, (0, pad_width), mode='constant')
    return arr


def pad_array_head(arr, target_len=PAST_LEN):
    """
    检查数组长度，不足则在头部补0
    :param arr: 数组
    :param target_len: 目标长度
    :return: 数组
    """
    if arr is None:
        return np.zeros(target_len, dtype=float)
    arr = np.asarray(arr)
    if arr.shape[0] < target_len:
        # 在前面补零
        pad_width = target_len - arr.shape[0]
        arr = np.pad(arr, (pad_width, 0), mode='constant')
    return arr


def panding_col_arr(df):
    """
    检查每一列的元素，确保每个元素的数组长度一致
    :param df: dataframe
    :return: None
    """
    for col in df.columns:
        if "_predict" not in col and "_future" not in col and col != TIME_COL:
            print(f"{col} {PAST_LEN}")
            df[col] = df[col].apply(pad_array_head, args=(PAST_LEN,))
        elif "_predict" in col or "_future" in col:
            print(f"{col} {FUTURE_LEN}")
            df[col] = df[col].apply(pad_array_tail, args=(FUTURE_LEN,))


def process_fillna(df):
    """
    检查每一列的元素，确保每个元素的数组内不含nan
    :param df: dataframe
    :return: None
    """
    for col in df.columns:
        if col != TIME_COL:
            df[col] = df[col].apply(apply_fill_nan)


def concont_feature_store(input_path, station_list, process_col, saved_path):
    """
    拼接feature store 数据，形成一个单站点数据
    :param input_path: 输入路径
    :param station_list: 站点列表
    :param process_col: list 需要处理的列名
    :param saved_path: 处理后站点数据保存路径
    :return: None
    """
    file_list = os.listdir(input_path)
    df_result = None
    for station in station_list:
        for file in file_list:
            if not file.endswith('parquet') and file not in process_col:  # filter col
                continue
            df = pd.read_parquet(os.path.join(input_path, file))
            df = df[df[STATION_COL] == station]
            df.drop(columns=STATION_COL, inplace=True)
            if df_result is None:
                df_result = df
            else:
                df_result = pd.merge(df_result, df, on=TIME_COL, how='left')

        panding_col_arr(df_result)  # 长度补齐
        process_fillna(df_result)  # 空值填充

        df_result.to_parquet(os.path.join(saved_path, f"{station}.parquet"))
        df_result = None  # clear


if __name__ == '__main__':
    main_path1 = "/data/pv_dataset/pv_challenge_2026_dataset/feature_store/features_0226"
    station_list1 = [
        # "华能北方润达光伏电站",
        "泗洪光伏电站",
        # "洛浦光伏电站",
        # "白马湖光伏电站",
        # "西北戈壁小壕兔",
        # "国能和煦光储电站"
    ]
    cols = [
        "ssrd_pos_1",
        "ssrd_pos_2",
        # "ssrd_pos_3",
        # "ssrd_pos_4",
        # "ssrd_pos_5",
        # "ssrd_pos_6",
        # "ssrd_pos_7",
        # "ssrd_pos_8",
        # "ssrd_pos_9"
    ]
    concont_feature_store(main_path1,
                          station_list1,
                          cols,
                          "/data/zzm_tmp/YLJ_data/pv_chanllenge_2026/0225/")