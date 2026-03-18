import pandas as pd 
import numpy as np 

# ==========================================
# 方法 1：每个站最后比例 K 的数据作为测试集
# ==========================================
def split_station_last_k(df, k=0.2):
    # 必须先按站点和时间排序
    df_sorted = df.sort_values(by=['station', 'timestamp_win'])
    
    train_list, test_list = [], []
    for _, group in df_sorted.groupby('station'):
        split_idx = int(len(group) * (1 - k))
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])
        
    return pd.concat(train_list).reset_index(drop=True), pd.concat(test_list).reset_index(drop=True)

# ==========================================
# 方法 2：每个站、每个月最后比例 K 的数据作为测试集
# ==========================================
def split_station_month_last_k(df, k=0.2):
    df_sorted = df.copy().sort_values(by=['station', 'timestamp_win'])
    # 提取年月作为一个新的分组键
    df_sorted['year_month'] = df_sorted['timestamp_win'].dt.to_period('M')
    
    train_list, test_list = [], []
    for _, group in df_sorted.groupby(['station', 'year_month']):
        split_idx = int(len(group) * (1 - k))
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])
        
    # 合并并删除辅助列
    train_df = pd.concat(train_list).drop(columns=['year_month']).reset_index(drop=True)
    test_df = pd.concat(test_list).drop(columns=['year_month']).reset_index(drop=True)
    return train_df, test_df

# ==========================================
# 方法 3：全局按时间排序后，最后比例 K 的数据作为测试集
# ==========================================
def split_global_last_k(df, k=0.2):
    df_sorted = df.sort_values(by='timestamp_win')
    split_idx = int(len(df_sorted) * (1 - k))
    
    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df

# ==========================================
# 方法 4：按固定时间截断 (Fixed Date Cutoff) - 【补充设计】
# ==========================================
def split_by_fixed_date(df, cutoff_date):
    # cutoff_date 可以是字符串，如 '2023-03-01'
    cutoff = pd.to_datetime(cutoff_date)
    
    train_df = df[df['timestamp_win'] < cutoff].reset_index(drop=True)
    test_df = df[df['timestamp_win'] >= cutoff].reset_index(drop=True)
    return train_df, test_df

# ==========================================
# 方法 5：留多站法 (Leave-Multiple-Stations-Out)
# ==========================================
def split_leave_stations_out(df, test_station_ids):
    """
    参数:
    df: 包含 'station' 列的 DataFrame
    test_station_ids: 列表形式的站点ID，例如 ['Station_A', 'Station_B']
    """
    # 增加一个小容错：如果用户不小心传入了单个字符串，自动转成列表
    if isinstance(test_station_ids, str):
        test_station_ids = [test_station_ids]
        
    # 使用 ~ 取反操作符获取不在列表中的站点作为训练集
    train_df = df[~df['station'].isin(test_station_ids)].reset_index(drop=True)
    
    # 获取在列表中的站点作为测试集
    test_df = df[df['station'].isin(test_station_ids)].reset_index(drop=True)
    
    return train_df, test_df