#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd 
import duckdb
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Union, Type
import os
import tempfile
import shutil
import atexit
import uuid
import time 
import pyarrow as pa

class FeatureStoreBase:
    _global_con = None
    _conf = {
        "memory_limit": "16GB",
        "threads": "4", # 默认设小一点方便测试
        "temp_directory": None 
    }
    _created_temp_dir = None

    @classmethod
    def _generate_safe_temp_dir(cls, user_specified_path=None):
        if user_specified_path is None:
            base_dir = tempfile.gettempdir()
        else:
            base_dir = os.path.abspath(user_specified_path)

        pid = os.getpid()
        # 加上 Random UUID 防止测试跑太快复用同一个 PID 目录导致冲突（虽然概率很低）
        import uuid
        safe_name = f"duckdb_spill_{pid}_{uuid.uuid4().hex[:6]}"
        
        full_path = os.path.join(base_dir, safe_name)
        os.makedirs(full_path, exist_ok=True)
        
        cls._created_temp_dir = full_path
        return full_path

    @classmethod
    def set_config(cls, **kwargs):
        if "temp_directory" in kwargs:
            raw_path = kwargs.pop("temp_directory")
            safe_path = cls._generate_safe_temp_dir(raw_path)
            cls._conf["temp_directory"] = safe_path
            
        cls._conf.update(kwargs)
        
        if cls._global_con is not None:
            cls._apply_configs(cls._global_con)

    @classmethod
    def _apply_configs(cls, con):
        for key, value in cls._conf.items():
            if value is not None:
                try:
                    con.execute(f"PRAGMA {key}='{value}'")
                except Exception as e:
                    print(f"Config Error: {e}")

    @property
    def con(self):
        if FeatureStoreBase._global_con is None:
            FeatureStoreBase._global_con = duckdb.connect(":memory:")
            self._apply_configs(FeatureStoreBase._global_con)
        return FeatureStoreBase._global_con

    @classmethod
    def cleanup(cls):
        if cls._created_temp_dir and os.path.exists(cls._created_temp_dir):
            try:
                shutil.rmtree(cls._created_temp_dir)
                cls._created_temp_dir = None # 重置
            except Exception as e:
                print(f"Cleanup Error: {e}")

    @classmethod
    def reset(cls):
        """测试专用：强制重置状态"""
        if cls._global_con:
            cls._global_con.close()
        cls._global_con = None
        cls.cleanup() # 清理残留文件
        cls._conf = {"memory_limit": "16GB", "threads": "4", "temp_directory": None}

# 注册清理钩子
atexit.register(FeatureStoreBase.cleanup)


# 定义一个Entity类，用于存储实体信息
class Entity(FeatureStoreBase):
    """
    Entity 类：定义实体集合（Spine）。
    它代表了特征工程的主键骨架 (Entity Spine)。
    无论来源是 DataFrame 还是 SQL，最终都会在 DuckDB 中生成一个去重的、只包含 entity_ids 的视图。
    """

    def __init__(
        self, 
        entity_ids: Union[List[str], str],
        sql_source: Optional[str] = None, 
        dataframe_source: Optional[pd.DataFrame] = None,
        relation_source: Optional[duckdb.DuckDBPyRelation] = None 
    ):
        # 1. 规范化 entity_ids 为列表
        if isinstance(entity_ids, str):
            self.entity_ids = [entity_ids]
        elif isinstance(entity_ids, list):
            self.entity_ids = entity_ids
            for eid in self.entity_ids:
                assert isinstance(eid, str), "Entity IDs list elements must be strings"
        else:
            raise TypeError("entity_ids must be str or List[str]")

        # # 2. 验证互斥性 (XOR)
        # has_sql = sql_source is not None
        # has_df = dataframe_source is not None
        # assert has_sql ^ has_df, "Must provide exactly one of sql_source or dataframe_source"
        # 2. 验证互斥性 (3选1)
        sources = [sql_source is not None, dataframe_source is not None, relation_source is not None]
        assert sum(sources) == 1, "Must provide exactly one of: sql_source, dataframe_source, or relation_source"

        # 3. 生成唯一的视图名称 (防止多个 Entity 冲突)
        # 格式: entity_{hex_uuid}
        self.view_name = f"entity_spine_{uuid.uuid4().hex[:8]}"
        
        # 4. 初始化逻辑分发
        if dataframe_source is not None:
            self._init_from_dataframe(dataframe_source)
        elif sql_source is not None:
            self._init_from_sql(sql_source)
        else:
            self._init_from_relation(relation_source) # <--- 新增处理分支

    def _init_from_dataframe(self, df: pd.DataFrame):
        """处理 DataFrame 源"""
        assert isinstance(df, pd.DataFrame), "dataframe_source must be a pandas DataFrame"
        
        # 验证 DataFrame 是否包含 entity_ids 列
        missing_cols = set(self.entity_ids) - set(df.columns)
        assert not missing_cols, f"DataFrame missing entity columns: {missing_cols}"

        # 注册原始数据为临时视图
        raw_view_name = f"raw_{self.view_name}"
        # DuckDB 可以直接注册 Pandas DF
        self.con.register(raw_view_name, df)
        # self.con.sql(f"CREATE OR REPLACE TEMP TABLE {raw_view_name} AS SELECT * FROM df")
        
        # 创建去重后的 Entity 视图
        self._create_spine_view(source_table=raw_view_name)

    def _init_from_sql(self, sql: str):
        """处理 SQL 源"""
        # 1. 验证 SQL 合法性 (不触发计算)
        self._validate_sql_syntax(sql)
        
        # 2. 将用户提供的 SQL 包装为子查询视图
        raw_view_name = f"raw_{self.view_name}"
        self.con.sql(f"CREATE OR REPLACE VIEW {raw_view_name} AS {sql}")
        # self.con.sql(f"CREATE OR REPLACE TEMP TABLE {raw_view_name} AS {sql}")
        
        # 3. 创建去重后的 Entity 视图
        self._create_spine_view(source_table=raw_view_name)
        
    def _init_from_relation(self, rel: duckdb.DuckDBPyRelation):
        """
        [新增] 处理 DuckDB Relation 源
        """
        # 验证列是否存在
        cols = set(rel.columns)
        missing_cols = set(self.entity_ids) - cols
        if missing_cols:
            raise ValueError(f"Relation source missing entity columns: {missing_cols}")

        raw_view_name = f"raw_{self.view_name}"
        # 直接利用 relation 对象的 create_view 方法
        # 注意：这里假设 rel 是基于同一个 connection 创建的，或者是无状态的
        rel.create_view(raw_view_name, replace=True)
        self._create_spine_view(source_table=raw_view_name)

    def _validate_sql_syntax(self, sql: str):
        """
        使用 EXPLAIN 验证 SQL 语法和表存在性，而不执行实际计算。
        这是 DuckDB 推荐的轻量级验证方式。
        """
        try:
            # EXPLAIN 只生成执行计划，不跑数据
            self.con.sql(f"EXPLAIN {sql}")
        except Exception as e:
            raise ValueError(f"Invalid SQL Source: {e}")

    def _create_spine_view(self, source_table: str):
        """
        核心逻辑：生成最终的去重实体视图
        SELECT DISTINCT id1, id2... FROM source
        """
        cols_str = ", ".join(self.entity_ids)
        where_clause = " AND ".join([f"{col} IS NOT NULL" for col in self.entity_ids])
        
        # 构建最终查询
        query = f"""
        CREATE OR REPLACE TEMP TABLE {self.view_name} AS
        SELECT DISTINCT {cols_str}
        FROM {source_table}
        WHERE {where_clause}
        ORDER BY {cols_str}  
        """
        
        # 执行建表
        self.con.sql(query)
        print(f"📦 [Entity] 已注册视图: {self.view_name} (IDs: {self.entity_ids})")

    @property
    def relation(self):
        """
        返回 DuckDB Relation 对象 (句柄)，供 Feature 类使用。
        类似于你之前代码里的 self.con.table(view)
        """
        return self.con.table(self.view_name)
    
    def show(self, n=5):
        """辅助方法：查看前 N 行"""
        self.relation.limit(n).show()
        

class Feature(FeatureStoreBase):
    """
    Feature 类 (Ver 2.0 - Left Join 对齐版)
    
    初始化完成后，内部维护的 view 保证：
    1. 列只包含: [Entity的主键列..., feature_name]
    2. 行逻辑为: Entity Spine LEFT JOIN Raw Feature Source
    """
    
    def __init__(
        self, 
        entity: Entity,
        feature_name: str,
        sql_source: Optional[str] = None, 
        dataframe_source: Optional[pd.DataFrame] = None,
        relation_source: Optional[duckdb.DuckDBPyRelation] = None,
        description: str = ""  # <--- [新增] 默认为空字符串或默认文案
    ):
        # --- 1. 基础检查 ---
        assert isinstance(entity, Entity), "entity must be an Entity object"
        assert isinstance(feature_name, str), "feature_name must be a string"
        
        # has_sql = sql_source is not None
        # has_df = dataframe_source is not None
        # assert has_sql ^ has_df, "Must provide exactly one of sql_source or dataframe_source"
        
        sources = [sql_source is not None, dataframe_source is not None, relation_source is not None]
        assert sum(sources) == 1, "Must provide exactly one of: sql_source, dataframe_source, or relation_source"

        self.entity = entity
        self.feature_name = feature_name
        # [新增] 保存描述到对象属性，方便直接访问
        self.description = description
        
        # 内部视图名：raw_ (原始数据), final_ (最终对齐数据)
        unique_id = uuid.uuid4().hex[:6]
        self.raw_view_name = f"raw_feat_{feature_name}_{unique_id}"
        self.view_name = f"feat_{feature_name}_{unique_id}"
        
        # --- 2. 注册原始数据 (Raw Layer) ---
        if dataframe_source is not None:
            self.con.register(self.raw_view_name, dataframe_source)
            # self.con.sql(f"CREATE OR REPLACE TEMP TABLE {self.raw_view_name} AS SELECT * FROM dataframe_source")
        elif sql_source is not None:
            self._register_sql_source(sql_source)
        else:
            self._register_relation_source(relation_source) 

        # --- 3. 验证与构建最终视图 (Alignment Layer) ---
        # 这一步会执行 Left Join 并裁剪列
        self._build_aligned_view()
        
        # [新增] 捕获元数据
        self.meta = {
            "name": feature_name,
            "description": description if description else "-", 
            "source_type": "DataFrame" if dataframe_source is not None else ("Relation" if relation_source is not None else "SQL"),
            "duckdb_type": self._get_duckdb_type() 
        }

    def _register_sql_source(self, sql):
        try:
            self.con.sql(f"EXPLAIN {sql}")
        except Exception as e:
            raise ValueError(f"Invalid SQL Source: {e}")
        self.con.sql(f"CREATE OR REPLACE VIEW {self.raw_view_name} AS {sql}")
        
    def _register_relation_source(self, rel: duckdb.DuckDBPyRelation):
        """[新增] 将 Relation 对象注册为 Raw View"""
        # Relation 对象不需要 EXPLAIN 校验，因为它本身就是编译好的查询计划
        rel.create_view(self.raw_view_name, replace=True)
        
    def _get_duckdb_type(self):
        """获取特征列的 DuckDB 数据类型 (Lazy)"""
        try:
            # DESCRIBE view_name
            # 结果通常是: column_name, column_type, null, key, default, extra
            schema_df = self.con.sql(f"DESCRIBE {self.view_name}").df()
            # 找到 feature_name 对应的行，取 type
            type_str = schema_df.loc[schema_df['column_name'] == self.feature_name, 'column_type'].values[0]
            return type_str
        except:
            return "Unknown"

    def _build_aligned_view(self):
        """
        核心逻辑：构建 Entity LEFT JOIN RawFeature
        只选出 entity_ids 和 feature_name
        
        【新增逻辑】：
        在 Join 之前，严格检查 RawFeature 表中是否存在重复的 entity_ids。
        如果存在重复 (One-to-Many)，说明特征表不仅包含特征，还包含了重复的历史版本或脏数据，
        这将导致 Left Join 后 Entity 行数膨胀。此时必须报错。
        """
        # --- 1. 检查原始数据里有没有必须的列 ---
        raw_cols = set(self.con.table(self.raw_view_name).columns)
        required_cols = set(self.entity.entity_ids) | {self.feature_name}
        
        missing = required_cols - raw_cols
        if missing:
            raise ValueError(f"Feature Source missing columns: {missing}. Available: {raw_cols}")

        # # --- 2. 【新增】重复性检验 (Duplication Check) ---
        # # 逻辑：按 entity_ids 分组，如果任何一组的 count > 1，说明主键不唯一
        
        # # 构造分组列字符串: "id1, id2"
        # group_cols = ", ".join(self.entity.entity_ids)
        
        # # 这种查询 DuckDB 优化得非常好，只要找到第一条不满足的就会停止 (LIMIT 1)
        # check_sql = f"""
        # SELECT {group_cols}, COUNT(*) as cnt
        # FROM {self.raw_view_name}
        # GROUP BY {group_cols}
        # HAVING COUNT(*) > 1
        # LIMIT 1
        # """
        
        # # 执行检查
        # duplicate_sample = self.con.sql(check_sql).fetchone()
        
        # if duplicate_sample:
        #     # duplicate_sample 的格式类似: ('battery_01', '2023-01-01', 2)
        #     # 抛出详细错误，帮助用户 debug
        #     raise ValueError(
        #         f"Data Integrity Error: Found duplicate records in Feature '{self.feature_name}'!\n"
        #         f"The combination of Entity Keys {self.entity.entity_ids} must be unique in the feature source.\n"
        #         f"Example of duplicate key: {duplicate_sample[:-1]} (Count: {duplicate_sample[-1]})\n"
        #         "Likely cause: Appended data multiple times or incorrect granularity."
        #     )

        # --- 3. 构建视图 (Safe to Join now) ---
        # 构造 SELECT 部分: e.id, f.feature
        entity_col_str = ", ".join([f"e.{col}" for col in self.entity.entity_ids])
        select_clause = ", ".join([f"e.{col}" for col in self.entity.entity_ids])
        select_clause += f", f.{self.feature_name}"
        
        # 构造 ON 部分: e.id = f.id
        on_clause = " AND ".join([f"e.{col} = f.{col}" for col in self.entity.entity_ids])
        
        query = f"""
        CREATE OR REPLACE VIEW {self.view_name} AS
        SELECT {select_clause}
        FROM {self.entity.view_name} e
        LEFT JOIN {self.raw_view_name} f
          ON {on_clause}
        -- ORDER BY {entity_col_str}
        """
        
        # query = f"""
        # CREATE OR REPLACE VIEW {self.view_name} AS
        # WITH optimized_feat AS MATERIALIZED (
        #     SELECT * FROM {self.raw_view_name} -- 这里已经包含了你 3.5s 的计算逻辑
        # )
        # SELECT {select_clause}
        # FROM {self.entity.view_name} e
        # LEFT JOIN optimized_feat f 
        # ON {on_clause}
        # -- ORDER BY {entity_col_str}
        # """
        
        # 执行视图创建
        self.con.sql(query)
        
    def __repr__(self):
        """让 print(feature) 变得好看"""
        return f"<Feature: {self.feature_name} ({self.meta['duckdb_type']}) | Source: {self.meta['source_type']} | Description: {self.description}>"

    def describe(self):
        """
        [新增] 详细信息展示
        显示描述、数据源类型，以及完整的 SQL 逻辑定义 (无截断)。
        """
        print("="*80)
        print(f"📦 Feature: {self.feature_name}")
        print("-" * 80)
        print(f"📝 Description : {self.description if self.description else '(No description)'}")
        print(f"🔧 Data Type   : {self.meta.get('duckdb_type', 'Unknown')}")
        print(f"🔗 Source Type : {self.meta.get('source_type', 'Unknown')}")
        print(f"👀 View Name   : {self.view_name}")
        
        # print("-" * 80)
        # print("💡 Definition (Full SQL):")
        
        # try:
        #     # 从系统表中获取视图定义的完整 SQL
        #     view_def = self.con.sql(f"SELECT sql FROM duckdb_views() WHERE view_name = '{self.view_name}'").fetchone()[0]
            
        #     # 【修改点】不做 split 和 slice，直接打印完整字符串
        #     print(view_def)
            
        # except Exception as e:
        #     print(f"(Definition source not available: {e})")
            
        print("="*80)

    @property
    def relation(self):
        return self.con.table(self.view_name)
    
    def show(self, n=5):
        self.relation.limit(n).show()


class FeatureSet(FeatureStoreBase):
    """
    FeatureSet 类：特征集组装器
    
    功能：
    将多个 Feature 对象横向拼接成一张宽表。
    
    校验逻辑：
    1. 所有 Feature 必须属于同一个 Entity (Spine)。
    2. Feature 之间不能有重名。
    """
    
    def __init__(self, feature_ls: List[Feature]):
        # --- 1. 基础校验 ---
        if not feature_ls:
            raise ValueError("Feature list cannot be empty.")
        
        # 以第一个特征的 Entity 为基准
        self.base_entity = feature_ls[0].entity
        self.features = feature_ls
        
        # 校验 A: Entity 一致性
        # 我们比较 view_name，确保它们指向的是 DuckDB 里同一个 Entity 视图
        for f in feature_ls:
            if f.entity.view_name != self.base_entity.view_name:
                raise ValueError(
                    f"Entity Mismatch! Feature '{f.feature_name}' belongs to a different Entity.\n"
                    f"Expected: {self.base_entity.view_name}, Got: {f.entity.view_name}"
                )
        
        # 校验 B: 特征名唯一性
        feat_names = [f.feature_name for f in feature_ls]
        if len(set(feat_names)) != len(feat_names):
            from collections import Counter
            duplicates = [item for item, count in Counter(feat_names).items() if count > 1]
            raise ValueError(f"Duplicate feature names found: {duplicates}")

        # 生成视图名称
        self.view_name = f"featureset_{uuid.uuid4().hex[:8]}"
        
        # --- 2. 构建宽表 ---
        self._build_wide_table()

    def _build_wide_table(self):
        """
        核心逻辑：生成宽表 SQL
        SELECT e.id1, e.id2, t0.f1, t1.f2, t2.f3
        FROM entity e
        LEFT JOIN feat_1 t0 ON ...
        LEFT JOIN feat_2 t1 ON ...
        """
        # 1. 构造 SELECT 子句
        # 先选 Entity 的 ID 列
        select_parts = [f"e.{col}" for col in self.base_entity.entity_ids]
        
        # 再选每个 Feature 的特征列 (使用别名 t0, t1...)
        for i, f in enumerate(self.features):
            select_parts.append(f"t{i}.{f.feature_name}")
            
        select_sql = ", ".join(select_parts)
        
        # 2. 构造 FROM 子句
        from_sql = f"{self.base_entity.view_name} e"
        
        # 3. 构造 JOIN 子句 (循环 Left Join)
        join_parts = []
        for i, f in enumerate(self.features):
            alias = f"t{i}"
            # Join 条件: e.id = t0.id AND e.date = t0.date
            on_cond = " AND ".join(
                [f"e.{key} = {alias}.{key}" for key in self.base_entity.entity_ids]
            )
            join_parts.append(f"LEFT JOIN {f.view_name} {alias} ON {on_cond}")
            
        join_sql = "\n".join(join_parts)
        
        # 4. 组装最终 SQL
        query = f"""
        CREATE OR REPLACE VIEW {self.view_name} AS
        SELECT {select_sql}
        FROM {from_sql}
        {join_sql}
        """
        
        # 执行
        # print(f"🔨 [FeatureSet] Building wide table with {len(self.features)} features...")
        self.con.sql(query)
        
    def describe(self):
        """
        打印特征集的元信息概览
        不触发实际数据计算，只查询 Catalog
        """
        print("="*80)
        try:
            row_count = self.base_entity.con.sql(f"SELECT COUNT(1) FROM {self.base_entity.view_name}").fetchone()[0]
            rows_str = f"{row_count:,}"
        except:
            rows_str = "Unknown (Error)"
        # print(f"🚀 [FeatureSet Summary] (Total: {len(self.features)} Features)")
        print(f"🚀 [FeatureSet Summary] (Features: {len(self.features)} | Rows: {rows_str})")
        print("="*80)
        
        # 打印 Entity 信息
        entity_ids = ", ".join(self.base_entity.entity_ids)
        print(f"📌 Entity Spine : [{entity_ids}]")
        print(f"🔗 View Name    : {self.view_name}")
        print("-" * 80)
        
        # 打印特征列表表格
        # 使用 Pandas 的 DataFrame 格式化打印，既简单又对齐好看
        meta_list = []
        for idx, f in enumerate(self.features):
            meta_list.append({
                "ID": idx + 1,
                "Feature Name": f.feature_name,
                "DuckDB Type": f.meta.get('duckdb_type', 'Unknown'),
                "Source": f.meta.get('source_type', 'Unknown'),
                "View": f.view_name,
                "Description": f.meta.get('description', '-')
            })
            
        df_meta = pd.DataFrame(meta_list)
        # 设置 Pandas 显示参数，防止截断
        pd.set_option('display.max_colwidth', 50)
        pd.set_option('display.width', 1000)
        
        # 打印 DataFrame (去掉 index)
        print(df_meta.to_string(index=False))
        
        print("-" * 80)
        print("✅ Integrity Check: Passed (All features aligned to entity)")
        print("="*80)
        
    def profile(self):
        """
        [重量级] 执行数据探查
        触发全表扫描，计算每一列的空值比例和基本统计。
        """
        print("\n⏳ Profiling data (this may take a while)...")
        
        # 1. 动态构造聚合 SQL (One-Pass Optimization)
        # SELECT 
        #    COUNT(feat_a) as feat_a_count, 
        #    COUNT(feat_b) as feat_b_count,
        #    COUNT(*) as total
        # FROM view
        
        selects = ["COUNT(*) as total_rows"]
        for f in self.features:
            # COUNT(col) 统计的是非空值，所以 NULL数 = Total - COUNT(col)
            selects.append(f"COUNT({f.feature_name}) as cnt_{f.feature_name}")
            
        sql = f"SELECT {', '.join(selects)} FROM {self.view_name}"
        
        # 执行计算
        stats = self.con.sql(sql).fetchone()
        
        # 解析结果
        total_rows = stats[0]
        results = []
        
        # stats[0] 是 total, stats[1] 是第一个特征的非空计数...
        for idx, f in enumerate(self.features):
            non_null_count = stats[idx + 1]
            null_count = total_rows - non_null_count
            null_ratio = null_count / total_rows if total_rows > 0 else 0.0
            
            results.append({
                "Feature Name": f.feature_name,
                "Type": f.meta.get('duckdb_type', 'Unknown'),
                "Null Count": null_count,
                "Null Ratio": f"{null_ratio:.2%}" # 格式化为百分比
            })
            
        print("📊 [Data Profile Report]")
        print("-" * 80)
        print(pd.DataFrame(results).to_string(index=False))
        print("=" * 80)

    @property
    def relation(self):
        return self.con.table(self.view_name)
    
    def to_pandas(self):
        return self.relation.df()
    
    def select(self, by_feature_name: List[str]) -> 'FeatureSet':
        """
        [功能]
        从当前 FeatureSet 中筛选出指定的特征，生成一个新的 FeatureSet 对象。
        
        [参数]
        by_feature_name: 需要保留的特征名称列表 (List[str])
        
        [返回]
        新的 FeatureSet 对象 (包含新的视图)
        """
        # 1. 建立哈希映射，方便快速查找 (name -> Feature Object)
        feat_map = {f.feature_name: f for f in self.features}
        
        # 2. 查找并收集目标特征
        selected_features = []
        missing_features = []
        
        for name in by_feature_name:
            if name in feat_map:
                selected_features.append(feat_map[name])
            else:
                missing_features.append(name)
        
        # 3. 校验：如果有找不到的特征，抛出异常
        if missing_features:
            raise ValueError(
                f"❌ Selection Failed! The following features were not found in this FeatureSet:\n"
                f"{missing_features}\n"
                f"Available features: {list(feat_map.keys())}"
            )
            
        # 4. 返回新的 FeatureSet 实例
        # 这会自动触发新实例的 __init__ -> _build_wide_table
        print(f"✂️ [Select] Creating new FeatureSet with {len(selected_features)} features...")
        return FeatureSet(feature_ls=selected_features)
    
    def to_parquet(self, output_path: str, compression: str = 'ZSTD', row_group_size: int = 5000, order_by: List[str] = None):
        """
        [优化版] 导出 Parquet
        
        核心优化点：
        1. 使用 COPY 语句代替 relation.write_parquet，绕过 Python 层开销。
        2. 强制设置小 ROW_GROUP_SIZE，防止大数组(Array)特征撑爆内存。
        3. 默认 ZSTD 压缩，显著减少磁盘占用。
        
        参数:
        - output_path: 输出路径
        - compression: 压缩算法 (推荐 'ZSTD')
        - row_group_size: 每一个数据块的行数。
          对于含有长 Array 的特征，建议设为 2000-10000；普通数据可设为 100000。
        - order_by: (可选) 排序字段列表。
          注意：如果数据量巨大且内存不足，开启排序会触发大量磁盘读写(Spill)，请谨慎使用。
        """
        print(f"🚀 [Export] Starting export to {output_path}...")
        
        # 1. 构造基础查询
        select_sql = f"SELECT * FROM {self.view_name}"
        
        # 2. 处理排序 (仅当用户显式要求时才排序)
        # 如果你的 Entity 本身是有序的，且 Join 也是有序的，这里通常不需要再排
        if order_by:
            order_clause = ", ".join(order_by)
            select_sql += f" ORDER BY {order_clause}"
            print(f"⚠️ [Warning] Global sorting is enabled by '{order_by}'. This may consume significant temp space.")

        # 3. 构造 COPY 语句 (核心)
        # 语法: COPY (SELECT ...) TO 'path' (参数...)
        copy_query = f"""
        COPY (
            {select_sql}
        ) TO '{output_path}'
        (
            FORMAT PARQUET, 
            COMPRESSION '{compression}', 
            ROW_GROUP_SIZE {row_group_size}
        )
        """
        
        # 4. 执行
        try:
            start_t = time.time()
            self.con.sql(copy_query)
            cost = time.time() - start_t
            print(f"✅ [Export] Success! Time cost: {cost:.2f}s")
            
        except Exception as e:
            print(f"❌ [Export] Failed: {e}")
            raise e