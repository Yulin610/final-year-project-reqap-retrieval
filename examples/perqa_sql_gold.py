"""
与 ReQAP CE 一致的 SQL→gold_obs_event_ids 派生（MiniQE + DuckDB + merge）。

供 verify_perqa_gold_obs_selfcheck.py、export_queries_jsonl.py 复用。
依赖: pip install pandas duckdb
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb
import pandas as pd

_EX = Path(__file__).resolve().parent
if str(_EX) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_EX))

# --- copied: reqap/library/library.py flatten_nested_df ---
def flatten_nested_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    try:
        type_map = df.map(type)
    except AttributeError:
        type_map = df.applymap(type)
    s = type_map.eq(list).all()
    list_columns = s[s].index.tolist()
    s2 = type_map.eq(dict).all()
    dict_columns = s2[s2].index.tolist()
    for col in dict_columns:
        horiz_exploded = pd.json_normalize(df[col]).add_prefix(f"{col}.")
        horiz_exploded.index = df.index
        df = pd.concat([df, horiz_exploded], axis=1).drop(columns=[col])
    for col in list_columns:
        df = df.drop(columns=[col]).join(df[col].explode().to_frame())
        df = df.reset_index(drop=True)
    return df


# --- copied: reqap/retrieval/query_execution.py constants + MiniQE ---
TABLE_RENAMING = {
    "annual_doctor_appointment": "doctor_appointment",
    "oneoff_event": "personal_milestone",
}

OBS_SCHEMAS = {
    "calendar": """CREATE TABLE IF NOT EXISTS calendar (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, summary VARCHAR, location VARCHAR, description VARCHAR);""",
    "mail": """CREATE TABLE IF NOT EXISTS mail (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR NULL, end_time VARCHAR NULL, subject VARCHAR, timestamp VARCHAR, sender VARCHAR, recipient VARCHAR, text VARCHAR);""",
    "movie_stream": """CREATE TABLE IF NOT EXISTS movie_stream (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, duration DOUBLE, duration_unit VARCHAR, country VARCHAR, stream_end_time VARCHAR, movie_title VARCHAR, stream_full_title VARCHAR, watching_continuation VARCHAR);""",
    "music_stream": """CREATE TABLE IF NOT EXISTS music_stream (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, duration DOUBLE, duration_unit VARCHAR, song_name VARCHAR, song_artist VARCHAR[]);""",
    "online_purchase": """CREATE TABLE IF NOT EXISTS online_purchase (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, "order" VARCHAR, product_quantity BIGINT, product VARCHAR, price VARCHAR);""",
    "social_media": """CREATE TABLE IF NOT EXISTS social_media (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR NULL, end_time VARCHAR NULL, text VARCHAR);""",
    "tvseries_stream": """CREATE TABLE IF NOT EXISTS tvseries_stream (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, duration DOUBLE, country VARCHAR, stream_end_time VARCHAR, tvseries_title VARCHAR, season_name VARCHAR, episode_name VARCHAR, episode_number BIGINT, tvseries_season BIGINT, watching_continuation VARCHAR);""",
    "workout": """CREATE TABLE IF NOT EXISTS workout (id BIGINT, start_date VARCHAR, start_time VARCHAR, end_date VARCHAR, end_time VARCHAR, event VARCHAR, workout_type VARCHAR, duration BIGINT, duration_unit VARCHAR, minimum_heart_rate BIGINT, maximum_heart_rate BIGINT, average_heart_rate DOUBLE, minimum_speed DOUBLE, maximum_speed DOUBLE, average_speed DOUBLE, speed_unit VARCHAR, distance DOUBLE, distance_unit VARCHAR);""",
}


class MiniQE:
    class SQLError(Exception):
        pass

    def __init__(self, obs_csv: str, str_csv: str) -> None:
        obs_df = pd.read_csv(obs_csv, converters={"event_data": json.loads})
        str_df = pd.read_csv(str_csv, converters={"event_data": json.loads})
        self.db = duckdb.connect(database=":memory:", read_only=False)
        self._init_db(str_df, obs_df)

    def _init_db(self, str_df: Optional[pd.DataFrame], obs_df: Optional[pd.DataFrame]) -> None:
        tables = self.db.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'BASE TABLE';"
        ).fetchall()
        for (table_name,) in tables:
            self.db.execute(f"DROP TABLE IF EXISTS {table_name};")
        if str_df is not None:
            self.db.register("all_events", str_df)
            df = str_df.groupby(["event_type"])
            for event_type, df_group in df:
                df_group = flatten_nested_df(df_group)
                df_group.columns = df_group.columns.str.replace("event_data.", "")
                df_group.columns = df_group.columns.str.replace(".", "_")
                et = event_type[0]
                if et in TABLE_RENAMING:
                    et = TABLE_RENAMING[et]
                self.db.register(et, df_group)
        if obs_df is not None:
            df = obs_df.groupby(["event_type"])
            for event_type, df_group in df:
                et = event_type[0]
                if str_df is not None and et not in ("calendar", "mail", "social_media"):
                    continue
                df_group = flatten_nested_df(df_group)
                df_group.columns = df_group.columns.str.replace("event_data.", "")
                df_group.columns = df_group.columns.str.replace(".", "_")
                self.db.register(et, df_group)
            for _, sql in OBS_SCHEMAS.items():
                self.db.execute(sql)

    @staticmethod
    def adjust_sql(sql_query: str, reference_date: Optional[str]) -> str:
        if reference_date is None:
            return sql_query
        datetime.strptime(reference_date, "%Y-%m-%d")
        return sql_query.replace("CURRENT_DATE", f"'{reference_date}'::DATE")

    @staticmethod
    def derive_retrieval_query(sql_query: str, retain_where_clause: bool, drop_dates_from_where_clause: bool = True) -> str:
        EXPRESSIONS_BLACKLIST = ["date", "time", "<", ">"]
        FROM_PATTERN = r"FROM\s+(\w+)(?:\s|$)"
        WHERE_PATTERN = r"WHERE\s+(.*?)(?:\s+GROUP BY|\s+ORDER BY|$)"
        sql_query = sql_query.replace(";", "")
        from_clause_match = re.search(FROM_PATTERN, sql_query, re.S)
        from_clause = from_clause_match.group(1).strip() if from_clause_match else "1"
        where_clause_match = re.search(WHERE_PATTERN, sql_query, re.S)
        where_clause = where_clause_match.group(1).strip() if where_clause_match else "1"
        query = re.sub(r"SELECT\s+.*?\s+FROM", "SELECT id FROM", sql_query)
        query = re.sub(r"ORDER BY.*", ";", query)
        if drop_dates_from_where_clause:
            where_clauses = re.split(r"\s+(AND|OR)\s+", where_clause)
            where_clauses = [
                clause.strip() if not any(exp in clause.lower() for exp in EXPRESSIONS_BLACKLIST) else "1"
                for clause in where_clauses
            ]
            where_clause = " ".join(where_clauses).strip()
        if not retain_where_clause:
            where_clause = "1"
        return f"SELECT id FROM {from_clause.strip()} WHERE {where_clause.strip()};"

    @staticmethod
    def is_simple_query(query: str) -> bool:
        BLACKLIST = ["WITH", "JOIN"]
        BLACKLIST_BEFORE_WHERE = ["<", ">"]
        for w in BLACKLIST:
            if w in query:
                return False
        if "WHERE" in query:
            before_where, _after = query.rsplit("WHERE", 1)
            for w in BLACKLIST_BEFORE_WHERE:
                if w in before_where:
                    return False
        return True

    @staticmethod
    def _is_none_answer(df: pd.DataFrame) -> bool:
        return df is None or df.empty

    def run_sql(self, sql_query: str, reference_date: Optional[str]) -> pd.DataFrame:
        sql_query = self.adjust_sql(sql_query, reference_date)
        sql_query = sql_query.strip()
        if sql_query.endswith("LIMIT 1;") or sql_query.endswith("LIMIT 1"):
            new_sql = sql_query.replace("LIMIT 1;", ";")
            if new_sql.endswith("LIMIT 1"):
                new_sql = new_sql[:-7].strip()
            df = self.run_sql(new_sql, reference_date)
            if self._is_none_answer(df):
                return df
            reference_value = df.iloc[:, -1].iloc[0]
            return df[df.iloc[:, -1] == reference_value]
        try:
            return self.db.query(sql_query).df()
        except Exception as e:
            raise self.SQLError(f"{sql_query}: {e}") from e


# --- copied: crossencoder retrieve_gold_events / derive_gold ---
def _is_obs_event_sql_query(query: str) -> bool:
    OBS_EVENTS_TABLES = ["social_media", "mail", "calendar"]
    return any(t_name in query for t_name in OBS_EVENTS_TABLES)


def retrieve_gold_events(
    retrieval_sql_query: str, observable_event_data: pd.DataFrame, qe: MiniQE, reference_date: Optional[str]
) -> Tuple[List[int], List[int]]:
    sql_query_res = qe.run_sql(retrieval_sql_query, reference_date)
    str_event_ids = sql_query_res.values.tolist()
    str_event_ids = [id_ for id_list in str_event_ids for id_ in id_list]
    if _is_obs_event_sql_query(retrieval_sql_query):
        return [], str_event_ids  # ids already from obs-level SQL
    observable_event_data = observable_event_data.copy()
    observable_event_data["structured_event_id"] = pd.to_numeric(observable_event_data["structured_event_id"])
    obs_events = pd.merge(sql_query_res, observable_event_data, left_on="id", right_on="structured_event_id")
    obs_event_ids = pd.to_numeric(obs_events["id_y"]).tolist()
    return str_event_ids, obs_event_ids


def derive_gold_retrieval_data(
    instance: Dict[str, Any],
    retrieval_sql_query: str,
    observable_event_data: pd.DataFrame,
    qe: MiniQE,
) -> Optional[Dict[str, Any]]:
    try:
        str_ids, obs_ids = retrieve_gold_events(
            retrieval_sql_query,
            observable_event_data,
            qe,
            instance.get("reference_date"),
        )
    except MiniQE.SQLError:
        return None
    return {
        "id": instance["id"],
        "gold_str_event_ids": str_ids,
        "gold_obs_event_ids": obs_ids,
    }


# --- drop_where_clause (dict version, same as DatasetCrossEncoderFactory) ---
def drop_where_clause(operator_tree_dict: Dict, retrieve_call: str, parents: Optional[List[str]] = None) -> bool:
    if parents is None:
        parents = []

    def matches_where_clause(qu_call: str) -> bool:
        if "date" in qu_call or "day" in qu_call or "time" in qu_call:
            return False
        return qu_call.startswith("FILTER")

    if operator_tree_dict["qu_input"] == retrieve_call:
        if len(parents) < 2:
            return False
        if parents[-1].startswith("SELECT") and matches_where_clause(parents[-2]):
            return True
        for i in range(len(parents) - 1):
            if parents[i + 1].startswith("SELECT") and matches_where_clause(parents[i]):
                return True
        return False
    new_parents = parents + [operator_tree_dict["qu_input"]]
    return any(
        drop_where_clause(child, retrieve_call, new_parents) for child in operator_tree_dict["childs"]
    )


def get_retrieve_calls(sub_dict: Dict) -> List[str]:
    out: List[str] = []
    if sub_dict["qu_input"].startswith("RETRIEVE"):
        out.append(sub_dict["qu_input"])
    for ch in sub_dict["childs"]:
        out.extend(get_retrieve_calls(ch))
    return out


def derived_pairs_for_instance(instance: Dict[str, Any]) -> Set[Tuple[str, str]]:
    sql_query = instance["sql_query"]
    pairs: Set[Tuple[str, str]] = set()
    for tree_dict in instance["operator_trees"]:
        retrieve_calls = list({c for c in get_retrieve_calls(tree_dict)})
        if not retrieve_calls:
            continue
        retrieve_call = retrieve_calls[0]
        retain_where = not drop_where_clause(tree_dict, retrieve_call)
        rsql = MiniQE.derive_retrieval_query(sql_query, retain_where)
        pairs.add((retrieve_call, rsql))
    return pairs


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def union_gold_obs_ids(
    instance: Dict[str, Any], obs_df: pd.DataFrame, qe: MiniQE
) -> Optional[Set[int]]:
    """多条 retrieval_sql 的 gold obs id 取并集。"""
    pairs = derived_pairs_for_instance(instance)
    if not pairs:
        return None
    u: Set[int] = set()
    for _rc, rsql in sorted(pairs):
        data = derive_gold_retrieval_data(instance, rsql, obs_df, qe)
        if data is None:
            continue
        u.update(int(x) for x in data["gold_obs_event_ids"])
    return u if u else None


def prepare_simple_pool(
    qu_path: Path,
    *,
    max_rows: Optional[int],
    randomize: bool,
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    pool = [r for r in load_jsonl(qu_path) if MiniQE.is_simple_query(r["sql_query"])]
    if randomize:
        if seed is not None:
            random.seed(seed)
        random.shuffle(pool)
    if max_rows is not None:
        pool = pool[:max_rows]
    return pool


def run_selfcheck_on_pool(
    pool: List[Dict[str, Any]],
    obs_df: pd.DataFrame,
    qe: MiniQE,
) -> Tuple[List[str], List[str]]:
    """返回 (failures, ok_lines)；ok_lines 每条对应一个 (qid, retrieval_sql) 的成功检查。"""
    valid_obs = set(int(x) for x in pd.to_numeric(obs_df["id"], errors="raise"))
    failures: List[str] = []
    ok_lines: List[str] = []
    for instance in pool:
        qid = instance.get("id", "?")
        pairs = derived_pairs_for_instance(instance)
        if not pairs:
            failures.append(f"{qid}: no RETRIEVE pairs")
            continue
        any_gold = False
        for _rc, rsql in sorted(pairs):
            data = derive_gold_retrieval_data(instance, rsql, obs_df, qe)
            if data is None:
                failures.append(f"{qid}: SQL error on {rsql[:100]}")
                continue
            gold = [int(x) for x in data["gold_obs_event_ids"]]
            bad = [g for g in gold if g not in valid_obs]
            if bad:
                failures.append(f"{qid}: ids not in obs: {bad[:6]}")
            if gold:
                any_gold = True
            ok_lines.append(f"OK {qid} n_gold={len(gold)} sql={rsql[:85]}...")
        if pairs and not any_gold:
            failures.append(f"{qid}: empty gold for all pairs")
    return failures, ok_lines


def build_export_rows(
    pool: List[Dict[str, Any]],
    obs_df: pd.DataFrame,
    qe: MiniQE,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """导出 queries.jsonl 行；relevant_ids 为并集排序后的 int 列表。"""
    rows: List[Dict[str, Any]] = []
    skipped = {"no_pairs": 0, "empty_gold": 0}
    for instance in pool:
        pairs = derived_pairs_for_instance(instance)
        if not pairs:
            skipped["no_pairs"] += 1
            continue
        u = union_gold_obs_ids(instance, obs_df, qe)
        if u is None:
            skipped["empty_gold"] += 1
            continue
        qtext = (instance.get("question") or "").strip()
        rk = qtext.split("\n", 1)[0].strip()[:200]
        rows.append(
            {
                "qid": instance.get("id", ""),
                "query": qtext,
                "query_key": rk,
                "relevant_ids": sorted(u),
            }
        )
    return rows, skipped
