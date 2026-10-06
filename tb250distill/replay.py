"""SQLite replay buffer（DESIGN.md「Replay DB 契約」）。

items   : teacher 前の raw item（生成時に一括 insert、決定的 seed）
teacher : teacher 出力（logits と probs の soft 分布を必須で保存）
teacher_errors : produce が採点に失敗した item の記録（契約外の補助テーブル。Student は読まない）

Teacher（produce.py）と Student（別プロセス）が同時に開くため WAL + busy_timeout を使う。
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Iterator, Sequence

SPLITS = ("train", "val", "test", "robust")
CATEGORIES = ("nli", "intent", "state_action", "ranking", "sentiment", "ambiguous", "agent_gate", "commonsense", "routing")
SOURCES = ("synth", "jcqa", "jnli", "massive", "wrime", "when2call", "routellm")  # 'public' は synth 以外すべての別名（produce の段階指定）
VARIANTS = ("perm", "cand_paraphrase", "ctx_paraphrase", "irrelevant_ctx", "ambiguous", "unseen_cand")

SCHEMA = """
CREATE TABLE IF NOT EXISTS items(
  item_id INTEGER PRIMARY KEY,
  split TEXT NOT NULL,
  category TEXT NOT NULL,
  lang TEXT NOT NULL,
  context TEXT NOT NULL,
  question TEXT NOT NULL,
  candidates TEXT NOT NULL,
  gold INTEGER,
  variant_of INTEGER,
  variant TEXT,
  gen_seed INTEGER NOT NULL,
  source TEXT NOT NULL DEFAULT 'synth',
  extra TEXT
);
CREATE TABLE IF NOT EXISTS teacher(
  item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
  teacher_model TEXT NOT NULL,
  method TEXT NOT NULL,
  logits TEXT NOT NULL,
  probs TEXT NOT NULL,
  raw TEXT,
  latency_ms REAL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS teacher_errors(
  item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
  attempts INTEGER NOT NULL DEFAULT 1,
  error TEXT,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_split ON items(split, item_id);
CREATE INDEX IF NOT EXISTS idx_items_variant_of ON items(variant_of);
"""

# 既存 DB（source/extra 列が無い）への追加列。init_schema が ALTER TABLE で足す（既存行は source='synth'）。
MIGRATIONS = (
    ("source", "ALTER TABLE items ADD COLUMN source TEXT NOT NULL DEFAULT 'synth'"),
    ("extra", "ALTER TABLE items ADD COLUMN extra TEXT"),
)
# source 列を使う索引は列追加後に作る（旧 DB で CREATE INDEX が失敗しないように）。
POST_MIGRATION_SQL = "CREATE INDEX IF NOT EXISTS idx_items_source ON items(source, split, item_id);"

ITEM_COLUMNS = (
    "item_id", "split", "category", "lang", "context", "question",
    "candidates", "gold", "variant_of", "variant", "gen_seed", "source", "extra",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def connect(path: str, *, readonly: bool = False, timeout: float = 60.0) -> sqlite3.Connection:
    """DB を開く（WAL）。readonly=True でも WAL の -shm を使うため mode=ro の URI で開く。"""
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    else:
        conn = sqlite3.connect(path, timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    conn.row_factory = sqlite3.Row
    if not readonly:
        init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(items)")}
    for col, ddl in MIGRATIONS:
        if col not in have:
            conn.execute(ddl)
    conn.executescript(POST_MIGRATION_SQL)
    conn.commit()


def _row_to_tuple(row: dict | Sequence) -> tuple:
    if isinstance(row, dict):
        cands = row["candidates"]
        if not isinstance(cands, str):
            cands = json.dumps(list(cands), ensure_ascii=False)
        extra = row.get("extra")
        if extra is not None and not isinstance(extra, str):
            extra = json.dumps(extra, ensure_ascii=False, sort_keys=True)
        return (
            row.get("item_id"), row["split"], row["category"], row["lang"],
            row["context"], row["question"], cands, row.get("gold"),
            row.get("variant_of"), row.get("variant"), int(row["gen_seed"]),
            row.get("source") or "synth", extra,
        )
    t = tuple(row)
    if len(t) == 11:  # 旧形式（source/extra 無し）
        t += ("synth", None)
    return t


def insert_items(conn: sqlite3.Connection, rows: Iterable[dict | Sequence]) -> int:
    """raw item を一括 insert。row は dict（item_id 省略可＝自動採番）。件数を返す。

    candidates は list[str] でも JSON 文字列でも良い。item_id が既存と衝突すると IntegrityError
    （部分挿入を避けるため単一トランザクション）。
    """
    tuples = [_row_to_tuple(r) for r in rows]
    for t in tuples:
        cands = json.loads(t[6])
        if not (1 <= len(cands) <= 16) or not all(isinstance(c, str) and c for c in cands):
            raise ValueError(f"bad candidates: {t[6]!r}")
        if t[7] is not None and not (0 <= t[7] < len(cands)):
            raise ValueError(f"gold out of range: {t[7]} for {t[6]!r}")
    with conn:
        conn.executemany(
            f"INSERT INTO items({','.join(ITEM_COLUMNS)}) VALUES ({','.join('?' * len(ITEM_COLUMNS))})",
            tuples,
        )
    return len(tuples)


def _decode_item(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["candidates"] = json.loads(d["candidates"])
    if isinstance(d.get("extra"), str):
        d["extra"] = json.loads(d["extra"])
    return d


def source_clause(sources: Sequence[str] | None, alias: str = "i") -> tuple[str, list]:
    """source 絞り込みの SQL 断片と params。'public' は synth 以外すべて、それ以外は完全一致（複数は OR）。"""
    if not sources:
        return "", []
    conds, params = [], []
    exact = []
    for s in sources:
        if s == "public":
            conds.append(f"{alias}.source != 'synth'")
        else:
            exact.append(s)
    if exact:
        conds.append(f"{alias}.source IN ({','.join('?' * len(exact))})")
        params += exact
    return "AND (" + " OR ".join(conds) + ") ", params


def _split_order_sql(splits: Sequence[str]) -> str:
    whens = " ".join(f"WHEN '{s}' THEN {i}" for i, s in enumerate(splits))
    return f"CASE i.split {whens} ELSE {len(splits)} END"


def pending_items(
    conn: sqlite3.Connection,
    limit: int,
    splits: Sequence[str] | None = None,
    max_attempts: int = 2,
    exclude: Iterable[int] = (),
    sources: Sequence[str] | None = None,
) -> list[dict]:
    """teacher 行が無い item を splits の優先順 → item_id 順で最大 limit 件返す。

    teacher_errors で attempts >= max_attempts の item は除外（無限リトライ防止）。
    exclude に今 run 中にスキップした item_id を渡せる。sources で source を絞れる（'public' = synth 以外）。
    """
    splits = list(splits) if splits else list(SPLITS)
    ph = ",".join("?" * len(splits))
    params: list = list(splits)
    sql = (
        "SELECT i.* FROM items i "
        "LEFT JOIN teacher t ON t.item_id = i.item_id "
        "LEFT JOIN teacher_errors e ON e.item_id = i.item_id "
        f"WHERE t.item_id IS NULL AND i.split IN ({ph}) "
        "AND (e.item_id IS NULL OR e.attempts < ?) "
    )
    params.append(max_attempts)
    sc, sp = source_clause(sources)
    sql += sc
    params += sp
    excl = list(exclude)
    if excl:
        sql += f"AND i.item_id NOT IN ({','.join('?' * len(excl))}) "
        params += excl
    sql += f"ORDER BY {_split_order_sql(splits)}, i.item_id LIMIT ?"
    params.append(limit)
    return [_decode_item(r) for r in conn.execute(sql, params)]


def write_teacher(
    conn: sqlite3.Connection,
    item_id: int,
    teacher_model: str,
    method: str,
    logits: Sequence[float],
    probs: Sequence[float],
    raw: dict | None = None,
    latency_ms: float | None = None,
    *,
    commit: bool = True,
) -> None:
    """teacher 出力を保存（INSERT OR REPLACE）。logits/probs は必須で、有限かつ同長であること。"""
    if len(logits) != len(probs) or not logits:
        raise ValueError("logits/probs length mismatch")
    if not all(math.isfinite(x) for x in logits) or not all(math.isfinite(x) for x in probs):
        raise ValueError("non-finite logits/probs")
    if abs(sum(probs) - 1.0) > 1e-4:
        raise ValueError(f"probs do not sum to 1: {sum(probs)}")
    conn.execute(
        "INSERT OR REPLACE INTO teacher(item_id, teacher_model, method, logits, probs, raw, latency_ms, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            item_id, teacher_model, method,
            json.dumps([float(x) for x in logits]),
            json.dumps([float(x) for x in probs]),
            None if raw is None else json.dumps(raw, ensure_ascii=False),
            latency_ms, _now(),
        ),
    )
    conn.execute("DELETE FROM teacher_errors WHERE item_id=?", (item_id,))
    if commit:
        conn.commit()


def write_error(conn: sqlite3.Connection, item_id: int, error: str, *, commit: bool = True) -> int:
    """採点失敗を記録し、現在の attempts を返す。"""
    conn.execute(
        "INSERT INTO teacher_errors(item_id, attempts, error, updated_at) VALUES (?,1,?,?) "
        "ON CONFLICT(item_id) DO UPDATE SET attempts=attempts+1, error=excluded.error, updated_at=excluded.updated_at",
        (item_id, error[:500], _now()),
    )
    if commit:
        conn.commit()
    return conn.execute("SELECT attempts FROM teacher_errors WHERE item_id=?", (item_id,)).fetchone()[0]


def iter_scored(
    conn: sqlite3.Connection, split: str | Sequence[str] | None = None, batch: int = 2000,
    sources: Sequence[str] | None = None,
) -> Iterator[dict]:
    """teacher 行がある item だけを item_id 順に yield（Student 用）。

    各 dict: items の全列（candidates は list）+ teacher_model, method, logits(list), probs(list), raw(dict|None), latency_ms。
    """
    if split is None:
        splits: list[str] = []
    elif isinstance(split, str):
        splits = [split]
    else:
        splits = list(split)
    where = f"AND i.split IN ({','.join('?' * len(splits))}) " if splits else ""
    sc, sp = source_clause(sources)
    where += sc
    last = -1
    while True:
        rows = conn.execute(
            "SELECT i.*, t.teacher_model, t.method, t.logits AS t_logits, t.probs AS t_probs, "
            "t.raw AS t_raw, t.latency_ms FROM items i JOIN teacher t ON t.item_id = i.item_id "
            f"WHERE i.item_id > ? {where}ORDER BY i.item_id LIMIT ?",
            [last, *splits, *sp, batch],
        ).fetchall()
        if not rows:
            return
        for r in rows:
            d = _decode_item(r)
            d["logits"] = json.loads(d.pop("t_logits"))
            d["probs"] = json.loads(d.pop("t_probs"))
            raw = d.pop("t_raw")
            d["raw"] = json.loads(raw) if raw else None
            last = d["item_id"]
            yield d


def get_items(conn: sqlite3.Connection, item_ids: Sequence[int]) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(item_ids), 500):
        chunk = list(item_ids[i : i + 500])
        rows = conn.execute(
            f"SELECT * FROM items WHERE item_id IN ({','.join('?' * len(chunk))}) ORDER BY item_id", chunk
        ).fetchall()
        out += [_decode_item(r) for r in rows]
    return out


def counts(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """split ごとの total / scored / errors。"""
    res: dict[str, dict[str, int]] = {}
    for r in conn.execute(
        "SELECT i.split AS split, COUNT(*) AS total, COUNT(t.item_id) AS scored, "
        "SUM(CASE WHEN t.item_id IS NULL AND e.item_id IS NOT NULL THEN 1 ELSE 0 END) AS errors "
        "FROM items i LEFT JOIN teacher t ON t.item_id=i.item_id LEFT JOIN teacher_errors e ON e.item_id=i.item_id "
        "GROUP BY i.split"
    ):
        res[r["split"]] = {"total": r["total"], "scored": r["scored"], "errors": r["errors"] or 0}
    return res


def scored_count(conn: sqlite3.Connection, split: str | None = None, sources: Sequence[str] | None = None) -> int:
    sc, sp = source_clause(sources)
    if split:
        return conn.execute(
            f"SELECT COUNT(*) FROM teacher t JOIN items i ON i.item_id=t.item_id WHERE i.split=? {sc}", (split, *sp)
        ).fetchone()[0]
    if sources:
        return conn.execute(
            f"SELECT COUNT(*) FROM teacher t JOIN items i ON i.item_id=t.item_id WHERE 1=1 {sc}", sp
        ).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM teacher").fetchone()[0]


def counts_by_source(conn: sqlite3.Connection) -> dict[str, dict[str, dict[str, int]]]:
    """source -> split -> {total, scored}。"""
    res: dict[str, dict[str, dict[str, int]]] = {}
    for r in conn.execute(
        "SELECT i.source AS source, i.split AS split, COUNT(*) AS total, COUNT(t.item_id) AS scored "
        "FROM items i LEFT JOIN teacher t ON t.item_id=i.item_id GROUP BY i.source, i.split"
    ):
        res.setdefault(r["source"], {})[r["split"]] = {"total": r["total"], "scored": r["scored"]}
    return res


def existing_keys(conn: sqlite3.Connection) -> set[tuple[str, str, str]]:
    """重複排除用: 既存 item の (context, question, candidates JSON) 集合。"""
    return {(r[0], r[1], r[2]) for r in conn.execute("SELECT context, question, candidates FROM items")}


def max_item_id(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COALESCE(MAX(item_id), 0) FROM items").fetchone()[0]


if __name__ == "__main__":  # 簡易ステータス表示: python -m tb250distill.replay data/replay.sqlite
    import sys

    c = connect(sys.argv[1] if len(sys.argv) > 1 else "data/replay.sqlite", readonly=True)
    print(json.dumps(counts(c), indent=1))
