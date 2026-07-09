import os
import re
import shutil
import sqlite3
import logging
import tempfile
import threading

from src.core.constants import (
    PROJECT_ROOT,
    DB_TIMEOUT, DB_CACHE_SIZE, DB_MMAP_SIZE,
    DB_UPSERT_BATCH, DB_QUERY_BATCH, DB_STREAM_BATCH,
)

logger = logging.getLogger("ai-engine")

DB_PATH = os.path.join(PROJECT_ROOT, "data", "admin.db")

# ── 커넥션 풀: 스레드별 커넥션 재사용 ──
_thread_local = threading.local()
_db_initialized = False
_init_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
# 시간대(timezone) 정책: 모든 timestamp 칼럼은 UTC로 저장한다.
#   - SQLite DEFAULT는 datetime('now') 사용 (자동으로 UTC).
#   - Python에서 명시적 INSERT/UPDATE 시에도 datetime.now(timezone.utc) 사용.
#   - 프론트엔드(AdminDashboard.jsx의 formatDate)가 'Z' 접미를 붙여 UTC로 파싱한 뒤
#     사용자 로컬 시간대(KST)로 변환해 표시한다.
#   - KST 기준 월별 필터(예: bulk-delete)에서만 strftime(... , 'localtime')으로 변환.
# ─────────────────────────────────────────────────────────────

_CREATE_UPLOADS = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now')),
    a_exam_path TEXT,
    b_exam_path TEXT,
    a_sago_path TEXT,
    b_sago_path TEXT,
    file_hash TEXT
);
"""

_CREATE_TRAINING_RUNS = """
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER REFERENCES uploads(id),
    status TEXT DEFAULT 'pending',
    step_detail TEXT,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    metrics_json TEXT,
    error_message TEXT
);
"""

_CREATE_MODEL_VERSIONS = """
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES training_runs(id),
    version_label TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    is_active INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    size_bytes INTEGER DEFAULT 0,
    metrics_json TEXT,
    description TEXT DEFAULT ''
);
"""

_CREATE_UPLOAD_METADATA = """
CREATE TABLE IF NOT EXISTS upload_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER REFERENCES uploads(id),
    domain TEXT NOT NULL,
    file_type TEXT NOT NULL,
    record_count INTEGER DEFAULT 0,
    date_from TEXT,
    date_to TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_EXAM_RECORDS = """
CREATE TABLE IF NOT EXISTS exam_records (
    test_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL CHECK(domain IN ('A','B')),
    primary_key TEXT NOT NULL,
    age TEXT,
    exam_age INTEGER,
    birth_yyyymmdd TEXT,
    test_date TEXT,
    features_json TEXT NOT NULL,
    label INTEGER,
    upload_id INTEGER REFERENCES uploads(id),
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_EXAM_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_exam_domain ON exam_records(domain)",
    "CREATE INDEX IF NOT EXISTS idx_exam_pk ON exam_records(primary_key)",
]

_CREATE_SAGO_RECORDS = """
CREATE TABLE IF NOT EXISTS sago_records (
    primary_key TEXT NOT NULL,
    acc_date TEXT NOT NULL,
    acc_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    seq INTEGER NOT NULL DEFAULT 1,
    count_1 INTEGER DEFAULT 0,
    count_2 INTEGER DEFAULT 0,
    count_3 INTEGER DEFAULT 0,
    count_4 INTEGER DEFAULT 0,
    count_5 INTEGER DEFAULT 0,
    count_6 INTEGER DEFAULT 0,
    upload_id INTEGER REFERENCES uploads(id),
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (primary_key, acc_date, acc_type, domain, seq)
);
"""

_CREATE_SAGO_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sago_pk ON sago_records(primary_key)",
]

_CREATE_MODEL_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS model_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES training_runs(id),
    artifact_key TEXT NOT NULL,
    artifact_type TEXT NOT NULL CHECK(artifact_type IN ('json','parquet','pkl')),
    data BLOB NOT NULL,
    size_bytes INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(run_id, artifact_key)
);
"""

_CREATE_MODEL_ARTIFACTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_model_artifacts_run ON model_artifacts(run_id)",
]

# 마이그레이션: uploads 테이블에 status, deleted_at 컬럼 추가
_MIGRATE_UPLOADS_STATUS = [
    "ALTER TABLE uploads ADD COLUMN status TEXT DEFAULT 'active'",
    "ALTER TABLE uploads ADD COLUMN deleted_at TEXT",
]


def _migrate_exam_records_add_exam_age(conn: sqlite3.Connection):
    """exam_records에 exam_age INTEGER 컬럼 추가 + 기존 Age 코드에서 중간값 백필."""
    cursor = conn.execute("PRAGMA table_info(exam_records)")
    cols = {row[1] for row in cursor.fetchall()}
    if "exam_age" in cols:
        return

    conn.execute("ALTER TABLE exam_records ADD COLUMN exam_age INTEGER")
    # Age 코드 → 중간값 백필: "30a"→32, "30b"→37, "40a"→42 등
    conn.execute("""
        UPDATE exam_records
        SET exam_age = CAST(SUBSTR(age, 1, LENGTH(age) - 1) AS INTEGER)
                     + CASE WHEN SUBSTR(age, -1) = 'a' THEN 2 ELSE 7 END
        WHERE exam_age IS NULL AND age IS NOT NULL AND LENGTH(age) >= 2
    """)
    conn.commit()
    logger.info("Migration: exam_records.exam_age 컬럼 추가 + 백필 완료")


def _migrate_exam_records_add_label(conn: sqlite3.Connection):
    """exam_records에 label INTEGER 컬럼 추가."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(exam_records)").fetchall()}
    if "label" in cols:
        return
    conn.execute("ALTER TABLE exam_records ADD COLUMN label INTEGER")
    conn.commit()
    logger.info("Migration: exam_records.label 컬럼 추가 완료")


def _migrate_exam_records_add_birth_yyyymmdd(conn: sqlite3.Connection):
    """exam_records에 birth_yyyymmdd TEXT 컬럼 추가.

    조회 시점에 RRN 없이도 정확한 현재 만나이를 계산하기 위함.
    기존 행은 NULL로 두고, main.py 조회에서 fallback 근사 계산을 사용한다.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(exam_records)").fetchall()}
    if "birth_yyyymmdd" in cols:
        return
    conn.execute("ALTER TABLE exam_records ADD COLUMN birth_yyyymmdd TEXT")
    conn.commit()
    logger.info("Migration: exam_records.birth_yyyymmdd 컬럼 추가 완료")


def _migrate_model_artifacts_add_pkl(conn: sqlite3.Connection):
    """model_artifacts CHECK 제약에 'pkl' 타입 추가. 테이블 재생성 방식."""
    cursor = conn.execute("PRAGMA table_info(model_artifacts)")
    if not cursor.fetchall():
        return  # 테이블 없음

    # pkl 타입 삽입 테스트
    try:
        conn.execute(
            "INSERT INTO model_artifacts (run_id, artifact_key, artifact_type, data, size_bytes) "
            "VALUES (-999, '__check_pkl__', 'pkl', X'00', 0)"
        )
        conn.execute("DELETE FROM model_artifacts WHERE artifact_key = '__check_pkl__'")
        conn.commit()
        return  # 이미 pkl 지원됨
    except sqlite3.IntegrityError:
        conn.rollback()

    # 테이블 재생성 (CHECK 제약 변경)
    conn.execute("ALTER TABLE model_artifacts RENAME TO _model_artifacts_old")
    conn.execute(_CREATE_MODEL_ARTIFACTS)
    for idx_sql in _CREATE_MODEL_ARTIFACTS_INDEXES:
        conn.execute(idx_sql)
    conn.execute(
        "INSERT INTO model_artifacts (id, run_id, artifact_key, artifact_type, data, size_bytes, created_at) "
        "SELECT id, run_id, artifact_key, artifact_type, data, size_bytes, created_at FROM _model_artifacts_old"
    )
    conn.execute("DROP TABLE _model_artifacts_old")
    conn.commit()
    logger.info("Migration: model_artifacts CHECK 제약에 'pkl' 타입 추가 완료")


def _migrate_sago_pk_add_seq(conn: sqlite3.Connection):
    """sago_records PRIMARY KEY에 seq 추가 마이그레이션.

    기존 PK: (primary_key, acc_date, acc_type, domain) 또는 그 이전 버전
    새 PK: (primary_key, acc_date, acc_type, domain, seq)

    동일 (pk, date, type, domain)에 여러 사고/위반 기록이 있을 수 있으므로
    seq로 구분하여 데이터 유실을 방지한다.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='sago_records'"
    ).fetchone()
    if row is None:
        return  # 테이블 미존재 — CREATE TABLE에서 생성됨

    create_sql = row[0] or ""
    pk_match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", create_sql, re.IGNORECASE)
    if pk_match:
        pk_cols = [c.strip().lower() for c in pk_match.group(1).split(",")]
        if "seq" in pk_cols:
            return  # 이미 마이그레이션 완료

    logger.info("[Migration] sago_records PK에 seq 추가 중...")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sago_records_new (
            primary_key TEXT NOT NULL,
            acc_date TEXT NOT NULL,
            acc_type TEXT NOT NULL,
            domain TEXT NOT NULL,
            seq INTEGER NOT NULL DEFAULT 1,
            count_1 INTEGER DEFAULT 0,
            count_2 INTEGER DEFAULT 0,
            count_3 INTEGER DEFAULT 0,
            count_4 INTEGER DEFAULT 0,
            count_5 INTEGER DEFAULT 0,
            count_6 INTEGER DEFAULT 0,
            upload_id INTEGER REFERENCES uploads(id),
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (primary_key, acc_date, acc_type, domain, seq)
        )
    """)
    # 기존 데이터 이관 (seq=1로 삽입, 기존 PK에 domain이 없을 수도 있음)
    old_cols = [c.strip().lower() for c in
                re.search(r"\((.*?)\)", create_sql, re.DOTALL).group(1).split(",")
                if "primary" not in c.lower() and "key" not in c.lower()]
    if "domain" in [c.split()[0] for c in old_cols]:
        conn.execute("""
            INSERT OR IGNORE INTO sago_records_new
            (primary_key, acc_date, acc_type, domain, seq,
             count_1, count_2, count_3, count_4, count_5, count_6,
             upload_id, created_at)
            SELECT primary_key, acc_date, acc_type, domain, 1,
                   count_1, count_2, count_3, count_4, count_5, count_6,
                   upload_id, created_at
            FROM sago_records
        """)
    else:
        conn.execute("""
            INSERT OR IGNORE INTO sago_records_new
            (primary_key, acc_date, acc_type, domain, seq,
             count_1, count_2, count_3, count_4, count_5, count_6,
             upload_id, created_at)
            SELECT primary_key, acc_date, acc_type, '', 1,
                   count_1, count_2, count_3, count_4, count_5, count_6,
                   upload_id, created_at
            FROM sago_records
        """)

    conn.execute("DROP TABLE sago_records")
    conn.execute("ALTER TABLE sago_records_new RENAME TO sago_records")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sago_pk ON sago_records(primary_key)")
    conn.commit()

    new_count = conn.execute("SELECT COUNT(*) as cnt FROM sago_records").fetchone()["cnt"]
    logger.info(f"[Migration] sago_records seq 마이그레이션 완료 (레코드: {new_count})")
    logger.warning(
        "[Migration] 기존 데이터는 seq=1로 설정됨 — "
        "동일 키 다중 기록 보존을 위해 데이터 재업로드를 권장합니다."
    )


def _run_migrations(conn: sqlite3.Connection):
    """스키마 마이그레이션을 안전하게 실행한다."""
    cursor = conn.execute("PRAGMA table_info(uploads)")
    cols = {row[1] for row in cursor.fetchall()}

    for sql in _MIGRATE_UPLOADS_STATUS:
        col_name = sql.split("ADD COLUMN ")[1].split(" ")[0]
        if col_name not in cols:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:
                logger.debug(f"Migration skip ({col_name}): {e}")

    conn.commit()

    # sago_records PK 마이그레이션 (domain 추가)
    _migrate_sago_pk_add_seq(conn)

    # exam_records에 exam_age 컬럼 추가 + 기존 데이터 백필
    _migrate_exam_records_add_exam_age(conn)

    # exam_records에 label 컬럼 추가
    _migrate_exam_records_add_label(conn)

    # exam_records에 birth_yyyymmdd 컬럼 추가
    _migrate_exam_records_add_birth_yyyymmdd(conn)

    # model_artifacts CHECK 제약에 'pkl' 추가
    _migrate_model_artifacts_add_pkl(conn)



def _init_schema(conn: sqlite3.Connection):
    """스키마와 인덱스를 1회만 초기화한다."""
    conn.execute(_CREATE_UPLOADS)
    conn.execute(_CREATE_TRAINING_RUNS)
    conn.execute(_CREATE_MODEL_VERSIONS)
    conn.execute(_CREATE_UPLOAD_METADATA)
    conn.execute(_CREATE_EXAM_RECORDS)
    for idx_sql in _CREATE_EXAM_INDEXES:
        conn.execute(idx_sql)
    conn.execute(_CREATE_SAGO_RECORDS)
    for idx_sql in _CREATE_SAGO_INDEXES:
        conn.execute(idx_sql)
    conn.execute(_CREATE_MODEL_ARTIFACTS)
    for idx_sql in _CREATE_MODEL_ARTIFACTS_INDEXES:
        conn.execute(idx_sql)
    conn.commit()
    _run_migrations(conn)


def get_db() -> sqlite3.Connection:
    """스레드별 커넥션을 재사용한다. 스키마 초기화는 1회만."""
    global _db_initialized
    conn = getattr(_thread_local, "conn", None)

    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.ProgrammingError:
            conn = None

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # int() 파싱 보장 → SQL injection 불가 (PRAGMA는 ? 바인딩 미지원)
    conn.execute(f"PRAGMA cache_size={DB_CACHE_SIZE}")
    conn.execute(f"PRAGMA mmap_size={DB_MMAP_SIZE}")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")
    _thread_local.conn = conn

    with _init_lock:
        if not _db_initialized:
            _init_schema(conn)
            _db_initialized = True

    return conn


def close_db():
    """현재 스레드의 커넥션을 닫는다 (서버 종료 시)."""
    conn = getattr(_thread_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _thread_local.conn = None



def insert_upload(
    a_exam_path: str,
    b_exam_path: str,
    a_sago_path: str,
    b_sago_path: str,
    file_hash: str,
) -> int:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO uploads (a_exam_path, b_exam_path, a_sago_path, b_sago_path, file_hash) VALUES (?, ?, ?, ?, ?)",
        (a_exam_path, b_exam_path, a_sago_path, b_sago_path, file_hash),
    )
    conn.commit()
    return cur.lastrowid


def insert_training_run(upload_id: int) -> int:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO training_runs (upload_id) VALUES (?)",
        (upload_id,),
    )
    conn.commit()
    return cur.lastrowid


def update_training_run(run_id: int, **kwargs) -> None:
    allowed = {"status", "step_detail", "completed_at", "metrics_json", "error_message"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [run_id]
    conn = get_db()
    conn.execute(
        f"UPDATE training_runs SET {set_clause} WHERE id = ?",
        values,
    )
    conn.commit()


def get_training_run(run_id: int) -> dict:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM training_runs WHERE id = ?", (run_id,)
    ).fetchone()
    return dict(row) if row else {}


def get_latest_training_run() -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM training_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def get_all_training_runs() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM training_runs ORDER BY id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


# ── model_versions CRUD ──


def insert_model_version(
    run_id: int | None,
    version_label: str,
    artifact_path: str,
    size_bytes: int = 0,
    metrics_json: str | None = None,
    description: str = "",
) -> int:
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO model_versions
           (run_id, version_label, artifact_path, size_bytes, metrics_json, description)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_id, version_label, artifact_path, size_bytes, metrics_json, description),
    )
    conn.commit()
    return cur.lastrowid


def get_active_model_version() -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM model_versions WHERE is_active = 1 LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def set_active_model_version(version_id: int) -> None:
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE model_versions SET is_active = 0")
        cur = conn.execute("UPDATE model_versions SET is_active = 1 WHERE id = ?", (version_id,))
        if cur.rowcount == 0:
            conn.rollback()
            raise ValueError(f"Model version {version_id} not found")
        conn.commit()
    except ValueError:
        raise
    except Exception:
        conn.rollback()
        raise


def get_all_model_versions() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        """SELECT mv.*, tr.started_at as training_started_at
           FROM model_versions mv
           LEFT JOIN training_runs tr ON mv.run_id = tr.id
           ORDER BY mv.id DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_model_version(version_id: int) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM model_versions WHERE id = ?", (version_id,)
    ).fetchone()
    return dict(row) if row else None


def delete_model_version(version_id: int) -> None:
    conn = get_db()
    conn.execute("DELETE FROM model_versions WHERE id = ?", (version_id,))
    conn.commit()


# ── model_artifacts CRUD ──


def store_artifact(run_id: int, artifact_key: str, artifact_type: str, data: bytes) -> int:
    """단건 아티팩트 BLOB 저장. 반환: artifact id."""
    conn = get_db()
    cur = conn.execute(
        """INSERT OR REPLACE INTO model_artifacts
           (run_id, artifact_key, artifact_type, data, size_bytes)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, artifact_key, artifact_type, sqlite3.Binary(data), len(data)),
    )
    conn.commit()
    return cur.lastrowid


def store_artifacts_batch(run_id: int, artifacts: list[tuple[str, str, bytes]]) -> int:
    """트랜잭션 일괄 저장. artifacts: [(key, type, data), ...]. 반환: 저장 건수."""
    if not artifacts:
        return 0
    conn = get_db()
    params = [
        (run_id, key, atype, sqlite3.Binary(data), len(data))
        for key, atype, data in artifacts
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO model_artifacts
           (run_id, artifact_key, artifact_type, data, size_bytes)
           VALUES (?, ?, ?, ?, ?)""",
        params,
    )
    conn.commit()
    return len(params)


def load_artifact(run_id: int, artifact_key: str) -> bytes | None:
    """단건 아티팩트 BLOB 로딩. 없으면 None."""
    conn = get_db()
    row = conn.execute(
        "SELECT data FROM model_artifacts WHERE run_id = ? AND artifact_key = ?",
        (run_id, artifact_key),
    ).fetchone()
    return bytes(row["data"]) if row else None


def list_artifact_keys(run_id: int, prefix: str) -> list[str]:
    """run_id의 아티팩트 키 중 prefix로 시작하는 것들을 정렬하여 반환."""
    conn = get_db()
    rows = conn.execute(
        "SELECT artifact_key FROM model_artifacts WHERE run_id = ? AND artifact_key LIKE ? ORDER BY artifact_key",
        (run_id, prefix + "%"),
    ).fetchall()
    return [r["artifact_key"] for r in rows]


def delete_artifacts_by_run(run_id: int) -> int:
    """특정 run_id의 모든 아티팩트 삭제. 반환: 삭제 건수."""
    conn = get_db()
    cur = conn.execute("DELETE FROM model_artifacts WHERE run_id = ?", (run_id,))
    conn.commit()
    return cur.rowcount


def get_run_id_for_active_version() -> int | None:
    """활성 모델 버전의 run_id 반환. 없으면 None."""
    v = get_active_model_version()
    return v["run_id"] if v else None


def get_artifacts_total_size() -> int:
    """model_artifacts 테이블의 전체 BLOB 크기(bytes) 반환."""
    conn = get_db()
    row = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) as total FROM model_artifacts").fetchone()
    return row["total"]


# ── upload_metadata CRUD ──


def insert_upload_metadata(
    upload_id: int,
    domain: str,
    file_type: str,
    record_count: int = 0,
    date_from: str | None = None,
    date_to: str | None = None,
) -> int:
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO upload_metadata
           (upload_id, domain, file_type, record_count, date_from, date_to)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (upload_id, domain, file_type, record_count, date_from, date_to),
    )
    conn.commit()
    return cur.lastrowid


def get_upload_metadata(upload_id: int) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM upload_metadata WHERE upload_id = ?", (upload_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_uploads_with_metadata() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT u.*, m.id AS meta_id, m.file_type, m.domain, m.date_from, m.date_to, m.record_count "
        "FROM uploads u LEFT JOIN upload_metadata m ON u.id = m.upload_id "
        "ORDER BY u.id DESC, m.id"
    ).fetchall()

    # upload_id별 그룹핑
    uploads_map: dict = {}
    for r in rows:
        r = dict(r)
        uid = r["id"]
        if uid not in uploads_map:
            uploads_map[uid] = {
                k: r[k] for k in r
                if k not in ("meta_id", "file_type", "domain", "date_from", "date_to", "record_count")
            }
            uploads_map[uid]["metadata"] = []
        if r["meta_id"] is not None:
            uploads_map[uid]["metadata"].append({
                "id": r["meta_id"], "file_type": r["file_type"], "domain": r["domain"],
                "date_from": r["date_from"], "date_to": r["date_to"], "record_count": r["record_count"],
            })
    return list(uploads_map.values())


def hard_delete_upload(upload_id: int) -> dict:
    """업로드와 종속된 모든 데이터를 한 트랜잭션으로 삭제한다.

    삭제 대상:
    - exam_records (upload_id FK)
    - sago_records (upload_id FK)
    - upload_metadata (upload_id FK)
    - training_runs.upload_id NULL 처리 (FK 무효화)
    - uploads 본 행

    한 conn.commit()으로 묶여 부분 실패 없음 (sqlite가 자동 rollback).
    """
    conn = get_db()
    cur_exam = conn.execute("DELETE FROM exam_records WHERE upload_id = ?", (upload_id,))
    cur_sago = conn.execute("DELETE FROM sago_records WHERE upload_id = ?", (upload_id,))
    conn.execute("DELETE FROM upload_metadata WHERE upload_id = ?", (upload_id,))
    conn.execute(
        "UPDATE training_runs SET upload_id = NULL WHERE upload_id = ?",
        (upload_id,),
    )
    conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    return {"exam_deleted": cur_exam.rowcount, "sago_deleted": cur_sago.rowcount}


def get_all_active_file_paths() -> list[dict]:
    """전체 업로드의 파일 경로를 반환한다 (완전 삭제 후 삭제되지 않은 행만 존재)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, a_exam_path, b_exam_path, a_sago_path, b_sago_path "
        "FROM uploads ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def bulk_hard_delete_uploads(upload_ids: list[int]) -> int:
    """다수의 업로드와 종속된 모든 데이터를 한 트랜잭션으로 삭제한다.

    hard_delete_upload와 동일 동작이지만 IN 절로 일괄 처리. 삭제된 uploads 행 수 반환.
    """
    if not upload_ids:
        return 0
    conn = get_db()
    placeholders = ",".join("?" for _ in upload_ids)
    conn.execute(
        f"DELETE FROM exam_records WHERE upload_id IN ({placeholders})",
        upload_ids,
    )
    conn.execute(
        f"DELETE FROM sago_records WHERE upload_id IN ({placeholders})",
        upload_ids,
    )
    conn.execute(
        f"DELETE FROM upload_metadata WHERE upload_id IN ({placeholders})",
        upload_ids,
    )
    conn.execute(
        f"UPDATE training_runs SET upload_id = NULL WHERE upload_id IN ({placeholders})",
        upload_ids,
    )
    cur = conn.execute(
        f"DELETE FROM uploads WHERE id IN ({placeholders})",
        upload_ids,
    )
    conn.commit()
    return cur.rowcount


def get_active_uploads_by_date_range(
    date_from: str, date_to: str
) -> list[dict]:
    """업로드 날짜(created_at) 기준으로 업로드를 조회한다.

    date_from/date_to: 'YYYY-MM' 형식.
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM uploads "
        # created_at은 UTC로 저장되므로 'localtime'으로 변환해 KST 기준 월(月)과 비교한다.
        "WHERE strftime('%Y-%m', created_at, 'localtime') >= ? "
        "AND strftime('%Y-%m', created_at, 'localtime') <= ? "
        "ORDER BY id",
        (date_from, date_to),
    ).fetchall()
    return [dict(r) for r in rows]


def _compact_db():
    """WAL checkpoint(TRUNCATE) + VACUUM으로 admin.db main 파일을 즉시 압축한다.

    큰 cleanup 작업 후에만 호출(reset 함수들). 매 commit마다 부르면 안 됨 — VACUUM은 무거움.
    WAL checkpoint를 먼저 호출해야 main 파일이 최신 상태가 되고, 그 후 VACUUM이 빈 페이지를 정리한다.
    """
    conn = get_db()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")


def reset_all_uploads() -> int:
    """전체 업로드와 그 종속 데이터를 모두 삭제하고 자동 증가 카운터를 초기화한다.

    종속 테이블(exam_records, sago_records, upload_metadata)도 함께 비워 FK 무결성 보장.
    한 트랜잭션으로 묶여 부분 실패 없음.
    """
    conn = get_db()
    # FK 종속 테이블부터 삭제 (upload_id 참조)
    conn.execute("DELETE FROM exam_records")
    conn.execute("DELETE FROM sago_records")
    conn.execute("DELETE FROM upload_metadata")
    conn.execute("UPDATE training_runs SET upload_id = NULL WHERE upload_id IS NOT NULL")
    cur = conn.execute("DELETE FROM uploads")
    # 자동 증가 초기화 (다음 업로드가 ID 1부터 시작)
    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('uploads', 'upload_metadata')")
    conn.commit()
    _compact_db()
    return cur.rowcount


def check_duplicate_hash(file_hash: str) -> list[dict]:
    """동일한 file_hash를 가진 업로드가 이미 존재하는지 확인한다."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, created_at FROM uploads WHERE file_hash = ?",
        (file_hash,),
    ).fetchall()
    return [dict(r) for r in rows]


def reset_all_training_runs() -> int:
    """전체 학습 실행과 그에 종속된 모델 아티팩트를 모두 삭제한다.

    model_artifacts.run_id가 NOT NULL FK라 training_runs 삭제 전에 함께 삭제해야
    FK constraint 위반이 안 난다. (단순히 NULL 할 수 없음)
    model_versions.run_id는 NULL 가능 → 무효화로 충분.
    """
    conn = get_db()
    # FK 종속 테이블부터: model_artifacts (NOT NULL FK)는 함께 삭제
    conn.execute("DELETE FROM model_artifacts")
    conn.execute("UPDATE model_versions SET run_id = NULL WHERE run_id IS NOT NULL")
    cur = conn.execute("DELETE FROM training_runs")
    conn.execute(
        "DELETE FROM sqlite_sequence WHERE name IN ('training_runs', 'model_artifacts')"
    )
    conn.commit()
    _compact_db()
    return cur.rowcount


def reset_all_model_versions() -> int:
    """전체 모델 버전을 완전 삭제한다. 삭제 수를 반환."""
    conn = get_db()
    conn.execute("DELETE FROM model_artifacts")
    cur = conn.execute("DELETE FROM model_versions")
    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('model_versions', 'model_artifacts')")
    conn.commit()
    _compact_db()
    return cur.rowcount


# ── exam_records / sago_records CRUD ──


def upsert_exam_records(records, upload_id: int, tuple_mode: bool = False) -> dict:
    """검사 레코드를 UPSERT한다. INSERT OR IGNORE로 test_id 충돌 시 무시 (기존 데이터 보존).

    Args:
        records: tuple_mode=True이면
                 (test_id, domain, pk, age, exam_age, birth_yyyymmdd, test_date, features_json, upload_id)
                 튜플 리스트, False이면 기존 dict 리스트.
        upload_id: 소유 업로드 ID.
        tuple_mode: True이면 admin.py에서 전달한 튜플 파라미터를 직접 사용.

    Returns:
        {"total": int, "new": int, "existing": int}
    """
    if not records:
        return {"total": 0, "new": 0, "existing": 0}

    conn = get_db()

    # COUNT(*) before — DB에 이미 있던 레코드 수
    count_before = conn.execute("SELECT COUNT(*) FROM exam_records").fetchone()[0]

    # 튜플 모드: admin.py에서 이미 9-튜플로 구성
    if tuple_mode:
        params = records
    else:
        params = [
            (r["test_id"], r["domain"], r["primary_key"], r.get("age", ""),
             r.get("exam_age"), r.get("birth_yyyymmdd", ""),
             r.get("test_date", ""), r["features_json"], upload_id)
            for r in records
        ]

    # 파일 내 중복 건수 (같은 test_id가 여러 번 등장)
    n_total = len(params)
    n_unique_in_batch = len({p[0] for p in params})
    n_dup_in_file = n_total - n_unique_in_batch

    # 벌크 INSERT 최적화: auto-checkpoint 비활성화 + PK 정렬
    conn.execute("PRAGMA wal_autocheckpoint=0")
    params.sort(key=lambda x: x[0])

    BATCH = DB_UPSERT_BATCH
    for i in range(0, len(params), BATCH):
        conn.executemany(
            """INSERT OR IGNORE INTO exam_records
               (test_id, domain, primary_key, age, exam_age, birth_yyyymmdd, test_date, features_json, upload_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            params[i:i + BATCH],
        )
    conn.commit()

    conn.execute("PRAGMA wal_autocheckpoint=1000")

    count_after = conn.execute("SELECT COUNT(*) FROM exam_records").fetchone()[0]
    n_db_new = count_after - count_before  # DB에 실제 추가된 행
    n_db_existing = n_unique_in_batch - n_db_new  # DB에 이미 있어서 교체된 행
    return {
        "total": n_total,
        "new": n_db_new,
        "existing": n_db_existing,
        "duplicates_in_file": n_dup_in_file,
    }


def upsert_sago_records(records: list[dict], upload_id: int) -> dict:
    """사고/위반 레코드를 UPSERT한다. PK=(primary_key, acc_date, acc_type, domain, seq).

    동일 (pk, date, type, domain)에 여러 기록이 있으면 seq로 구분하여 모두 보존한다.

    Args:
        records: primary_key, acc_date, acc_type, domain, count_1..count_6 포함 dict 리스트.
        upload_id: 소유 업로드 ID.

    Returns:
        {"total": int, "new": int, "existing": int}
    """
    if not records:
        return {"total": 0, "new": 0, "existing": 0}

    conn = get_db()

    # COUNT(*) before
    count_before = conn.execute("SELECT COUNT(*) FROM sago_records").fetchone()[0]

    # seq 할당: 동일 (pk, date, type, domain) 그룹 내 순번
    from collections import Counter
    key_counter = Counter()
    params_list = []
    for r in records:
        key = (r["primary_key"], r["acc_date"], r["acc_type"], r["domain"])
        key_counter[key] += 1
        seq = key_counter[key]
        params_list.append((
            r["primary_key"], r["acc_date"], r["acc_type"], r["domain"], seq,
            r.get("count_1", 0), r.get("count_2", 0), r.get("count_3", 0),
            r.get("count_4", 0), r.get("count_5", 0), r.get("count_6", 0),
            upload_id,
        ))

    n_total = len(params_list)

    # 벌크 INSERT 최적화: auto-checkpoint 비활성화 + PK 정렬
    conn.execute("PRAGMA wal_autocheckpoint=0")
    params_list.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    BATCH = DB_UPSERT_BATCH
    for i in range(0, len(params_list), BATCH):
        conn.executemany(
            """INSERT OR IGNORE INTO sago_records
               (primary_key, acc_date, acc_type, domain, seq,
                count_1, count_2, count_3, count_4, count_5, count_6, upload_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            params_list[i:i + BATCH],
        )
    conn.commit()

    conn.execute("PRAGMA wal_autocheckpoint=1000")

    count_after = conn.execute("SELECT COUNT(*) FROM sago_records").fetchone()[0]
    n_new = count_after - count_before
    return {
        "total": n_total,
        "new": n_new,
        "existing": n_total - n_new,
    }


def get_exam_records_by_pk(primary_key: str) -> list[dict]:
    """특정 PrimaryKey의 모든 검사 레코드를 반환한다."""
    conn = get_db()
    rows = conn.execute(
        "SELECT test_id, domain, age, exam_age, birth_yyyymmdd, test_date, features_json "
        "FROM exam_records WHERE primary_key = ? ORDER BY test_date",
        (primary_key,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_exam_labels(label_pairs: list[tuple[str, int]]):
    """학습 후 exam_records의 label을 일괄 업데이트한다. [(test_id, label), ...]"""
    if not label_pairs:
        return
    conn = get_db()
    BATCH = DB_UPSERT_BATCH
    for i in range(0, len(label_pairs), BATCH):
        conn.executemany(
            "UPDATE exam_records SET label = ? WHERE test_id = ?",
            [(lbl, tid) for tid, lbl in label_pairs[i:i + BATCH]],
        )
    conn.commit()
    logger.info("exam_records label 업데이트: %d건", len(label_pairs))


def get_latest_exam_by_pks(
    primary_keys: list[str],
    exclude_test_ids: list[str] | None = None,
) -> dict[str, dict]:
    """여러 PrimaryKey의 가장 최근 검사 기록을 반환한다.

    exclude_test_ids에 포함된 test_id는 제외(self-prev 방지) — 추론 호출 시
    추론 대상 검사가 이미 DB에 저장돼 있을 경우 자기 자신을 prev로 가져가는 것을
    막는다.

    Returns:
        {pk: {test_id, domain, test_date, features_json, label, age}} — 각 PK의 최신 기록
    """
    if not primary_keys:
        return {}
    conn = get_db()
    exclude_set = set(exclude_test_ids or [])
    result = {}
    BATCH = DB_QUERY_BATCH
    for i in range(0, len(primary_keys), BATCH):
        batch = primary_keys[i:i + BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"""SELECT primary_key, test_id, domain, test_date, features_json, label, age
                FROM exam_records
                WHERE primary_key IN ({placeholders})
                ORDER BY primary_key, test_date DESC
            """,
            batch,
        ).fetchall()
        for r in rows:
            pk = r["primary_key"]
            if pk in result:
                continue
            if r["test_id"] in exclude_set:
                continue
            result[pk] = dict(r)
    return result


def get_label_cummean_by_pks(
    primary_keys: list[str],
    exclude_test_ids: list[str] | None = None,
) -> dict[str, float]:
    """학습의 before_label_cummean(seq_trainer.py:543-554)과 동일한 값을 PK별로 계산.

    학습은 PK 그룹 내 expanding mean(2/3 제외)을 누적하지만, 추론 시점은 새 검사 직전이므로
    "이 PK의 모든 이전 검사 라벨(0/1만, 신규/증강 제외)의 평균" 한 값으로 충분하다.
    """
    if not primary_keys:
        return {}
    conn = get_db()
    exclude_set = set(exclude_test_ids or [])
    result: dict[str, float] = {}
    BATCH = DB_QUERY_BATCH
    for i in range(0, len(primary_keys), BATCH):
        batch = primary_keys[i:i + BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"""SELECT primary_key, test_id, label
                FROM exam_records
                WHERE primary_key IN ({placeholders})
                  AND label IN (0, 1)
            """,
            batch,
        ).fetchall()
        # PK별 라벨 수집 (exclude_test_ids 필터)
        per_pk: dict[str, list[int]] = {}
        for r in rows:
            if r["test_id"] in exclude_set:
                continue
            per_pk.setdefault(r["primary_key"], []).append(int(r["label"]))
        for pk, labels in per_pk.items():
            if labels:
                result[pk] = float(sum(labels) / len(labels))
    return result


def get_all_exam_records(domain: str, batch_size: int = DB_STREAM_BATCH) -> list[dict]:
    """특정 도메인의 모든 검사 레코드를 스트리밍으로 반환한다."""
    conn = get_db()
    cursor = conn.execute(
        "SELECT * FROM exam_records WHERE domain = ? ORDER BY test_id",
        (domain,),
    )
    result = []
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        result.extend(dict(r) for r in rows)
    return result


def get_all_sago_records(batch_size: int = DB_STREAM_BATCH) -> list[dict]:
    """전체 사고/위반 레코드를 스트리밍으로 반환한다."""
    conn = get_db()
    cursor = conn.execute(
        "SELECT * FROM sago_records ORDER BY primary_key, acc_date"
    )
    result = []
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        result.extend(dict(r) for r in rows)
    return result


def get_data_summary() -> dict:
    """exam_records/sago_records의 도메인별 건수를 반환한다."""
    conn = get_db()
    a_exam = conn.execute(
        "SELECT COUNT(*) as cnt FROM exam_records WHERE domain = 'A'"
    ).fetchone()["cnt"]
    b_exam = conn.execute(
        "SELECT COUNT(*) as cnt FROM exam_records WHERE domain = 'B'"
    ).fetchone()["cnt"]
    sago = conn.execute(
        "SELECT COUNT(*) as cnt FROM sago_records"
    ).fetchone()["cnt"]
    return {"a_exam": a_exam, "b_exam": b_exam, "sago": sago}


def cleanup_orphan_version_dirs() -> list[str]:
    """DB 레코드에 대응하지 않는 디스크 상의 버전 디렉터리를 제거한다.

    ``artifacts/versions/v_*``를 스캔하여 ``model_versions`` 테이블에
    ``artifact_path``가 없는 디렉터리를 삭제한다.
    삭제된 디렉터리 이름 목록을 반환.
    """
    from src.core.constants import VERSIONS_DIR

    if not os.path.isdir(VERSIONS_DIR):
        return []

    # DB에 등록된 artifact_path 수집
    conn = get_db()
    rows = conn.execute("SELECT artifact_path FROM model_versions").fetchall()
    registered = {row["artifact_path"] for row in rows}

    deleted = []
    for entry in os.listdir(VERSIONS_DIR):
        if not entry.startswith("v_"):
            continue
        dir_path = os.path.join(VERSIONS_DIR, entry)
        if not os.path.isdir(dir_path):
            continue
        rel_path = os.path.relpath(dir_path, PROJECT_ROOT)
        if rel_path not in registered:
            shutil.rmtree(dir_path)
            deleted.append(entry)
            logger.info(f"Cleaned up orphan version directory: {entry}")

    # 고아 model_artifacts 정리: model_versions에 없는 run_id의 아티팩트 삭제
    conn.execute("""
        DELETE FROM model_artifacts
        WHERE run_id NOT IN (
            SELECT DISTINCT run_id FROM model_versions WHERE run_id IS NOT NULL
        )
    """)
    conn.commit()

    return deleted
