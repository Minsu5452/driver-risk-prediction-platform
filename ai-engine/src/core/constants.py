import os
import platform
from typing import Dict

# ── .env 로드 (모든 entry point에서 동작하도록 여기서 처리) ──
_env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".env",
)
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

SEED = 2025
CLIP_EPS = 1e-5
EPS = 1e-6
STRICT_CONFIG = True

# ═══════════════════════════════════════════════════════════════
# 시스템 사양 자동 감지 (외부 라이브러리 없음)
# ═══════════════════════════════════════════════════════════════


def _detect_ram_gb() -> float:
    """총 물리 RAM(GB) 감지. Windows/Mac/Linux 모두 지원."""
    try:
        system = platform.system()
        if system == "Darwin":
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
            )
            return int(r.stdout.strip()) / (1024 ** 3)
        elif system == "Linux":
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
        elif system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
    except Exception as e:
        import sys
        print(f"[WARNING] RAM 감지 실패 (기본값 8GB 사용): {e}", file=sys.stderr)
    return 8.0  # 감지 실패 시 보수적 기본값


def _int_env(key: str, default: int) -> int:
    """환경 변수에서 int 읽기. 없으면 default 반환."""
    return int(os.environ.get(key, str(default)))


# ── 시스템 정보 ──
_CPU_COUNT = os.cpu_count() or 4
_RAM_GB = _detect_ram_gb()

# ═══════════════════════════════════════════════════════════════
# 성능 파라미터 — 자동 계산 (환경 변수 RISK_* 로 오버라이드 가능)
# 기본값: 벤치마크 결과 기반, 저사양(8GB/4코어)에서도 안전
# ═══════════════════════════════════════════════════════════════

# ── 병렬 처리 ──
# 벤치마크: XGB/LGB 모두 4코어 최적 (6+ 효과 없거나 역효과)
_auto_n_jobs = min(max(2, _CPU_COUNT // 2), 4)
N_JOBS = _int_env("RISK_N_JOBS", _auto_n_jobs)

TRAIN_DOMAIN_WORKERS = _int_env("RISK_TRAIN_DOMAIN_WORKERS", 2)
EXPLAIN_BATCH_WORKERS = _int_env("RISK_EXPLAIN_BATCH_WORKERS", min(max(2, N_JOBS), 6))
SEQ_PARSE_JOBS = _int_env("RISK_SEQ_PARSE_JOBS", 1)

# ── SQLite ──
DB_TIMEOUT = _int_env("RISK_DB_TIMEOUT", 120)
_auto_cache_mb = max(64, min(int(_RAM_GB * 10), 256))
DB_CACHE_SIZE = _int_env("RISK_DB_CACHE_SIZE", -_auto_cache_mb * 1000)  # 음수=KB
_auto_mmap_mb = max(256, min(int(_RAM_GB * 40), 1024))
DB_MMAP_SIZE = _int_env("RISK_DB_MMAP_SIZE", _auto_mmap_mb * 1024 * 1024)

# ── 배치 크기 (벤치마크: 1K~20K 차이 없음) ──
DB_UPSERT_BATCH = _int_env("RISK_DB_UPSERT_BATCH", 5000)
DB_QUERY_BATCH = _int_env("RISK_DB_QUERY_BATCH", 500)
DB_STREAM_BATCH = _int_env("RISK_DB_STREAM_BATCH", 10000)

# ── 학습 ──
ENSEMBLE_DIRICHLET_SAMPLES = _int_env("RISK_ENSEMBLE_SAMPLES", 192)
EARLY_STOPPING_ROUNDS = _int_env("RISK_EARLY_STOPPING_ROUNDS", 200)
SEQ_DIFF_CHUNK = _int_env("RISK_SEQ_DIFF_CHUNK", 50)

# ── 추론 ──
MAX_SHAP_SAMPLES = _int_env("RISK_MAX_SHAP_SAMPLES", 5000 if _RAM_GB >= 8 else 3000)
EXPLAIN_BATCH_LIMIT = _int_env("RISK_EXPLAIN_BATCH_LIMIT", 100)
EXPLAIN_TIMEOUT = _int_env("RISK_EXPLAIN_TIMEOUT", 60)

# ── 파일 I/O ──
UPLOAD_CHUNK_BYTES = _int_env("RISK_UPLOAD_CHUNK_BYTES", 1024 * 1024)
MAX_UPLOAD_FILE_SIZE = _int_env("RISK_MAX_FILE_SIZE", 10 * 1024 * 1024 * 1024)

# ═══════════════════════════════════════════════════════════════
# 프로젝트 경로 (변경 없음)
# ═══════════════════════════════════════════════════════════════

_CURRENT_FILE = os.path.abspath(__file__)
_SRC_CORE_DIR = os.path.dirname(_CURRENT_FILE)
_SRC_DIR = os.path.dirname(_SRC_CORE_DIR)
PROJECT_ROOT = os.path.dirname(_SRC_DIR)

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
VERSIONS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "versions")
BASE_DIR = os.path.join(PROJECT_ROOT, "data")


SEQ_LENGTHS = {
    "A1": 18,
    "A2": 18,
    "A3": 32,
    "A4": 80,
    "A5": 36,
    "B1": 16,
    "B2": 16,
    "B3": 15,
    "B4": 60,
    "B5": 20,
    "B6": 15,
    "B7": 15,
    "B8": 12,
    "B9": 50,
    "B10": 80,
}

# ── 컬럼 그룹 상수 (하드코딩 제거용) ──
# A 도메인: 시퀀스 컬럼 (str_to_num_array / seq_matrix로 파싱)
A_SEQ_COLS = [
    "A1-1", "A1-2", "A1-3", "A1-4",
    "A2-1", "A2-2", "A2-3", "A2-4",
    "A3-1", "A3-3", "A3-5", "A3-6", "A3-7",
    "A4-1", "A4-2", "A4-3", "A4-4", "A4-5",
    "A5-1", "A5-2", "A5-3",
]
# A 도메인: 정수 카운트 컬럼 (시퀀스가 아닌 단일 숫자값)
A_INT_COLS = ["A6-1", "A7-1", "A8-1", "A8-2", "A9-1", "A9-2", "A9-3", "A9-4", "A9-5"]
# A 도메인: 추론/학습 시 원본 컬럼으로 전달 (숫자 변환 필요)
A_RAW_COLS = ["Test_id", "Test"] + A_INT_COLS

# B 도메인: 시퀀스 컬럼
B_SEQ_COLS = [
    "B1-1", "B1-2", "B1-3",
    "B2-1", "B2-2", "B2-3",
    "B3-1", "B3-2",
    "B4-1", "B4-2",
    "B5-1", "B5-2",
    "B6", "B7", "B8",
]
# B 도메인: 점수 컬럼 (숫자 변환 필요)
B_SCORE_COLS = ["B9-1", "B9-2", "B9-3", "B9-4", "B9-5", "B10-1", "B10-2", "B10-3", "B10-4", "B10-5", "B10-6"]
# B 도메인: 추론/학습 시 원본 컬럼으로 전달
B_RAW_COLS = ["Test_id", "Test"] + B_SCORE_COLS

DOMAIN_CFG: Dict[str, Dict] = {
    "A": {
        "M_SMOOTH": 20.0,
        "DECAY_HALF_LIFE": 15.0,
        "MODELS": {},
    },
    "B": {
        "M_SMOOTH": 20.0,
        "DECAY_HALF_LIFE": 15.0,
        "MODELS": {},
    },
}
