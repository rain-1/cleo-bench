"""Project constants."""

from pathlib import Path

DEFAULT_ACCOUNT_ID = 3364210
DEFAULT_SITE = "math.stackexchange"
DEFAULT_SITE_URL = "https://math.stackexchange.com"
DEFAULT_SITE_HOST = "math.stackexchange.com"
DEFAULT_SNAPSHOT_DATE = "2026-02-07"
DEFAULT_DPS = 80
DEFAULT_TOLERANCE = 1e-6

ROOT = Path.cwd()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INSPECT = ROOT / "data" / "inspect"
DATA_MANUAL = ROOT / "data" / "manual_queue"
REPORTS = ROOT / "reports"

STATUS_OK = "ok"
STATUS_NEEDS_REVIEW = "needs_manual_review"
STATUS_NON_INTEGRAL = "non_integral"
STATUS_FETCH_ERROR = "fetch_error"

ALL_STATUSES = {
    STATUS_OK,
    STATUS_NEEDS_REVIEW,
    STATUS_NON_INTEGRAL,
    STATUS_FETCH_ERROR,
}
