import math
import datetime as dt
from typing import Final

DURATION_SECONDS: Final[dict[str, int]] = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2592000,
    "year": 31536000,
}


def align_timestamp(timestamp: dt.datetime, duration: str) -> dt.datetime:
    secs = DURATION_SECONDS.get(duration, 60)
    if secs <= 0:
        secs = 60
    epoch = math.floor(timestamp.replace(tzinfo=dt.timezone.utc).timestamp())
    aligned = math.floor(epoch / secs) * secs
    aligned_dt = dt.datetime.fromtimestamp(aligned, tz=dt.timezone.utc)
    return aligned_dt.replace(tzinfo=None)
