import json
from typing import Any, Optional

import redis

from services import settings

_redis_client: Optional[redis.Redis] = None


def _get_client() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not settings.REDIS_URL:
        return None
    _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def cache_get_json(key: str) -> Optional[Any]:
    client = _get_client()
    if not client:
        return None
    value = client.get(key)
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def cache_set_json(key: str, payload: Any, ttl: int = settings.CACHE_TTL_SECONDS) -> None:
    client = _get_client()
    if not client:
        return
    try:
        client.setex(key, ttl, json.dumps(payload))
    except TypeError:
        client.setex(key, ttl, json.dumps(str(payload)))


def cache_delete(key: str) -> None:
    client = _get_client()
    if not client:
        return
    client.delete(key)


def cache_delete_prefix(prefix: str) -> None:
    client = _get_client()
    if not client:
        return
    cursor = 0
    pattern = f"{prefix}*"
    while True:
        cursor, keys = client.scan(cursor=cursor, match=pattern, count=100)
        if keys:
            client.delete(*keys)
        if cursor == 0:
            break
