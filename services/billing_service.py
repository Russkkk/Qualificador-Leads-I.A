import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from psycopg.rows import dict_row

from services import settings
from services.db import db

_KIWIFY_OAUTH_CACHE = {"token": "", "expires_at": 0}


def stripe_price_id(plan: str) -> Optional[str]:
    if not settings.STRIPE_PRICE_IDS_JSON:
        return None
    try:
        mapping = json.loads(settings.STRIPE_PRICE_IDS_JSON)
        return (mapping.get(plan) or "").strip() or None
    except Exception:
        return None


def upsert_subscription(
    client_id: str,
    plan: str,
    status: str,
    provider: str = "manual",
    period_start: Optional[datetime] = None,
    period_end: Optional[datetime] = None,
    cancel_at_period_end: bool = False,
):
    if plan not in settings.PLAN_CATALOG:
        plan = "trial"
    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO subscriptions (client_id, provider, status, plan, current_period_start, current_period_end, cancel_at_period_end, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
                    ON CONFLICT (client_id) DO UPDATE SET
                      provider=EXCLUDED.provider,
                      status=EXCLUDED.status,
                      plan=EXCLUDED.plan,
                      current_period_start=EXCLUDED.current_period_start,
                      current_period_end=EXCLUDED.current_period_end,
                      cancel_at_period_end=EXCLUDED.cancel_at_period_end,
                      updated_at=NOW()
                    """,
                    (client_id, provider, status, plan, period_start, period_end, cancel_at_period_end),
                )

                if status == "active":
                    cur.execute(
                        "UPDATE clients SET plan=%s, status='active', updated_at=NOW() WHERE client_id=%s",
                        (plan, client_id),
                    )
                elif status in ("past_due", "canceled", "inactive"):
                    cur.execute(
                        "UPDATE clients SET status='inactive', updated_at=NOW() WHERE client_id=%s",
                        (client_id,),
                    )
    finally:
        conn.close()


def kiwify_get_token() -> Optional[str]:
    if not (settings.KIWIFY_API_KEY and settings.KIWIFY_CLIENT_SECRET and settings.KIWIFY_ACCOUNT_ID):
        return None
    now = int(time.time())
    if _KIWIFY_OAUTH_CACHE.get("token") and now < int(_KIWIFY_OAUTH_CACHE.get("expires_at") or 0) - 60:
        return _KIWIFY_OAUTH_CACHE["token"]
    import requests

    url = "https://public-api.kiwify.com/oauth/token"
    response = requests.post(
        url,
        json={"api_key": settings.KIWIFY_API_KEY, "client_secret": settings.KIWIFY_CLIENT_SECRET},
        timeout=20,
    )
    if response.status_code >= 400:
        return None
    try:
        data = response.json()
    except Exception:
        data = {}
    token = (data.get("access_token") or "").strip()
    expires_in = int(data.get("expires_in") or 96 * 3600)
    if token:
        _KIWIFY_OAUTH_CACHE["token"] = token
        _KIWIFY_OAUTH_CACHE["expires_at"] = now + expires_in
        return token
    return None


def kiwify_get_sale(order_id: str) -> Optional[Dict[str, Any]]:
    tok = kiwify_get_token()
    if not tok:
        return None
    import requests

    url = f"https://public-api.kiwify.com/v1/sales/{order_id}"
    headers = {"Authorization": f"Bearer {tok}", "x-kiwify-account-id": settings.KIWIFY_ACCOUNT_ID}
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code >= 400:
        return None
    try:
        return response.json()
    except Exception:
        return None


def extract_first(payload: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def find_client_id_from_payload(payload: Dict[str, Any]) -> str:
    direct = extract_first(payload, ["client_id", "clientId", "workspace_id", "workspaceId", "s1"])
    if direct:
        return direct
    for key in ("tracking", "utm", "data", "sale", "order", "customer"):
        value = payload.get(key)
        if isinstance(value, dict):
            got = extract_first(value, ["client_id", "clientId", "workspace_id", "workspaceId", "s1"])
            if got:
                return got
    return ""


def kiwify_event_to_status(event_type: str) -> str:
    event = (event_type or "").strip().lower()
    if event in ("compra_aprovada", "subscription_renewed"):
        return "active"
    if event in ("subscription_late",):
        return "past_due"
    if event in ("compra_reembolsada", "chargeback", "subscription_canceled"):
        return "canceled"
    if event in ("compra_recusada",):
        return "inactive"
    return "inactive"
