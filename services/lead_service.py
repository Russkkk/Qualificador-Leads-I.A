from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional
from zoneinfo import ZoneInfo

from psycopg.rows import dict_row

from services import settings
from services.db import db, ensure_client_row, get_active_leads_query
from services.utils import iso, safe_float, safe_int

_SP_TZ = ZoneInfo("America/Sao_Paulo")


def sp_today_bounds_utc() -> tuple[datetime, datetime]:
    now_sp = datetime.now(_SP_TZ)
    start_sp = now_sp.replace(hour=0, minute=0, second=0, microsecond=0)
    end_sp = start_sp.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start_sp.astimezone(timezone.utc), end_sp.astimezone(timezone.utc)


def lead_temperature(probabilidade: Optional[float], score: Optional[int]) -> str:
    prob = safe_float(probabilidade, None)
    score_val = safe_int(score, 0) if score is not None else None

    if prob is None and score_val is None:
        return "unknown"
    if (prob is not None and prob >= 0.70) or (score_val is not None and score_val >= 70):
        return "hot"
    if (prob is not None and prob >= 0.35) or (score_val is not None and score_val >= 35):
        return "warm"
    return "cold"


def top_origens(client_id: str, days: int = 30, limit: int = 6):
    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT COALESCE(NULLIF(TRIM(origem), ''), 'desconhecida') AS origem,
                           COUNT(*)::int AS total
                    {active_leads_query}
                      AND client_id=%s
                      AND created_at >= (NOW() - (%s || ' days')::interval)
                    GROUP BY 1
                    ORDER BY total DESC, origem ASC
                    LIMIT %s
                    """,
                    (client_id, int(days), int(limit)),
                )
                return cur.fetchall()
    finally:
        conn.close()


def hot_leads_today(client_id: str, limit: int = 20):
    start_utc, end_utc = sp_today_bounds_utc()
    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT id, nome, telefone, email_lead, origem,
                           probabilidade, score, created_at, virou_cliente
                    {active_leads_query}
                      AND client_id=%s
                      AND created_at >= %s AND created_at <= %s
                      AND (
                            (probabilidade IS NOT NULL AND probabilidade >= 0.70)
                            OR (score IS NOT NULL AND score >= 70)
                          )
                    ORDER BY COALESCE(probabilidade, score/100.0) DESC NULLS LAST,
                             created_at DESC
                    LIMIT %s
                    """,
                    (client_id, start_utc, end_utc, int(limit)),
                )
                rows = cur.fetchall()
                for r in rows:
                    r["created_at"] = iso(r.get("created_at"))
                return rows
    finally:
        conn.close()


def get_threshold(client_id: str) -> float:
    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT threshold FROM thresholds WHERE client_id=%s", (client_id,))
                row = cur.fetchone()
                if row and row.get("threshold") is not None:
                    return float(row["threshold"])
        return settings.DEFAULT_THRESHOLD
    finally:
        conn.close()


def set_threshold(client_id: str, threshold: float):
    conn = db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO thresholds (client_id, threshold, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (client_id)
                    DO UPDATE SET threshold=EXCLUDED.threshold, updated_at=NOW()
                    """,
                    (client_id, float(threshold)),
                )
    finally:
        conn.close()


def fetch_recent_leads(
    client_id: str,
    limit: int = settings.DEFAULT_LIMIT,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT id, client_id, nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco,
                           probabilidade, virou_cliente, created_at
                    {active_leads_query}
                      AND client_id=%s
                    ORDER BY created_at DESC
                    LIMIT %s
                    OFFSET %s
                    """,
                    (client_id, int(limit), int(offset)),
                )
                return [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()


def count_leads(client_id: str) -> int:
    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT COUNT(*)::int AS total
                    {active_leads_query}
                      AND client_id=%s
                    """,
                    (client_id,),
                )
                row = cur.fetchone() or {}
                return int(row.get("total") or 0)
    finally:
        conn.close()


def count_status(rows: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    convertidos = sum(1 for r in rows if r.get("virou_cliente") in (1, 1.0))
    negados = sum(1 for r in rows if r.get("virou_cliente") in (0, 0.0))
    pendentes = len(rows) - convertidos - negados
    return convertidos, negados, pendentes


def get_labeled_rows(client_id: str) -> List[Dict[str, Any]]:
    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT id, tempo_site, paginas_visitadas, clicou_preco, probabilidade, virou_cliente
                    {active_leads_query}
                      AND client_id=%s
                      AND virou_cliente IS NOT NULL
                    ORDER BY created_at DESC
                    """,
                    (client_id,),
                )
                return [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()


def update_probabilities(client_id: str, ids: List[int], probs: List[float]) -> int:
    if not ids:
        return 0
    conn = db()
    try:
        with conn:
            with conn.cursor() as cur:
                for lead_id, p in zip(ids, probs):
                    cur.execute(
                        "UPDATE leads SET probabilidade=%s, updated_at=NOW() WHERE client_id=%s AND id=%s AND deleted_at IS NULL",
                        (float(p), client_id, int(lead_id)),
                    )
        return len(ids)
    finally:
        conn.close()


def check_quota_and_bump(client_id: str, client_row: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    plan = (client_row.get("plan") or "trial").strip().lower()
    meta = settings.PLAN_CATALOG.get(plan, settings.PLAN_CATALOG["trial"])
    used = int(client_row.get("leads_used_month") or 0)
    limit = int(meta.get("lead_limit_month") or 0)

    if limit > 0 and used >= limit:
        return False, "Limite mensal atingido. Faça upgrade para continuar.", {
            "code": "plan_limit",
            "plan": plan,
            "used": used,
            "limit": limit,
            "price_brl_month": meta.get("price_brl_month"),
            "setup_fee_brl": meta.get("setup_fee_brl", 0),
        }

    conn = db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE clients SET leads_used_month = leads_used_month + 1, updated_at=NOW() WHERE client_id=%s",
                    (client_id,),
                )
        return True, "", {}
    finally:
        conn.close()


def prever_rate_limit(client_id: str) -> str:
    if not client_id:
        return "20 per minute"
    try:
        row = ensure_client_row(client_id, plan="trial")
        plan = (row.get("plan") or "trial").strip().lower()
    except Exception:
        plan = "trial"
    if plan in ("trial", "demo"):
        return "20 per minute"
    return "600 per minute"
