import json

from flask import Blueprint, request
from psycopg.rows import dict_row

from extensions import limiter
from services import settings
from services.auth_service import require_client_auth
from services.billing_service import (
    extract_first,
    find_client_id_from_payload,
    kiwify_event_to_status,
    kiwify_get_sale,
    stripe_price_id,
    upsert_subscription,
)
from services.db import db
from services.utils import get_client_id_from_request, get_header, json_err, json_ok, rate_limit_client_id

billing_bp = Blueprint("billing", __name__)


@billing_bp.get("/billing_status")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def billing_status():
    client_id = get_client_id_from_request()
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, client_row, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM subscriptions WHERE client_id=%s", (client_id,))
                sub = cur.fetchone()
        enabled = bool(settings.STRIPE_SECRET_KEY and settings.STRIPE_PRICE_IDS_JSON)
        return json_ok(
            {
                "client_id": client_id,
                "checkout_enabled": enabled,
                "subscription": sub,
                "client": {
                    "plan": client_row.get("plan"),
                    "status": client_row.get("status"),
                    "usage_month": client_row.get("usage_month"),
                    "leads_used_month": int(client_row.get("leads_used_month") or 0),
                },
            }
        )
    finally:
        conn.close()


@billing_bp.post("/billing/checkout")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def billing_checkout():
    data = request.get_json(silent=True) or {}
    client_id = get_client_id_from_request() or (data.get("client_id") or "").strip()
    plan = (data.get("plan") or "").strip().lower()
    success_url = (data.get("success_url") or "").strip()
    cancel_url = (data.get("cancel_url") or "").strip()

    if not client_id:
        return json_err("client_id obrigatório", 400)
    if plan not in settings.PLAN_CATALOG or plan in ("trial", "demo"):
        return json_err("plan inválido para checkout", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    if not (settings.STRIPE_SECRET_KEY and settings.STRIPE_PRICE_IDS_JSON):
        return json_err("Checkout ainda não configurado. Use WhatsApp para ativar.", 501, fallback="whatsapp")

    price_id = stripe_price_id(plan)
    if not price_id:
        return json_err("Price ID do Stripe não encontrado para este plan.", 500)

    import requests

    url = "https://api.stripe.com/v1/checkout/sessions"
    headers = {"Authorization": f"Bearer {settings.STRIPE_SECRET_KEY}"}
    payload = {
        "mode": "subscription",
        "line_items[0][price]": price_id,
        "line_items[0][quantity]": "1",
        "success_url": success_url or "https://qualificador-leads-ia.onrender.com/?checkout=success",
        "cancel_url": cancel_url or "https://qualificador-leads-ia.onrender.com/?checkout=cancel",
        "client_reference_id": client_id,
        "metadata[client_id]": client_id,
        "metadata[plan]": plan,
    }
    response = requests.post(url, headers=headers, data=payload, timeout=20)
    if response.status_code >= 400:
        return json_err(
            "Falha ao criar checkout no Stripe.",
            502,
            stripe_status=response.status_code,
            stripe_body=response.text[:500],
        )

    payload = response.json()
    return json_ok({"checkout_url": payload.get("url"), "session_id": payload.get("id"), "provider": "stripe"})


@billing_bp.post("/billing/webhook")
@limiter.limit("100 per minute")
def billing_webhook():
    if not settings.BILLING_WEBHOOK_SECRET:
        return json_err("Webhook não configurado (BILLING_WEBHOOK_SECRET ausente).", 501)

    got = get_header("X-BILLING-SECRET")
    if got != settings.BILLING_WEBHOOK_SECRET:
        return json_err("Unauthorized", 403)

    payload = request.get_json(silent=True) or {}
    provider = (payload.get("provider") or "manual").strip().lower()
    event_type = (payload.get("type") or payload.get("event_type") or "unknown").strip()
    client_id = (payload.get("client_id") or (payload.get("data") or {}).get("client_id") or "").strip()

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO billing_events (provider, event_type, client_id, payload)
                    VALUES (%s,%s,%s,%s::jsonb)
                    """,
                    (provider, event_type, client_id or None, json.dumps(payload)),
                )
    finally:
        conn.close()

    plan = (payload.get("plan") or "").strip().lower()
    status = (payload.get("status") or "").strip().lower()

    if client_id and plan and status:
        try:
            upsert_subscription(client_id, plan=plan, status=status, provider=provider)
        except Exception as exc:
            return json_err("Evento recebido, mas falhou ao aplicar.", 500, detail=repr(exc))

    return json_ok({"received": True})


@billing_bp.post("/kiwify/webhook")
@limiter.limit("100 per minute")
def kiwify_webhook():
    payload = request.get_json(silent=True) or {}

    if settings.KIWIFY_WEBHOOK_TOKEN:
        got = extract_first(payload, ["token", "webhook_token", "secret"])
        if got != settings.KIWIFY_WEBHOOK_TOKEN:
            return json_err("Unauthorized", 403)

    event_type = extract_first(payload, ["event", "event_type", "type", "trigger"]) or "unknown"
    provider = "kiwify"

    client_id = find_client_id_from_payload(payload)

    order_id = extract_first(payload, ["order_id", "orderId", "sale_id", "saleId", "id"])
    sale = None
    if order_id and not client_id:
        sale = kiwify_get_sale(order_id)
        if isinstance(sale, dict):
            client_id = find_client_id_from_payload(sale)

    plan = extract_first(payload, ["plan", "s2"])
    if not plan and isinstance(sale, dict):
        plan = extract_first(sale, ["plan", "s2"])

    status = kiwify_event_to_status(event_type)

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "INSERT INTO billing_events (provider, event_type, client_id, payload) VALUES (%s,%s,%s,%s::jsonb)",
                    (provider, event_type, client_id or None, json.dumps(payload)),
                )
    finally:
        conn.close()

    if client_id and plan:
        upsert_subscription(client_id, plan=plan, status=status, provider=provider)

    return json_ok({"received": True})
