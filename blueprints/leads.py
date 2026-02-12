import json
import random
import string
from datetime import timedelta
from typing import Any, Dict

import psycopg
from flask import Blueprint, Response, request
from psycopg.rows import dict_row

from extensions import limiter
from services import settings
from services.auth_service import gen_api_key, require_client_auth
from services.db import db, ensure_client_row, get_active_leads_query
from services.demo_service import bump_demo_counter, demo_rate_limited, require_demo_key
from services.cache import cache_delete, cache_delete_prefix, cache_get_json, cache_set_json
from services.lead_service import (
    check_quota_and_bump,
    count_leads,
    count_status,
    fetch_recent_leads,
    get_threshold,
    hot_leads_today,
    lead_temperature,
    prever_rate_limit,
    top_origens,
)
from services.utils import (
    client_ip,
    get_client_id_from_request,
    json_err,
    json_ok,
    iso,
    month_key,
    now_utc,
    rate_limit_client_id,
    safe_float,
    safe_int,
)
from services.validation import sanitize_name, sanitize_origin, sanitize_phone

leads_bp = Blueprint("leads", __name__)


def _prever_limit() -> str:
    return prever_rate_limit(get_client_id_from_request())


@leads_bp.get("/prever_example")
def prever_example():
    """Return an official example payload for the /prever endpoint.

    Supports:
      - /prever_example?format=lead  -> {"lead": {...}}
      - /prever_example?format=root  -> {...}

    Safe, unauthenticated, placeholders only.
    """

    fmt = (request.args.get("format") or "lead").strip().lower()
    if fmt not in ["lead", "root"]:
        return json_err("format inválido", 400, allowed=["lead", "root"])

    lead_obj = {
        "nome": "João da Silva",
        "email": "joao@empresa.com",
        "telefone": "11999999999",
        "origem": "site",
        "tempo_site": 185,
        "paginas_visitadas": 4,
        "clicou_preco": 1,
    }

    example = {"lead": lead_obj} if fmt == "lead" else lead_obj

    notes = {
        "headers_required": {
            "X-API-KEY": "SUA_API_KEY",
            "X-CLIENT-ID": "SEU_WORKSPACE",
        },
        "accepted_fields": [
            "nome" if fmt == "root" else "lead.nome",
            "email" if fmt == "root" else "lead.email",
            "telefone" if fmt == "root" else "lead.telefone",
            "origem" if fmt == "root" else "lead.origem",
            "tempo_site" if fmt == "root" else "lead.tempo_site",
            "paginas_visitadas" if fmt == "root" else "lead.paginas_visitadas",
            "clicou_preco" if fmt == "root" else "lead.clicou_preco",
        ],
        "format": fmt,
        "allowed_formats": ["lead", "root"],
        "tip": "Você pode alternar o exemplo via /prever_example?format=lead|root.",
    }

    return json_ok({"example": example, "notes": notes})



@leads_bp.post("/criar_cliente")
def criar_cliente():
    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    plan = (data.get("plan") or "trial").strip().lower()

    if not client_id:
        return json_err("client_id obrigatório", 400)
    if plan not in settings.PLAN_CATALOG:
        plan = "trial"

    row = ensure_client_row(client_id, plan=plan)

    api_key = (row.get("api_key") or "").strip()
    if not api_key:
        api_key = gen_api_key(client_id)
        conn = db()
        try:
            with conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "UPDATE clients SET api_key=%s, plan=%s, updated_at=NOW() WHERE client_id=%s",
                        (api_key, plan, client_id),
                    )
                    cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                    row = cur.fetchone() or row
        finally:
            conn.close()

    meta = settings.PLAN_CATALOG.get((row.get("plan") or plan).lower(), settings.PLAN_CATALOG["trial"])
    return json_ok(
        {
            "client_id": client_id,
            "api_key": api_key,
            "plan": (row.get("plan") or plan),
            "price_brl_month": meta["price_brl_month"],
            "setup_fee_brl": meta.get("setup_fee_brl", 0),
            "lead_limit_month": meta["lead_limit_month"],
        }
    )


@leads_bp.get("/client_meta")
def client_meta():
    client_id = get_client_id_from_request()
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, row, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    plan = (row.get("plan") or "trial").strip().lower()
    cat = settings.PLAN_CATALOG.get(plan, settings.PLAN_CATALOG["trial"])
    return json_ok(
        {
            "client_id": client_id,
            "plan": plan,
            "status": row.get("status") or "active",
            "price_brl_month": cat["price_brl_month"],
            "setup_fee_brl": cat.get("setup_fee_brl", 0),
            "lead_limit_month": cat["lead_limit_month"],
            "leads_used_this_month": int(row.get("leads_used_month") or 0),
            "usage_month": row.get("usage_month") or month_key(),
            "ts": iso(now_utc()),
        }
    )


@leads_bp.post("/set_plan")
def set_plan():
    ok, err = require_demo_key()
    if not ok:
        return json_err("Unauthorized (DEMO_KEY)", 403, reason=err)

    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    plan = (data.get("plan") or "").strip().lower()
    status = (data.get("status") or "").strip().lower()

    if not client_id:
        return json_err("client_id obrigatório", 400)
    if plan and plan not in settings.PLAN_CATALOG:
        return json_err("plan inválido", 400, allowed=list(settings.PLAN_CATALOG.keys()))
    if status and status not in ["active", "inactive"]:
        return json_err("status inválido", 400, allowed=["active", "inactive"])

    ensure_client_row(client_id, plan=plan or "trial")

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                sets = []
                vals = []
                if plan:
                    sets.append("plan=%s")
                    vals.append(plan)
                if status:
                    sets.append("status=%s")
                    vals.append(status)
                sets.append("updated_at=NOW()")
                q = f"UPDATE clients SET {', '.join(sets)} WHERE client_id=%s"
                vals.append(client_id)
                cur.execute(q, tuple(vals))
                cur.execute("SELECT * FROM clients WHERE client_id=%s", (client_id,))
                row = cur.fetchone() or {}
        return json_ok({"client_id": client_id, "plan": row.get("plan"), "status": row.get("status")})
    finally:
        conn.close()


@leads_bp.post("/prever")
@limiter.limit(_prever_limit, key_func=rate_limit_client_id)
def prever():
    raw_payload = request.get_data(cache=True, as_text=False) or b""
    max_bytes = settings.MAX_PREVER_PAYLOAD_BYTES
    if max_bytes and len(raw_payload) > max_bytes:
        return json_err(
            "Payload muito grande para /prever.",
            413,
            code="payload_too_large",
            limit_bytes=max_bytes,
            size_bytes=len(raw_payload),
        )

    data = request.get_json(silent=True) or {}
    client_id = get_client_id_from_request()
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, client_row, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    if (client_row.get("status") or "active") != "active":
        return json_err("Workspace inativo. Fale com o suporte para reativar.", 403, code="inactive")

    plan = (client_row.get("plan") or "trial").lower()
    cat = settings.PLAN_CATALOG.get(plan, settings.PLAN_CATALOG["trial"])
    used = int(client_row.get("leads_used_month") or 0)
    limit = int(cat.get("lead_limit_month") or 0)
    if limit > 0 and used >= limit:
        return json_err(
            "Limite mensal atingido. Faça upgrade para continuar.",
            402,
            code="plan_limit",
            plan=plan,
            used=used,
            limit=limit,
            price_brl_month=cat.get("price_brl_month"),
            setup_fee_brl=cat.get("setup_fee_brl", 0),
        )

    lead = data.get("lead") or {}
    nome = sanitize_name(data.get("nome") or lead.get("nome") or "")
    email = (data.get("email_lead") or data.get("email") or lead.get("email_lead") or lead.get("email") or "").strip()
    telefone = sanitize_phone(data.get("telefone") or lead.get("telefone") or "")

    origem = sanitize_origin(data.get("origem") or lead.get("origem") or lead.get("source") or "")

    tempo_site = safe_int(data.get("tempo_site") if "tempo_site" in data else lead.get("tempo_site"), 0)
    paginas_visitadas = safe_int(data.get("paginas_visitadas") if "paginas_visitadas" in data else lead.get("paginas_visitadas"), 0)
    clicou_preco = safe_int(data.get("clicou_preco") if "clicou_preco" in data else lead.get("clicou_preco"), 0)

    base = 0.10
    base += min(tempo_site / 400, 0.25)
    base += min(paginas_visitadas / 10, 0.25)
    base += 0.20 if clicou_preco else 0.0
    if telefone and len(telefone) >= 10:
        base += 0.06
    if nome and len(nome) >= 4:
        base += 0.04

    prob = max(0.02, min(0.98, base))
    score = int(round(prob * 100))
    label = 1 if prob >= 0.70 else (0 if prob < 0.35 else None)

    payload = lead if isinstance(lead, dict) else {}
    payload.setdefault("nome", nome)
    payload.setdefault("email", email)
    payload.setdefault("email_lead", email)
    payload.setdefault("telefone", telefone)
    payload.setdefault("origem", origem or payload.get("origem", ""))
    payload.setdefault("tempo_site", tempo_site)
    payload.setdefault("paginas_visitadas", paginas_visitadas)
    payload.setdefault("clicou_preco", clicou_preco)

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT plan, leads_used_month FROM clients WHERE client_id=%s FOR UPDATE",
                    (client_id,),
                )
                locked_client = cur.fetchone() or {}
                plan_locked = (locked_client.get("plan") or client_row.get("plan") or "trial").lower()
                cat_locked = settings.PLAN_CATALOG.get(plan_locked, settings.PLAN_CATALOG["trial"])
                used_locked = int(locked_client.get("leads_used_month") or 0)
                limit_locked = int(cat_locked.get("lead_limit_month") or 0)
                if limit_locked > 0 and used_locked >= limit_locked:
                    return json_err(
                        "Limite mensal atingido. Faça upgrade para continuar.",
                        402,
                        code="plan_limit",
                        plan=plan_locked,
                        used=used_locked,
                        limit=limit_locked,
                        price_brl_month=cat_locked.get("price_brl_month"),
                        setup_fee_brl=cat_locked.get("setup_fee_brl", 0),
                    )

                cur.execute(
                    """
                    INSERT INTO leads
                      (client_id, nome, email_lead, telefone, origem, tempo_site, paginas_visitadas, clicou_preco,
                       payload, probabilidade, score, label, virou_cliente, created_at, updated_at)
                    VALUES
                      (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,NULL,NOW(),NOW())
                    RETURNING id, created_at
                    """,
                    (
                        client_id,
                        nome,
                        email,
                        telefone,
                        origem,
                        tempo_site,
                        paginas_visitadas,
                        clicou_preco,
                        json.dumps(payload),
                        float(prob),
                        int(score),
                        label,
                    ),
                )
                row = cur.fetchone() or {}
                cur.execute(
                    "UPDATE clients SET leads_used_month = leads_used_month + 1, updated_at=NOW() WHERE client_id=%s",
                    (client_id,),
                )

        cache_delete(f"acao_do_dia:{client_id}")
        cache_delete_prefix(f"insights:{client_id}:")

        return json_ok(
            {
                "client_id": client_id,
                "lead_id": int(row.get("id") or 0),
                "probabilidade": float(prob),
                "score": int(score),
                "label": label,
                "plan": plan,
                "created_at": iso(row.get("created_at")),
            }
        )
    except (psycopg.errors.UndefinedColumn, psycopg.errors.NotNullViolation):
        return json_err(
            "Esquema do banco desatualizado. Rode as migrations do Alembic.",
            500,
            code="schema_outdated",
        )
    finally:
        conn.close()


@leads_bp.get("/dashboard_data")
@limiter.limit("30 per minute", key_func=rate_limit_client_id)
def dashboard_data():
    client_id = get_client_id_from_request()
    page = safe_int(request.args.get("page"), 1)
    per_page = safe_int(request.args.get("limit"), settings.DEFAULT_LIMIT)
    per_page = max(10, min(per_page, 200))
    page = max(1, page)
    offset = (page - 1) * per_page

    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    total_leads = count_leads(client_id)
    rows = fetch_recent_leads(client_id, limit=per_page, offset=offset)
    convertidos, negados, pendentes = count_status(rows)
    top_origens_rows = top_origens(client_id, days=30, limit=6)
    hot_leads = hot_leads_today(client_id, limit=20)

    def norm(item: Dict[str, Any]) -> Dict[str, Any]:
        rr = dict(item)
        rr["created_at"] = iso(rr.get("created_at"))
        return rr

    return json_ok(
        {
            "client_id": client_id,
            "convertidos": convertidos,
            "negados": negados,
            "pendentes": pendentes,
            "page": page,
            "per_page": per_page,
            "total_leads": total_leads,
            "top_origens_30d": top_origens_rows,
            "hot_leads_today": hot_leads,
            "hot_leads_today_tz": "America/Sao_Paulo",
            "dados": [norm(r) for r in rows],
            "total_recentes_considerados": len(rows),
        }
    )


@leads_bp.post("/confirmar_venda")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def confirmar_venda():
    data = request.get_json(silent=True) or {}
    client_id = get_client_id_from_request()
    lead_id = safe_int(data.get("lead_id"), 0)
    if not client_id or not lead_id:
        return json_err("client_id e lead_id obrigatórios", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    conn = db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE leads SET virou_cliente=1, updated_at=NOW() WHERE client_id=%s AND id=%s",
                    (client_id, lead_id),
                )
        cache_delete(f"acao_do_dia:{client_id}")
        cache_delete_prefix(f"insights:{client_id}:")
        return json_ok({"client_id": client_id, "lead_id": lead_id, "virou_cliente": 1})
    finally:
        conn.close()


@leads_bp.post("/negar_venda")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def negar_venda():
    data = request.get_json(silent=True) or {}
    client_id = get_client_id_from_request()
    lead_id = safe_int(data.get("lead_id"), 0)
    if not client_id or not lead_id:
        return json_err("client_id e lead_id obrigatórios", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    conn = db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE leads SET virou_cliente=0, updated_at=NOW() WHERE client_id=%s AND id=%s",
                    (client_id, lead_id),
                )
        cache_delete(f"acao_do_dia:{client_id}")
        cache_delete_prefix(f"insights:{client_id}:")
        return json_ok({"client_id": client_id, "lead_id": lead_id, "virou_cliente": 0})
    finally:
        conn.close()


@leads_bp.get("/metrics")
@limiter.limit("100 per minute")
def metrics():
    if not settings.DATABASE_URL:
        return json_ok({"db": False, "reason": "DATABASE_URL ausente", "ts": iso(now_utc())})

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(f"SELECT COUNT(*) AS total {active_leads_query};")
                total = int(cur.fetchone()["total"])
                cur.execute(
                    f"SELECT COUNT(*) AS labeled {active_leads_query} AND virou_cliente IS NOT NULL;"
                )
                labeled = int(cur.fetchone()["labeled"])
                cur.execute(
                    f"SELECT COUNT(*) AS pending {active_leads_query} AND virou_cliente IS NULL;"
                )
                pending = int(cur.fetchone()["pending"])
        return json_ok(
            {
                "db": True,
                "total_leads": total,
                "labeled": labeled,
                "pending": pending,
                "ts": iso(now_utc()),
            }
        )
    finally:
        conn.close()


@leads_bp.get("/insights")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def insights():
    client_id = get_client_id_from_request()
    days = safe_int(request.args.get("days"), 14)
    days = max(7, min(days, 90))
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    cache_key = f"insights:{client_id}:{days}"
    cached = cache_get_json(cache_key)
    if cached:
        return json_ok(cached)

    threshold = get_threshold(client_id)
    since = now_utc() - timedelta(days=days)

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT probabilidade, virou_cliente, created_at
                    {active_leads_query}
                      AND client_id=%s
                      AND created_at >= %s
                    ORDER BY created_at ASC
                    """,
                    (client_id, since),
                )
                rows = [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()

    bands_def = [
        ("0-0.2", 0.0, 0.2),
        ("0.2-0.4", 0.2, 0.4),
        ("0.4-0.6", 0.4, 0.6),
        ("0.6-0.8", 0.6, 0.8),
        ("0.8-1.0", 0.8, 1.01),
    ]

    bands = []
    for name, lo, hi in bands_def:
        subset = [r for r in rows if r.get("probabilidade") is not None and lo <= float(r["probabilidade"]) < hi]
        labeled = [r for r in subset if r.get("virou_cliente") is not None]
        conv = sum(1 for r in labeled if float(r["virou_cliente"]) == 1.0)
        total = len(labeled)
        rate = (conv / total) if total else 0.0
        bands.append({"band": name, "labeled": total, "converted": conv, "conversion_rate": round(float(rate), 4)})

    by_day: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        dt = r.get("created_at")
        if not dt:
            continue
        day = dt.date().isoformat()
        by_day.setdefault(day, {"day": day, "total": 0, "converted": 0, "denied": 0, "pending": 0})
        by_day[day]["total"] += 1
        vc = r.get("virou_cliente")
        if vc is None:
            by_day[day]["pending"] += 1
        elif float(vc) == 1.0:
            by_day[day]["converted"] += 1
        else:
            by_day[day]["denied"] += 1

    series = [by_day[k] for k in sorted(by_day.keys())]

    labeled_all = [r for r in rows if r.get("virou_cliente") is not None]
    conv_all = sum(1 for r in labeled_all if float(r["virou_cliente"]) == 1.0)
    den_all = sum(1 for r in labeled_all if float(r["virou_cliente"]) == 0.0)
    overall_rate = (conv_all / len(labeled_all)) if labeled_all else 0.0

    payload = {
        "client_id": client_id,
        "threshold": float(threshold),
        "overall": {
            "window_total": len(rows),
            "labeled": len(labeled_all),
            "converted": conv_all,
            "denied": den_all,
            "conversion_rate": round(float(overall_rate), 4),
        },
        "bands": bands,
        "series": series,
        "window_days": days,
    }
    cache_set_json(cache_key, payload)
    return json_ok(payload)


@leads_bp.get("/leads_export.csv")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def leads_export():
    client_id = get_client_id_from_request()
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco,
                           probabilidade, score, created_at, virou_cliente
                    {active_leads_query}
                      AND client_id=%s
                    ORDER BY created_at DESC
                    """,
                    (client_id,),
                )
                rows = cur.fetchall() or []
    finally:
        conn.close()

    header = "nome,email,telefone,tempo_site,paginas_visitadas,clicou_preco,probabilidade,score,created_at,virou_cliente\n"
    lines = [header]
    for r in rows:
        created_at = iso(r.get("created_at")) or ""
        lines.append(
            f"{r.get('nome','')},{r.get('email_lead','')},{r.get('telefone','')},{r.get('tempo_site','')},"
            f"{r.get('paginas_visitadas','')},{r.get('clicou_preco','')},{r.get('probabilidade','')},"
            f"{r.get('score','')},{created_at},{r.get('virou_cliente','')}\n"
        )
    return Response("".join(lines), mimetype="text/csv")


@leads_bp.post("/demo_public")
@limiter.limit("100 per minute")
def demo_public():
    data = request.get_json(silent=True) or {}
    key = client_ip()
    if demo_rate_limited(key):
        return json_err("Limite de demos atingido para este IP neste mês.", 429, code="rate_limit")

    bump_demo_counter(key)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    client_id = f"demo_{suffix}"
    ensure_client_row(client_id, plan="demo")

    conn = db()
    try:
        with conn:
            with conn.cursor() as cur:
                for _ in range(6):
                    tempo_site = random.randint(1, 10)
                    paginas = random.randint(1, 8)
                    clicou_preco = random.randint(0, 1)
                    base = 0.10
                    base += min(tempo_site / 400, 0.25)
                    base += min(paginas / 10, 0.25)
                    base += 0.20 if clicou_preco else 0.0
                    prob = max(0.02, min(0.98, base))
                    score = int(round(prob * 100))
                    label = 1 if prob >= 0.70 else (0 if prob < 0.35 else None)
                    cur.execute(
                        """
                        INSERT INTO leads (client_id, nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco,
                                           payload, probabilidade, score, label, created_at, updated_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,NOW(),NOW())
                        """,
                        (
                            client_id,
                            "Demo Lead",
                            "demo@leadrank.local",
                            "(11) 90000-0000",
                            tempo_site,
                            paginas,
                            clicou_preco,
                            json.dumps(data),
                            float(prob),
                            int(score),
                            label,
                        ),
                    )
        return json_ok({"client_id": client_id, "inserted": 6})
    finally:
        conn.close()


@leads_bp.post("/seed_demo")
@limiter.limit("100 per minute")
def seed_demo():
    ok, err = require_demo_key()
    if not ok:
        return json_err(err or "Unauthorized", 403)

    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not client_id:
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        client_id = f"demo_{suffix}"

    ensure_client_row(client_id, plan="demo")
    conn = db()
    try:
        with conn:
            with conn.cursor() as cur:
                for _ in range(6):
                    tempo_site = random.randint(1, 10)
                    paginas = random.randint(1, 8)
                    clicou_preco = random.randint(0, 1)
                    base = 0.10
                    base += min(tempo_site / 400, 0.25)
                    base += min(paginas / 10, 0.25)
                    base += 0.20 if clicou_preco else 0.0
                    prob = max(0.02, min(0.98, base))
                    score = int(round(prob * 100))
                    label = 1 if prob >= 0.70 else (0 if prob < 0.35 else None)
                    cur.execute(
                        """
                        INSERT INTO leads (client_id, nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco,
                                           payload, probabilidade, score, label, created_at, updated_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,NOW(),NOW())
                        """,
                        (
                            client_id,
                            "Demo Lead",
                            "demo@leadrank.local",
                            "(11) 90000-0000",
                            tempo_site,
                            paginas,
                            clicou_preco,
                            json.dumps(data),
                            float(prob),
                            int(score),
                            label,
                        ),
                    )
        return json_ok({"client_id": client_id, "inserted": 6})
    finally:
        conn.close()


@leads_bp.post("/seed_test_leads")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def seed_test_leads():
    data = request.get_json(silent=True) or {}
    client_id = get_client_id_from_request()
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, row, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    n = safe_int(data.get("n"), 15)
    n = max(1, min(n, 200))

    ok_quota, msg_quota, extra = check_quota_and_bump(client_id, row)
    if not ok_quota:
        return json_err(msg_quota, 402, **extra)

    conn = db()
    inserted = 0
    conv = 0
    neg = 0
    try:
        with conn:
            with conn.cursor() as cur:
                for _ in range(n):
                    tempo_site = random.randint(1, 10)
                    paginas = random.randint(1, 8)
                    clicou_preco = random.randint(0, 1)
                    base = 0.10
                    base += min(tempo_site / 400, 0.25)
                    base += min(paginas / 10, 0.25)
                    base += 0.20 if clicou_preco else 0.0
                    prob = max(0.02, min(0.98, base))
                    score = int(round(prob * 100))
                    label_vc = None
                    if prob >= 0.7 and random.random() > 0.3:
                        label_vc = 1
                        conv += 1
                    elif prob < 0.35 and random.random() > 0.7:
                        label_vc = 0
                        neg += 1
                    label = 1 if prob >= 0.70 else (0 if prob < 0.35 else None)

                    payload = {
                        "tempo_site": tempo_site,
                        "paginas_visitadas": paginas,
                        "clicou_preco": clicou_preco,
                    }
                    cur.execute(
                        """
                        INSERT INTO leads (client_id, nome, email_lead, telefone, tempo_site, paginas_visitadas, clicou_preco,
                                           payload, probabilidade, score, label, virou_cliente, created_at, updated_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,NOW(),NOW())
                        """,
                        (
                            client_id,
                            "Lead Teste",
                            "lead@teste.com",
                            "(11) 90000-0000",
                            tempo_site,
                            paginas,
                            clicou_preco,
                            json.dumps(payload),
                            float(prob),
                            int(score),
                            label,
                            label_vc,
                        ),
                    )
                    inserted += 1

            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE clients SET leads_used_month = leads_used_month + %s, updated_at=NOW() WHERE client_id=%s",
                    (inserted, client_id),
                )

        cache_delete(f"acao_do_dia:{client_id}")
        cache_delete_prefix(f"insights:{client_id}:")
        return json_ok(
            {
                "client_id": client_id,
                "inserted": inserted,
                "converted": conv,
                "denied": neg,
                "pending": inserted - conv - neg,
            }
        )
    finally:
        conn.close()


@leads_bp.get("/funnels")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def funnels():
    client_id = get_client_id_from_request()
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT
                      COUNT(*) FILTER (WHERE (probabilidade IS NOT NULL AND probabilidade >= 0.70)
                                     OR (score IS NOT NULL AND score >= 70)) AS hot,
                      COUNT(*) FILTER (WHERE (probabilidade IS NOT NULL AND probabilidade >= 0.35 AND probabilidade < 0.70)
                                     OR (score IS NOT NULL AND score >= 35 AND score < 70)) AS warm,
                      COUNT(*) FILTER (WHERE (probabilidade IS NOT NULL AND probabilidade < 0.35)
                                     OR (score IS NOT NULL AND score < 35)) AS cold
                    {active_leads_query}
                      AND client_id=%s
                    """,
                    (client_id,),
                )
                row = cur.fetchone() or {}
        return json_ok(
            {
                "client_id": client_id,
                "hot": int(row.get("hot") or 0),
                "warm": int(row.get("warm") or 0),
                "cold": int(row.get("cold") or 0),
            }
        )
    finally:
        conn.close()


@leads_bp.get("/acao_do_dia")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def acao_do_dia():
    client_id = get_client_id_from_request()
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    cache_key = f"acao_do_dia:{client_id}"
    cached = cache_get_json(cache_key)
    if cached:
        return json_ok(cached)

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT id, nome, email_lead, telefone, origem,
                           score, probabilidade, created_at, virou_cliente
                    {active_leads_query}
                      AND client_id=%s
                    ORDER BY COALESCE(probabilidade, score / 100.0) DESC NULLS LAST,
                             created_at DESC
                    LIMIT 30
                    """,
                    (client_id,),
                )
                rows = cur.fetchall() or []
        res = []
        for item in rows:
            res.append(
                {
                    "id": item.get("id"),
                    "nome": item.get("nome"),
                    "email": item.get("email_lead"),
                    "telefone": item.get("telefone"),
                    "origem": item.get("origem"),
                    "score": safe_int(item.get("score"), 0) if item.get("score") is not None else None,
                    "probabilidade": safe_float(item.get("probabilidade"), 0.0),
                    "created_at": iso(item.get("created_at")),
                    "virou_cliente": item.get("virou_cliente"),
                    "temperatura": lead_temperature(item.get("probabilidade"), item.get("score")),
                }
            )
        payload = {"client_id": client_id, "rows": res}
        cache_set_json(cache_key, payload)
        return json_ok(payload)
    finally:
        conn.close()


@leads_bp.get("/lead_explain")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def lead_explain():
    client_id = get_client_id_from_request()
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    lead_id = safe_int(request.args.get("lead_id"), 0)
    if not lead_id:
        return json_err("lead_id obrigatório", 400)

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT tempo_site, paginas_visitadas, clicou_preco, probabilidade
                    {active_leads_query}
                      AND client_id=%s
                      AND id=%s
                    """,
                    (client_id, lead_id),
                )
                lead = cur.fetchone()
        if not lead:
            return json_err("Lead não encontrado", 404)

        score = float(lead.get("probabilidade") or 0.0)
        return json_ok(
            {
                "client_id": client_id,
                "lead_id": lead_id,
                "tempo_site": lead.get("tempo_site"),
                "paginas_visitadas": lead.get("paginas_visitadas"),
                "clicou_preco": lead.get("clicou_preco"),
                "probabilidade": score,
                "note": "Explicação do score heurístico (antes do treino).",
            }
        )
    finally:
        conn.close()
