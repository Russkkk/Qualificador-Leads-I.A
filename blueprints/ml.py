from flask import Blueprint, request

from extensions import limiter
from services.auth_service import require_client_auth
from services.db import get_active_leads_query
from services.lead_service import get_labeled_rows, get_threshold, set_threshold, update_probabilities
from services.ml_service import HAS_ML, best_threshold, can_train, compute_precision_recall, features_from_row, predict_for_rows, train_pipeline
from services.utils import get_client_id_from_request, json_err, json_ok, rate_limit_client_id, safe_int

ml_bp = Blueprint("ml", __name__)


@ml_bp.get("/recalc_pending")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def recalc_pending():
    if not HAS_ML:
        return json_ok(
            {
                "client_id": get_client_id_from_request() or "",
                "can_train": False,
                "updated": 0,
                "reason": "Modelo de ML não instalado. Mantendo heurística base.",
                "code": "ml_missing",
            }
        )

    client_id = get_client_id_from_request()
    limit = safe_int(request.args.get("limit"), 500)
    limit = max(10, min(limit, 5000))
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    labeled = get_labeled_rows(client_id)
    can, reason, classes = can_train(labeled)
    if not can:
        return json_ok(
            {
                "client_id": client_id,
                "can_train": False,
                "classes_rotuladas": classes,
                "labeled_count": len(labeled),
                "reason": reason,
                "updated": 0,
            }
        )

    import numpy as np

    X = np.vstack([features_from_row(r) for r in labeled])
    y = np.array([1 if float(r["virou_cliente"]) == 1.0 else 0 for r in labeled], dtype=int)
    pipe = train_pipeline(X, y)

    from services.db import db
    from psycopg.rows import dict_row

    conn = db()
    try:
        with conn:
            with conn.cursor(row_factory=dict_row) as cur:
                active_leads_query = get_active_leads_query()
                cur.execute(
                    f"""
                    SELECT id, tempo_site, paginas_visitadas, clicou_preco
                    {active_leads_query}
                      AND client_id=%s
                      AND virou_cliente IS NULL
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (client_id, int(limit)),
                )
                pending = [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()

    ids = [int(r["id"]) for r in pending]
    probs = predict_for_rows(pipe, pending)
    updated = update_probabilities(client_id, ids, probs)

    return json_ok(
        {
            "client_id": client_id,
            "can_train": True,
            "classes_rotuladas": classes,
            "labeled_count": len(labeled),
            "updated": updated,
            "min_prob": float(min(probs)) if probs else None,
            "max_prob": float(max(probs)) if probs else None,
            "sample": [{"id": ids[i], "prob": float(probs[i])} for i in range(min(5, len(ids)))],
        }
    )


@ml_bp.post("/auto_threshold")
@limiter.limit("600 per minute", key_func=rate_limit_client_id)
def auto_threshold():
    if not HAS_ML:
        client_id = get_client_id_from_request()
        threshold = get_threshold(client_id) if client_id else 0.35
        return json_ok(
            {
                "client_id": client_id or "",
                "can_train": False,
                "threshold": float(threshold),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "reason": "Modelo de ML não instalado. Usando heurística base.",
                "code": "ml_missing",
            }
        )

    client_id = get_client_id_from_request()
    if not client_id:
        return json_err("client_id obrigatório", 400)

    ok_auth, _, msg = require_client_auth(client_id)
    if not ok_auth:
        return json_err(msg, 403, code="auth_required")

    labeled = get_labeled_rows(client_id)
    can, reason, classes = can_train(labeled)
    if not can:
        return json_ok(
            {
                "client_id": client_id,
                "can_train": False,
                "classes_rotuladas": classes,
                "labeled_count": len(labeled),
                "reason": reason,
                "threshold": get_threshold(client_id),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        )

    import numpy as np

    missing = [r for r in labeled if r.get("probabilidade") is None]
    if missing:
        X = np.vstack([features_from_row(r) for r in labeled])
        y = np.array([1 if float(r["virou_cliente"]) == 1.0 else 0 for r in labeled], dtype=int)
        pipe = train_pipeline(X, y)
        ids = [int(r["id"]) for r in missing]
        probs = predict_for_rows(pipe, missing)
        update_probabilities(client_id, ids, probs)
        labeled = get_labeled_rows(client_id)

    best_t = best_threshold(labeled)
    set_threshold(client_id, best_t)

    metrics = compute_precision_recall(labeled, best_t)
    return json_ok(
        {
            "client_id": client_id,
            "threshold": float(best_t),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        }
    )
