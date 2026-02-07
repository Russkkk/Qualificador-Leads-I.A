from typing import Any, Dict, List, Tuple

from services import settings
from services.utils import safe_int

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    HAS_ML = True
except Exception:
    HAS_ML = False


def features_from_row(row: Dict[str, Any]):
    tempo = safe_int(row.get("tempo_site"), 0)
    paginas = safe_int(row.get("paginas_visitadas"), 0)
    clicou = safe_int(row.get("clicou_preco"), 0)
    return np.array([tempo, paginas, clicou], dtype=float)


def train_pipeline(X, y):
    pipe = Pipeline(steps=[("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=200, solver="lbfgs"))])
    pipe.fit(X, y)
    return pipe


def can_train(labeled_rows: List[Dict[str, Any]]) -> Tuple[bool, str, List[float]]:
    classes = sorted(list({float(r["virou_cliente"]) for r in labeled_rows if r.get("virou_cliente") is not None}))
    if len(labeled_rows) < settings.MIN_LABELED_TO_TRAIN:
        return False, (
            f"Poucos exemplos rotulados. Recomendo no mínimo {settings.MIN_LABELED_TO_TRAIN} (2 de cada classe) para começar."
        ), classes
    if len(classes) < 2:
        return False, "Precisa de exemplos das duas classes (convertido e negado) para treinar.", classes
    return True, "", classes


def predict_for_rows(pipe, rows: List[Dict[str, Any]]) -> List[float]:
    if not rows:
        return []
    X = np.vstack([features_from_row(r) for r in rows])
    probs = pipe.predict_proba(X)[:, 1]
    return probs.tolist()


def compute_precision_recall(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, float]:
    y_true = []
    y_pred = []
    for row in rows:
        y = row.get("virou_cliente")
        p = row.get("probabilidade")
        if y is None or p is None:
            continue
        y_true.append(1 if float(y) == 1.0 else 0)
        y_pred.append(1 if float(p) >= threshold else 0)

    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def best_threshold(rows: List[Dict[str, Any]]) -> float:
    candidates = [i / 100 for i in range(5, 96, 5)]
    best_t = settings.DEFAULT_THRESHOLD
    best_f1 = -1.0
    for t in candidates:
        metrics = compute_precision_recall(rows, t)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = t
    return float(best_t)
