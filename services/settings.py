import os

from dotenv import load_dotenv

from config import get_config

load_dotenv()
config = get_config()

DATABASE_URL = config.DATABASE_URL
DEMO_KEY = config.DEMO_KEY
ADMIN_KEY = os.getenv("ADMIN_KEY", "").strip()
DEBUG_MODE = config.DEBUG_MODE
INCLUDE_TRACEBACK = config.INCLUDE_TRACEBACK
TRUST_PROXY = config.TRUST_PROXY
REQUIRE_API_KEY = config.REQUIRE_API_KEY

STRIPE_SECRET_KEY = config.STRIPE_SECRET_KEY
STRIPE_PRICE_IDS_JSON = config.STRIPE_PRICE_IDS_JSON
BILLING_WEBHOOK_SECRET = config.BILLING_WEBHOOK_SECRET

KIWIFY_ACCOUNT_ID = config.KIWIFY_ACCOUNT_ID
KIWIFY_CLIENT_SECRET = config.KIWIFY_CLIENT_SECRET
KIWIFY_API_KEY = config.KIWIFY_API_KEY
KIWIFY_WEBHOOK_TOKEN = config.KIWIFY_WEBHOOK_TOKEN

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "")


def _bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


ADMIN_ALLOW_DEMO_KEY = _bool(os.getenv("ADMIN_ALLOW_DEMO_KEY", ""))

def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


# CORS: em produção, deny-by-default quando ALLOWED_ORIGINS não estiver definido.
_allowed_origins_env = _split_csv(os.getenv("ALLOWED_ORIGINS", ""))
ALLOWED_ORIGINS = [origin for origin in _allowed_origins_env if origin not in {"null", "*"}]
if not ALLOWED_ORIGINS and DEBUG_MODE:
    ALLOWED_ORIGINS = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost",
        "http://127.0.0.1",
    ]

PLAN_CATALOG = {
    "demo": {"price_brl_month": 0, "setup_fee_brl": 0, "lead_limit_month": 30},
    "trial": {"price_brl_month": 0, "setup_fee_brl": 0, "lead_limit_month": 100},
    "starter": {"price_brl_month": 79, "setup_fee_brl": 0, "lead_limit_month": 1000},
    "pro": {"price_brl_month": 179, "setup_fee_brl": 0, "lead_limit_month": 5000},
    "enterprise": {"price_brl_month": 279, "setup_fee_brl": 0, "lead_limit_month": 20000},
    "vip": {"price_brl_month": 279, "setup_fee_brl": 0, "lead_limit_month": 20000},
}

MAX_PREVER_PAYLOAD_BYTES = int(os.getenv("MAX_PREVER_PAYLOAD_BYTES", "51200"))
ENABLE_HSTS = _bool(os.getenv("ENABLE_HSTS", "true"))
HSTS_MAX_AGE = int(os.getenv("HSTS_MAX_AGE", "31536000"))

SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "").strip()
SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))

REDIS_URL = os.getenv("REDIS_URL", "").strip()
RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", REDIS_URL).strip()
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "60"))

DEFAULT_CSP = (
    "default-src 'self'; "
    "base-uri 'self'; "
    "object-src 'none'; "
    "frame-ancestors 'none'; "
    "img-src 'self' data: https:; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "script-src 'self' https://cdn.jsdelivr.net https://unpkg.com https://challenges.cloudflare.com; "
    "connect-src 'self' https://qualificador-leads-ia.onrender.com https://leadrank.com.br; "
    "frame-src https://www.youtube.com https://challenges.cloudflare.com"
)
CSP_POLICY = os.getenv("CSP_POLICY", DEFAULT_CSP)

DEFAULT_LIMIT = 200
DEFAULT_THRESHOLD = 0.35
MIN_LABELED_TO_TRAIN = 4
PBKDF2_ITERATIONS = 390_000

# --- Public (front-end) feature flags / anti-abuse ---
# Demo mode: expõe endpoints read-only com dados de exemplo.
DEMO_MODE = _bool(os.getenv("DEMO_MODE", ""))

# Captcha (Cloudflare Turnstile): opcional e sem breaking change.
# - Se TURNSTILE_SECRET_KEY estiver vazio: captcha desativado.
# - Se CAPTCHA_ENFORCE=true: token passa a ser obrigatório em /signup.
TURNSTILE_SITE_KEY = os.getenv("TURNSTILE_SITE_KEY", "").strip()
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "").strip()
CAPTCHA_ENFORCE = _bool(os.getenv("CAPTCHA_ENFORCE", ""))
CAPTCHA_TIMEOUT_SECONDS = _float(os.getenv("CAPTCHA_TIMEOUT_SECONDS", "3"), 3.0)
CAPTCHA_SOFT_FAIL = _bool(os.getenv("CAPTCHA_SOFT_FAIL", "true"))

# Report de erros do front (opcional). Se ligado, o front pode enviar erros JS para /client_error.
CLIENT_ERROR_REPORTING = _bool(os.getenv("CLIENT_ERROR_REPORTING", ""))
CLIENT_ERROR_SAMPLE_RATE = _float(os.getenv("CLIENT_ERROR_SAMPLE_RATE", "0.05"), 0.05)
