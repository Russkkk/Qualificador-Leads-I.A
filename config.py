import os


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


class BaseConfig:
    DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
    DEMO_KEY = os.getenv("DEMO_KEY", "").strip()
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "").strip()
    DEBUG_MODE = _bool(os.getenv("DEBUG", ""))
    INCLUDE_TRACEBACK = _bool(os.getenv("INCLUDE_TRACEBACK", ""))
    TRUST_PROXY = _bool(os.getenv("TRUST_PROXY", ""))
    REQUIRE_API_KEY = _bool(os.getenv("REQUIRE_API_KEY", ""))

    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
    STRIPE_PRICE_IDS_JSON = os.getenv("STRIPE_PRICE_IDS_JSON", "").strip()
    BILLING_WEBHOOK_SECRET = os.getenv("BILLING_WEBHOOK_SECRET", "").strip()

    KIWIFY_ACCOUNT_ID = os.getenv("KIWIFY_ACCOUNT_ID", "").strip()
    KIWIFY_CLIENT_SECRET = os.getenv("KIWIFY_CLIENT_SECRET", "").strip()
    KIWIFY_API_KEY = os.getenv("KIWIFY_API_KEY", "").strip()
    KIWIFY_WEBHOOK_TOKEN = os.getenv("KIWIFY_WEBHOOK_TOKEN", "").strip()


class DevConfig(BaseConfig):
    DEBUG_MODE = True
    INCLUDE_TRACEBACK = True


class ProdConfig(BaseConfig):
    DEBUG_MODE = False


class TestConfig(BaseConfig):
    DEBUG_MODE = True
    INCLUDE_TRACEBACK = True
    TESTING = True


def get_config() -> BaseConfig:
    env = os.getenv("APP_ENV", "prod").strip().lower()
    if env == "dev":
        return DevConfig()
    if env == "test":
        return TestConfig()
    return ProdConfig()
