from flask_limiter import Limiter
from flask_login import LoginManager

from services import settings
from services.utils import client_ip

storage_uri = settings.RATELIMIT_STORAGE_URI or "memory://"
limiter = Limiter(
    key_func=client_ip,
    default_limits=["100 per minute"],
    storage_uri=storage_uri,
)
login_manager = LoginManager()
