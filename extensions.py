from flask_limiter import Limiter
from flask_login import LoginManager

from services.utils import client_ip

limiter = Limiter(key_func=client_ip, default_limits=["100 per minute"])
login_manager = LoginManager()
