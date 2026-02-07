from dataclasses import dataclass

from flask_login import UserMixin


@dataclass
class AuthUser(UserMixin):
    client_id: str
    email: str
    plan: str
    status: str

    def get_id(self) -> str:
        return self.client_id
