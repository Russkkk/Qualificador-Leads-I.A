from __future__ import annotations

from dataclasses import dataclass

import requests

from services import settings


@dataclass
class CaptchaResult:
    ok: bool
    error: str = ""


def verify_turnstile(token: str, remoteip: str | None = None) -> CaptchaResult:
    """Verifica token do Cloudflare Turnstile.

    - Não faz nada se TURNSTILE_SECRET_KEY não estiver configurado.
    - Em caso de falha de rede, respeita CAPTCHA_SOFT_FAIL.
    """

    if not settings.TURNSTILE_SECRET_KEY:
        return CaptchaResult(ok=True)

    token = (token or "").strip()
    if not token:
        return CaptchaResult(ok=False, error="missing_token")

    payload = {
        "secret": settings.TURNSTILE_SECRET_KEY,
        "response": token,
    }
    if remoteip:
        payload["remoteip"] = remoteip

    try:
        resp = requests.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data=payload,
            timeout=settings.CAPTCHA_TIMEOUT_SECONDS,
        )
        data = resp.json() if resp.ok else {}
        if data.get("success") is True:
            return CaptchaResult(ok=True)

        # Turnstile retorna error-codes como lista.
        codes = data.get("error-codes") or []
        code = codes[0] if isinstance(codes, list) and codes else "invalid"
        return CaptchaResult(ok=False, error=str(code))
    except Exception:
        # Se estiver em soft-fail, não bloqueia usuário por instabilidade.
        if settings.CAPTCHA_SOFT_FAIL:
            return CaptchaResult(ok=True, error="soft_fail")
        return CaptchaResult(ok=False, error="network_error")
