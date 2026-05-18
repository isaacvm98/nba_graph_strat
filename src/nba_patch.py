"""Monkey-patch nba_api to route requests through curl_cffi + webshare proxy.

NBA's Akamai bot manager blocks requests by Python's default `requests` TLS
fingerprint. We use `curl_cffi` to impersonate Chrome's TLS handshake, a
webshare proxy to get a fresh egress IP, and a session that warms up by
visiting nba.com so Akamai issues us valid `bm_*` cookies before we hit
sensitive endpoints like leaguedashlineups.

Apply by importing this module before using any nba_api endpoint:

    from src import nba_patch  # noqa: F401
"""
import random
import threading
from pathlib import Path
from urllib.parse import quote_plus

from curl_cffi import requests as creq
from nba_api.library import http as nba_http

PROXY_FILE = Path(__file__).resolve().parent.parent / "data" / "secrets" / "proxies.txt"
IMPERSONATE = "chrome"
WARMUP_URL = "https://www.nba.com/stats/lineups/traditional"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def _load_proxies() -> list[str]:
    if not PROXY_FILE.exists():
        return []
    proxies = []
    for line in PROXY_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) == 4:
            host, port, user, pwd = parts
            proxies.append(f"http://{user}:{pwd}@{host}:{port}")
    return proxies


_PROXIES = _load_proxies()
_session_lock = threading.Lock()
_session = None
_session_proxy = None


def _get_warmed_session():
    global _session, _session_proxy
    with _session_lock:
        if _session is not None:
            return _session
        proxy = random.choice(_PROXIES) if _PROXIES else None
        s = creq.Session(impersonate=IMPERSONATE)
        if proxy:
            s.proxies = {"http": proxy, "https": proxy}
        s.headers.update(DEFAULT_HEADERS)
        try:
            s.get(WARMUP_URL, timeout=30)
        except Exception:
            pass
        _session = s
        _session_proxy = proxy
        return _session


def reset_session():
    """Force a new warmed session (call if cookies expire or 401/403 starts)."""
    global _session
    with _session_lock:
        _session = None


def _send_api_request(
    self,
    endpoint,
    parameters,
    referer=None,
    proxy=None,
    headers=None,
    timeout=None,
    raise_exception_on_error=False,
):
    if not self.base_url:
        raise Exception("Cannot use send_api_request from _HTTP class.")
    base_url = self.base_url.format(endpoint=endpoint)

    s = _get_warmed_session()

    request_headers = dict(s.headers)
    request_headers.update(self.headers or {})
    if headers:
        request_headers.update(headers)
    if referer:
        request_headers["Referer"] = referer

    parameters = sorted(parameters.items(), key=lambda kv: kv[0])
    param_string = "&".join(
        "{}={}".format(k, "" if v is None else quote_plus(str(v))) for k, v in parameters
    )
    url = f"{base_url}?{param_string}"

    response = s.get(url, headers=request_headers, timeout=timeout or 30)

    self.nba_response = self.nba_response(
        response=self.clean_contents(response.text),
        status_code=response.status_code,
        url=str(response.url),
    )
    return self.nba_response


nba_http.NBAHTTP.send_api_request = _send_api_request
