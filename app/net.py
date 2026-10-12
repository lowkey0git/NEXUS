"""SSRF-resistant HTTP GET. Every hop (including redirects) is resolved and checked against
private/loopback/link-local/reserved ranges before connecting.

Known limitation: DNS is resolved once for the check and again by the HTTP client, so a hostile DNS
server could in theory rebind between the two. For hostile multi-tenant deployments, run fetches
through an egress proxy that enforces the same rules."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 5


class BlockedURL(Exception):
    pass


def resolve_host(host: str) -> list[str]:
    try:
        return sorted({ai[4][0] for ai in socket.getaddrinfo(host, None)})
    except socket.gaierror as e:
        raise BlockedURL(f"Could not resolve host '{host}'.") from e


def check_url(url: str, resolver=None) -> None:
    resolver = resolver or resolve_host
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise BlockedURL("Only http and https URLs are allowed.")
    if not p.hostname:
        raise BlockedURL("URL has no host.")
    if p.username or p.password:
        raise BlockedURL("URLs with embedded credentials are not allowed.")
    for ip in resolver(p.hostname):
        a = ipaddress.ip_address(ip.split("%")[0])
        if a.is_private or a.is_loopback or a.is_link_local or a.is_reserved or a.is_multicast or a.is_unspecified:
            raise BlockedURL("That address is on a private or internal network and is blocked.")


def safe_get(url: str, timeout: float = 20, transport: httpx.BaseTransport | None = None, resolver=None):
    """Returns (final_url, content_type, body_bytes, truncated)."""
    headers = {"User-Agent": "PersonalAssistant/1.0 (+fetch tool)", "Accept": "text/html,text/plain,application/json,application/pdf;q=0.9,*/*;q=0.1"}
    with httpx.Client(timeout=timeout, follow_redirects=False, transport=transport, headers=headers) as c:
        for _ in range(MAX_REDIRECTS + 1):
            check_url(url, resolver)
            with c.stream("GET", url) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    url = urljoin(url, r.headers["location"])
                    continue
                if r.status_code >= 400:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                buf, truncated = bytearray(), False
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > MAX_BYTES:
                        truncated = True
                        del buf[MAX_BYTES:]
                        break
                return url, r.headers.get("content-type", ""), bytes(buf), truncated
    raise BlockedURL("Too many redirects.")
