"""Tools: web search and page reading, with a disk cache so re-runs cost nothing."""
import hashlib
import ipaddress
import json
import os
import socket
from urllib.parse import urlparse
from ddgs import DDGS
import httpx
import trafilatura

CACHE_DIR = ".cache"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def is_safe_url(url: str) -> bool:
    """Only public http(s) pages. Blocks localhost, private networks and odd schemes."""
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        if p.scheme not in ("http", "https") or not host:
            return False
        if host == "localhost" or host.endswith((".local", ".internal")):
            return False
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except Exception:
        return False


def _fetch_html(url: str):
    """Try the default downloader first, then a browser-like request."""
    html = trafilatura.fetch_url(url)
    if html:
        return html
    try:
        r = httpx.get(url, headers=UA, timeout=15, follow_redirects=True)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def clean_text(s: str) -> str:
    """Replace odd Unicode characters that cause glued words like 'usersafter'."""
    if not s:
        return ""
    for ch in ("\u00a0", "\u202f", "\u2009", "\u200a", "\u2007", "\u2002", "\u2003"):
        s = s.replace(ch, " ")
    for ch in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        s = s.replace(ch, "-")
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"):
        s = s.replace(ch, "")
    return (s.replace("\u2018", "'").replace("\u2019", "'")
             .replace("\u201c", '"').replace("\u201d", '"'))


def _cache_path(kind: str, key: str) -> str:
    return os.path.join(CACHE_DIR, f"{kind}_{hashlib.md5(key.encode()).hexdigest()}.json")


def _cache_get(kind, key):
    try:
        with open(_cache_path(kind, key), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _cache_set(kind, key, value):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(kind, key), "w", encoding="utf-8") as f:
            json.dump(value, f)
    except OSError:
        pass


def search_web(query: str, max_results: int = 5) -> list[dict]:
    cached = _cache_get("search", query)
    if cached is not None:
        return cached
    try:
        with DDGS() as d:
            results = d.text(query, max_results=max_results)
        out = [{"title": r["title"], "url": r["href"], "snippet": r["body"][:200]} for r in results]
        _cache_set("search", query, out)
        return out
    except Exception as e:
        return [{"error": f"search failed: {e}"}]


def read_url(url: str, max_chars: int = 4000) -> tuple[str, str]:
    """Returns (title, text). Text starts with 'ERROR' if the page can't be read."""
    if not is_safe_url(url):
        return "", "ERROR: blocked unsafe or unreachable URL"
    cached = _cache_get("page", url)
    if cached:
        return cached["title"], cached["text"]
    try:
        html = _fetch_html(url)
        if not html:
            return "", "ERROR: could not fetch this page"
        text = trafilatura.extract(html) or ""
        if not text.strip():
            return "", "ERROR: no readable text on this page"
        meta = trafilatura.extract_metadata(html)
        title = meta.title if meta and meta.title else url
        result = {"title": clean_text(title), "text": clean_text(text)[:max_chars]}
        _cache_set("page", url, result)
        return result["title"], result["text"]
    except Exception as e:
        return "", f"ERROR: {e}"