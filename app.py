#!/usr/bin/env python3
"""
================================================================================
 TikTok Downloader — a single-file, no-watermark TikTok video downloader
================================================================================

Local run:
    python app.py

    Starts a local Flask dev server, finds a free port, and opens your
    default browser automatically.

Production (e.g. Render, or any host running gunicorn):
    gunicorn app:app --workers 1 --threads 8 --timeout 120 -b 0.0.0.0:$PORT

    Production hosts import the `app` object directly and never execute
    main(), so the port-scanning/browser-auto-open logic below is local-dev
    only and simply never runs in that path. Keep --workers 1: the resolve
    cache and rate limiter are in-process memory, so multiple worker
    processes would not share state (use --threads to add concurrency
    instead).

Everything — HTML, CSS and JavaScript — is embedded in this one file. No
templates/ or static/ folders are used.

See requirements.txt and Procfile alongside this file for deployment.

Author footer credit: Dr. Khert Laguna Garde
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse, quote

import requests
from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.middleware.proxy_fix import ProxyFix

# ==============================================================================
# Configuration
# ==============================================================================

APP_NAME = "GlassTok"
APP_TAGLINE = "TikTok, without the watermark."
HOST = "127.0.0.1"
PORT_RANGE = range(5100, 5200)          # ports we will try, in order (local dev only)
REQUEST_TIMEOUT = 15                     # seconds, for metadata calls
STREAM_TIMEOUT = 30                      # seconds, for the download proxy
CACHE_TTL_SECONDS = 60 * 30              # how long resolved links stay cached
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Basic per-IP rate limits so a public deployment can't hammer the free
# upstream providers into rate-limiting (or banning) everyone at once.
# This is intentionally a lightweight in-memory limiter — no Redis, no extra
# service — which is why the production start command pins a single worker
# (see Procfile). Good enough for a small free-tier deployment; swap in
# Flask-Limiter + Redis if this ever needs to scale past one dyno/instance.
RATE_LIMIT_RESOLVE = (10, 60)     # 10 resolves per 60s per IP
RATE_LIMIT_DOWNLOAD = (15, 300)   # 15 downloads per 5 minutes per IP

# Domains we trust the download proxy to fetch from. This is a hard allowlist
# used to stop the /api/download endpoint from being abused as an open,
# server-side-request-forgery-capable proxy for arbitrary URLs.
ALLOWED_MEDIA_HOST_SUFFIXES = (
    "tiktokcdn.com",
    "tiktokcdn-us.com",
    "tiktokcdn-eu.com",
    "tiktokv.com",
    "tiktokv.us",
    "muscdn.com",
    "ibyteimg.com",
    "byteoversea.com",
    "tikwm.com",
    "tiktok.com",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("glasstok")

app = Flask(__name__)
app.url_map.strict_slashes = False
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # requests to this app are tiny JSON bodies only

# Render (and most PaaS providers) sit the app behind a reverse proxy, so the
# real client IP/scheme arrive via X-Forwarded-* headers. ProxyFix makes
# request.remote_addr trustworthy again, which the rate limiter below relies on.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)


# ==============================================================================
# In-memory cache
# ==============================================================================
# Resolved metadata is cached briefly, keyed by video id, so the "Download"
# button doesn't have to re-resolve the video and so repeated clicks are fast.
# This is intentionally a plain dict guarded by a lock — no external services,
# no database, nothing that would violate the "zero configuration" requirement.

_cache: dict[str, "VideoInfo"] = {}
_cache_lock = threading.Lock()


def _cache_get(video_id: str) -> Optional["VideoInfo"]:
    with _cache_lock:
        entry = _cache.get(video_id)
        if entry is None:
            return None
        if time.time() - entry.cached_at > CACHE_TTL_SECONDS:
            del _cache[video_id]
            return None
        return entry


def _cache_set(info: "VideoInfo") -> None:
    with _cache_lock:
        info.cached_at = time.time()
        _cache[info.video_id] = info
        # Opportunistic cleanup so the dict never grows unbounded during a
        # long-running session.
        stale = [
            vid
            for vid, v in _cache.items()
            if time.time() - v.cached_at > CACHE_TTL_SECONDS
        ]
        for vid in stale:
            del _cache[vid]


# ==============================================================================
# Data model
# ==============================================================================

@dataclass
class VideoInfo:
    video_id: str
    title: str
    author_username: str
    author_nickname: str
    cover_url: str
    duration: int                # seconds
    width: int
    height: int
    no_watermark_url: str
    hd_url: Optional[str]
    watermark_url: Optional[str]
    music_url: Optional[str]
    size_bytes: Optional[int]
    hd_size_bytes: Optional[int]
    source_provider: str
    cached_at: float = field(default_factory=time.time)

    def to_public_dict(self) -> dict:
        """Everything the frontend needs to render the preview card."""
        return {
            "id": self.video_id,
            "title": self.title or "Untitled TikTok video",
            "author": self.author_username,
            "authorNickname": self.author_nickname or self.author_username,
            "cover": self.cover_url,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}" if self.width and self.height else "Unknown",
            "sizeBytes": self.size_bytes,
            "hdSizeBytes": self.hd_size_bytes,
            "hasHd": bool(self.hd_url),
            "hasMusic": bool(self.music_url),
            "provider": self.source_provider,
            "downloadUrls": {
                "sd": f"/api/download?id={quote(self.video_id)}&type=sd",
                "hd": f"/api/download?id={quote(self.video_id)}&type=hd" if self.hd_url else None,
                "watermark": (
                    f"/api/download?id={quote(self.video_id)}&type=watermark"
                    if self.watermark_url
                    else None
                ),
                "music": f"/api/download?id={quote(self.video_id)}&type=music" if self.music_url else None,
            },
        }


class ResolveError(Exception):
    """Raised when a TikTok URL cannot be resolved into media links."""


# ==============================================================================
# Rate limiting (in-memory, per-IP sliding window)
# ==============================================================================

_rl_lock = threading.Lock()
_rl_buckets: dict[str, list[float]] = {}


def _client_key() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()


def rate_limited(scope: str, max_calls: int, window_seconds: int):
    """Decorator: reject a client with HTTP 429 once they exceed max_calls
    inside window_seconds. Bucketed per scope so /api/resolve and
    /api/download are limited independently."""

    def decorator(fn):
        def wrapped(*args, **kwargs):
            key = f"{scope}:{_client_key()}"
            now = time.time()
            with _rl_lock:
                hits = [t for t in _rl_buckets.get(key, []) if now - t < window_seconds]
                if len(hits) >= max_calls:
                    retry_after = int(window_seconds - (now - hits[0])) + 1
                    _rl_buckets[key] = hits
                    return (
                        jsonify(
                            ok=False,
                            error="Too many requests — please slow down and try again shortly.",
                        ),
                        429,
                        {"Retry-After": str(retry_after)},
                    )
                hits.append(now)
                _rl_buckets[key] = hits
            return fn(*args, **kwargs)

        wrapped.__name__ = fn.__name__
        return wrapped

    return decorator


# ==============================================================================
# URL validation & normalisation
# ==============================================================================

TIKTOK_HOST_PATTERN = re.compile(
    r"^(?:www\.|vm\.|vt\.|m\.)?tiktok\.com$", re.IGNORECASE
)

TIKTOK_URL_HINT = re.compile(
    r"tiktok\.com/(?:@[\w.\-]+/video/\d+|v/\d+|t/\w+|\S+)", re.IGNORECASE
)


def is_probably_tiktok_url(raw: str) -> bool:
    """Cheap, fast sanity check before we spend a network round trip."""
    if not raw or len(raw) > 2048:
        return False
    try:
        parsed = urlparse(raw.strip())
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return host.endswith("tiktok.com")


def resolve_short_link(url: str) -> str:
    """
    Short links (vm.tiktok.com/..., vt.tiktok.com/...) redirect to the full
    canonical URL. We follow redirects manually (HEAD, falling back to GET)
    so we can validate the final host before trusting it.
    """
    try:
        resp = requests.head(
            url,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
            timeout=REQUEST_TIMEOUT,
        )
        final_url = resp.url
        if resp.status_code >= 400 or final_url == url:
            # Some CDNs reject HEAD requests; retry with a lightweight GET.
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                allow_redirects=True,
                timeout=REQUEST_TIMEOUT,
                stream=True,
            )
            final_url = resp.url
            resp.close()
        return final_url
    except requests.RequestException as exc:
        raise ResolveError(f"Could not follow that link ({exc.__class__.__name__}).") from exc


def sanitize_filename(name: str, fallback: str = "tiktok_video") -> str:
    """Strip anything that isn't safe for a filesystem filename."""
    name = (name or fallback).strip()
    name = re.sub(r"[^\w\s\-.]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._") or fallback
    return name[:80]


# ==============================================================================
# Providers — each knows how to turn a TikTok URL into a VideoInfo.
# If one fails (network error, schema change, rate limit) the caller moves on
# to the next provider automatically.
# ==============================================================================

def _provider_tikwm(url: str) -> VideoInfo:
    """Primary provider: tikwm.com public JSON API. No key required."""
    api_url = "https://www.tikwm.com/api/"
    resp = requests.post(
        api_url,
        data={"url": url, "hd": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("code") != 0 or "data" not in payload:
        msg = payload.get("msg", "Unknown provider error")
        raise ResolveError(f"tikwm: {msg}")

    d = payload["data"]
    video_id = str(d.get("id") or "")
    if not video_id:
        raise ResolveError("tikwm: response did not include a video id.")

    def _abs(u: Optional[str]) -> Optional[str]:
        if not u:
            return None
        if u.startswith("http"):
            return u
        return f"https://www.tikwm.com{u}"

    author = d.get("author") or {}

    return VideoInfo(
        video_id=video_id,
        title=d.get("title", ""),
        author_username=author.get("unique_id", ""),
        author_nickname=author.get("nickname", ""),
        cover_url=_abs(d.get("cover") or d.get("origin_cover")) or "",
        duration=int(d.get("duration") or 0),
        width=int(d.get("width") or 0),
        height=int(d.get("height") or 0),
        no_watermark_url=_abs(d.get("play")) or "",
        hd_url=_abs(d.get("hdplay")),
        watermark_url=_abs(d.get("wmplay")),
        music_url=_abs(d.get("music")),
        size_bytes=d.get("size"),
        hd_size_bytes=d.get("hd_size"),
        source_provider="tikwm",
    )


def _provider_tiklydown(url: str) -> VideoInfo:
    """Fallback provider: tiklydown public JSON API. No key required."""
    api_url = "https://api.tiklydown.eu.org/api/download"
    resp = requests.get(
        api_url,
        params={"url": url},
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()

    video = payload.get("video") or {}
    author = payload.get("author") or {}
    music = payload.get("music") or {}

    no_wm = video.get("noWatermark") or video.get("no_watermark")
    if not no_wm:
        raise ResolveError("tiklydown: no watermark-free URL in response.")

    video_id = str(payload.get("id") or payload.get("video_id") or "")
    if not video_id:
        # Derive a stable id from the URL itself so caching still works.
        match = re.search(r"/video/(\d+)", url)
        video_id = match.group(1) if match else str(abs(hash(url)))

    return VideoInfo(
        video_id=video_id,
        title=payload.get("title", ""),
        author_username=author.get("username", "") or author.get("uniqueId", ""),
        author_nickname=author.get("nickname", ""),
        cover_url=video.get("cover", "") or video.get("thumbnail", ""),
        duration=int(payload.get("duration") or 0),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        no_watermark_url=no_wm,
        hd_url=video.get("hd") or video.get("hdWatermark"),
        watermark_url=video.get("watermark"),
        music_url=music.get("playUrl") or music.get("play_url"),
        size_bytes=video.get("size"),
        hd_size_bytes=video.get("hdSize"),
        source_provider="tiklydown",
    )


PROVIDERS = (_provider_tikwm, _provider_tiklydown)


def resolve_video(raw_url: str) -> VideoInfo:
    """
    Try each provider in order until one succeeds. Raises ResolveError with a
    friendly message if every provider fails.
    """
    if not is_probably_tiktok_url(raw_url):
        raise ResolveError("That doesn't look like a TikTok link. Paste a full tiktok.com URL.")

    url = raw_url.strip()
    host = (urlparse(url).hostname or "").lower()

    # Short links need to be expanded first so providers receive a canonical
    # /@user/video/<id> URL, which is the format they understand best.
    if host in ("vm.tiktok.com", "vt.tiktok.com", "m.tiktok.com"):
        try:
            url = resolve_short_link(url)
        except ResolveError:
            pass  # Fall through — providers can sometimes handle short links directly.

    last_error: Optional[str] = None
    for provider in PROVIDERS:
        try:
            info = provider(url)
            if not info.no_watermark_url:
                raise ResolveError(f"{info.source_provider}: empty media URL.")
            _cache_set(info)
            log.info("Resolved %s via %s", info.video_id, info.source_provider)
            return info
        except ResolveError as exc:
            last_error = str(exc)
            log.warning("Provider failed: %s", exc)
            continue
        except requests.Timeout:
            last_error = f"{provider.__name__} timed out."
            log.warning(last_error)
            continue
        except requests.RequestException as exc:
            last_error = f"{provider.__name__} network error: {exc.__class__.__name__}."
            log.warning(last_error)
            continue
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            last_error = f"{provider.__name__} returned an unexpected response."
            log.warning("%s (%s)", last_error, exc)
            continue

    raise ResolveError(
        last_error
        or "All download providers are currently unavailable. Please try again shortly."
    )


# ==============================================================================
# Streaming download proxy
# ==============================================================================

def _is_allowed_media_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == suf or host.endswith("." + suf) for suf in ALLOWED_MEDIA_HOST_SUFFIXES)


def _pick_media_url(info: VideoInfo, media_type: str) -> tuple[str, str]:
    """Returns (url, filename) for the requested media type."""
    base_name = sanitize_filename(f"{info.author_username or 'tiktok'}_{info.video_id}")
    if media_type == "hd" and info.hd_url:
        return info.hd_url, f"{base_name}_hd.mp4"
    if media_type == "watermark" and info.watermark_url:
        return info.watermark_url, f"{base_name}_watermark.mp4"
    if media_type == "music" and info.music_url:
        return info.music_url, f"{base_name}.mp3"
    if media_type in ("sd", "hd", "watermark"):
        return info.no_watermark_url, f"{base_name}.mp4"
    raise ResolveError("Unsupported media type requested.")


# ==============================================================================
# Routes — API
# ==============================================================================

@app.post("/api/resolve")
@rate_limited("resolve", *RATE_LIMIT_RESOLVE)
def api_resolve():
    body = request.get_json(silent=True) or {}
    raw_url = (body.get("url") or "").strip()

    if not raw_url:
        return jsonify(ok=False, error="Paste a TikTok link first."), 400

    try:
        info = resolve_video(raw_url)
    except ResolveError as exc:
        return jsonify(ok=False, error=str(exc)), 422
    except Exception as exc:  # noqa: BLE001 — last line of defense, never crash
        log.exception("Unexpected error resolving %s", raw_url)
        return jsonify(ok=False, error="Something went wrong reading that video. Please try again."), 500

    return jsonify(ok=True, video=info.to_public_dict())


@app.get("/api/download")
@rate_limited("download", *RATE_LIMIT_DOWNLOAD)
def api_download():
    video_id = request.args.get("id", "")
    media_type = request.args.get("type", "sd")

    if not video_id:
        return jsonify(ok=False, error="Missing video id."), 400

    info = _cache_get(video_id)
    if info is None:
        return jsonify(
            ok=False,
            error="This preview has expired. Please paste the link again.",
        ), 410

    try:
        media_url, filename = _pick_media_url(info, media_type)
    except ResolveError as exc:
        return jsonify(ok=False, error=str(exc)), 400

    if not _is_allowed_media_host(media_url):
        log.error("Blocked proxy request to disallowed host: %s", media_url)
        return jsonify(ok=False, error="Refused to fetch from an untrusted host."), 400

    try:
        upstream = requests.get(
            media_url,
            headers={"User-Agent": USER_AGENT, "Referer": "https://www.tiktok.com/"},
            stream=True,
            timeout=STREAM_TIMEOUT,
        )
        upstream.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Upstream fetch failed for %s: %s", video_id, exc)
        return jsonify(ok=False, error="The video source is unavailable right now. Please try again."), 502

    content_length = upstream.headers.get("Content-Length")
    content_type = upstream.headers.get("Content-Type", "video/mp4")

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=256 * 1024):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Content-Type-Options": "nosniff",
    }
    if content_length:
        headers["Content-Length"] = content_length

    return Response(
        stream_with_context(generate()),
        headers=headers,
        content_type=content_type,
        status=200,
    )


@app.get("/api/health")
def api_health():
    return jsonify(ok=True, app=APP_NAME, cached_videos=len(_cache))


@app.errorhandler(404)
def not_found(_exc):
    return jsonify(ok=False, error="Not found."), 404


@app.errorhandler(500)
def server_error(_exc):
    log.exception("Unhandled server error")
    return jsonify(ok=False, error="Internal server error."), 500


# ==============================================================================
# Route — the app shell (HTML/CSS/JS all embedded, single string)
# ==============================================================================

@app.get("/")
def index():
    return Response(PAGE_HTML, content_type="text/html; charset=utf-8")


# ==============================================================================
# Frontend — embedded HTML/CSS/JS
# ==============================================================================

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<title>GlassTok — TikTok Downloader</title>
<meta name="color-scheme" content="light dark" />
<style>
:root{
  --bg-0:#eef1f7;
  --bg-1:#e2e7f2;
  --glass:rgba(255,255,255,0.55);
  --glass-strong:rgba(255,255,255,0.72);
  --glass-border:rgba(255,255,255,0.65);
  --ink-1:#12131a;
  --ink-2:#4a4d5c;
  --ink-3:#83869a;
  --accent-1:#7c6cf6;
  --accent-2:#ff6ec4;
  --accent-3:#3ddad7;
  --danger:#ff5470;
  --success:#2bd576;
  --shadow-lg:0 24px 60px -20px rgba(30,20,70,0.35);
  --shadow-sm:0 8px 24px -12px rgba(30,20,70,0.25);
  --radius-xl:28px;
  --radius-lg:20px;
  --radius-md:14px;
  --radius-sm:10px;
  color-scheme: light;
}
:root[data-theme="dark"]{
  --bg-0:#0c0d14;
  --bg-1:#15172a;
  --glass:rgba(28,30,46,0.55);
  --glass-strong:rgba(28,30,46,0.72);
  --glass-border:rgba(255,255,255,0.12);
  --ink-1:#f3f4fa;
  --ink-2:#c3c5da;
  --ink-3:#7f83a0;
  --shadow-lg:0 24px 70px -20px rgba(0,0,0,0.6);
  --shadow-sm:0 8px 24px -12px rgba(0,0,0,0.5);
  color-scheme: dark;
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
html,body{height:100%;}
body{
  margin:0;
  min-height:100vh;
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--ink-1);
  background:
    radial-gradient(1200px 800px at 10% -10%, color-mix(in srgb, var(--accent-1) 22%, transparent), transparent),
    radial-gradient(1000px 700px at 110% 10%, color-mix(in srgb, var(--accent-2) 18%, transparent), transparent),
    radial-gradient(900px 900px at 50% 120%, color-mix(in srgb, var(--accent-3) 16%, transparent), transparent),
    linear-gradient(180deg, var(--bg-0), var(--bg-1));
  background-attachment:fixed;
  overflow-x:hidden;
  transition:background-color .4s ease, color .4s ease;
}
/* ---------- floating glass particles (pure CSS, ambient background) ---------- */
.particles{position:fixed;inset:0;overflow:hidden;z-index:0;pointer-events:none;}
.particle{
  position:absolute;border-radius:50%;
  background:linear-gradient(135deg, rgba(255,255,255,0.5), rgba(255,255,255,0.05));
  backdrop-filter:blur(2px);
  animation:float 18s ease-in-out infinite;
  opacity:.5;
}
.particle:nth-child(1){width:120px;height:120px;left:8%;top:15%;animation-duration:22s;}
.particle:nth-child(2){width:70px;height:70px;left:80%;top:10%;animation-duration:16s;animation-delay:-4s;}
.particle:nth-child(3){width:160px;height:160px;left:65%;top:65%;animation-duration:26s;animation-delay:-9s;}
.particle:nth-child(4){width:50px;height:50px;left:20%;top:75%;animation-duration:14s;animation-delay:-2s;}
.particle:nth-child(5){width:90px;height:90px;left:45%;top:35%;animation-duration:20s;animation-delay:-11s;}
@keyframes float{
  0%,100%{transform:translate(0,0) rotate(0deg);}
  33%{transform:translate(24px,-30px) rotate(8deg);}
  66%{transform:translate(-18px,20px) rotate(-6deg);}
}
@media (prefers-reduced-motion: reduce){
  .particle{animation:none;}
  *{animation-duration:.001ms !important; transition-duration:.001ms !important;}
}

.shell{position:relative;z-index:1;max-width:640px;margin:0 auto;padding:28px 20px 60px;min-height:100vh;display:flex;flex-direction:column;}

/* ---------- top bar ---------- */
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:19px;letter-spacing:-.02em;}
.brand .mark{
  width:34px;height:34px;border-radius:11px;
  background:linear-gradient(135deg,var(--accent-1),var(--accent-2));
  display:flex;align-items:center;justify-content:center;
  box-shadow:var(--shadow-sm);
  flex-shrink:0;
}
.brand .mark svg{width:18px;height:18px;}
.tagline{color:var(--ink-3);font-size:12.5px;margin-top:1px;font-weight:500;}
.theme-toggle{
  width:44px;height:44px;border-radius:50%;
  background:var(--glass);
  border:1px solid var(--glass-border);
  backdrop-filter:blur(20px) saturate(160%);
  -webkit-backdrop-filter:blur(20px) saturate(160%);
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;box-shadow:var(--shadow-sm);
  transition:transform .2s ease, box-shadow .2s ease;
}
.theme-toggle:hover{transform:translateY(-2px) scale(1.04);}
.theme-toggle:active{transform:scale(.94);}
.theme-toggle svg{width:19px;height:19px;color:var(--ink-1);}

/* ---------- glass panel base ---------- */
.panel{
  background:var(--glass);
  border:1px solid var(--glass-border);
  backdrop-filter:blur(28px) saturate(180%);
  -webkit-backdrop-filter:blur(28px) saturate(180%);
  border-radius:var(--radius-xl);
  box-shadow:var(--shadow-lg);
  position:relative;
  overflow:hidden;
}
.panel::before{
  content:"";position:absolute;inset:0;border-radius:inherit;padding:1px;
  background:linear-gradient(135deg, rgba(255,255,255,.8), rgba(255,255,255,0) 40%);
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite:xor; mask-composite:exclude;
  pointer-events:none;
}

/* ---------- input card ---------- */
.input-card{padding:22px;margin-bottom:20px;}
.input-row{display:flex;gap:10px;}
.url-field{
  flex:1;
  display:flex;align-items:center;gap:10px;
  background:color-mix(in srgb, var(--glass-strong) 100%, transparent);
  border:1.5px solid var(--glass-border);
  border-radius:var(--radius-lg);
  padding:14px 16px;
  transition:border-color .2s ease, box-shadow .2s ease;
}
.url-field.drag-over{border-color:var(--accent-1); box-shadow:0 0 0 4px color-mix(in srgb, var(--accent-1) 18%, transparent);}
.url-field svg{width:18px;height:18px;color:var(--ink-3);flex-shrink:0;}
.url-field input{
  flex:1;border:none;outline:none;background:transparent;
  font-size:15.5px;color:var(--ink-1);font-family:inherit;min-width:0;
}
.url-field input::placeholder{color:var(--ink-3);}
.fetch-btn{
  border:none;cursor:pointer;flex-shrink:0;
  padding:0 22px;border-radius:var(--radius-lg);
  background:linear-gradient(135deg,var(--accent-1),var(--accent-2));
  color:#fff;font-weight:600;font-size:15px;
  box-shadow:0 10px 24px -8px color-mix(in srgb, var(--accent-1) 60%, transparent);
  transition:transform .15s ease, box-shadow .15s ease, opacity .15s ease;
  display:flex;align-items:center;gap:8px;
  min-height:52px;line-height:1;
}
.fetch-btn svg{width:16px;height:16px;flex-shrink:0;}
.fetch-btn:hover{transform:translateY(-2px);}
.fetch-btn:active{transform:translateY(0) scale(.97);}
.fetch-btn:disabled{opacity:.6;cursor:not-allowed;transform:none;}
.hint-row{display:flex;align-items:center;justify-content:space-between;margin-top:10px;padding:0 2px;}
.hint{font-size:12px;color:var(--ink-3);}
.kbd{
  display:inline-flex;align-items:center;justify-content:center;
  font-size:10.5px;padding:2px 6px;border-radius:6px;
  background:color-mix(in srgb, var(--ink-1) 8%, transparent);
  border:1px solid var(--glass-border);color:var(--ink-2);font-family:inherit;
}

/* ---------- toast ---------- */
.toast-stack{position:fixed;top:18px;left:50%;transform:translateX(-50%);z-index:50;display:flex;flex-direction:column;gap:8px;width:min(92vw,420px);}
.toast{
  padding:13px 16px;border-radius:var(--radius-md);
  background:var(--glass-strong);border:1px solid var(--glass-border);
  backdrop-filter:blur(24px) saturate(180%);-webkit-backdrop-filter:blur(24px) saturate(180%);
  box-shadow:var(--shadow-sm);font-size:13.5px;font-weight:500;
  display:flex;align-items:center;gap:10px;
  animation:toast-in .35s cubic-bezier(.2,.9,.3,1.3);
}
.toast.error{border-color:color-mix(in srgb, var(--danger) 50%, var(--glass-border));}
.toast.success{border-color:color-mix(in srgb, var(--success) 50%, var(--glass-border));}
.toast .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.toast.error .dot{background:var(--danger);}
.toast.success .dot{background:var(--success);}
.toast.info .dot{background:var(--accent-1);}
@keyframes toast-in{from{opacity:0;transform:translateY(-14px) scale(.96);}to{opacity:1;transform:translateY(0) scale(1);}}
@keyframes toast-out{to{opacity:0;transform:translateY(-10px) scale(.96);}}

/* ---------- states ---------- */
.state{display:none;}
.state.active{display:block;animation:fade-up .4s cubic-bezier(.2,.85,.3,1);}
@keyframes fade-up{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}

/* empty state */
.empty{padding:56px 28px;text-align:center;}
.empty svg{width:120px;height:120px;margin:0 auto 18px;display:block;}
.empty h3{margin:0 0 6px;font-size:17px;font-weight:700;}
.empty p{margin:0;color:var(--ink-3);font-size:13.5px;line-height:1.5;}

/* skeleton loader */
.skeleton-card{padding:20px;display:flex;gap:16px;}
.sk{border-radius:var(--radius-md);background:linear-gradient(100deg, color-mix(in srgb, var(--ink-1) 6%, transparent) 8%, color-mix(in srgb, var(--ink-1) 12%, transparent) 18%, color-mix(in srgb, var(--ink-1) 6%, transparent) 33%);background-size:200% 100%;animation:shimmer 1.4s ease infinite;}
@keyframes shimmer{0%{background-position:200% 0;}100%{background-position:-200% 0;}}
.sk-thumb{width:110px;height:150px;flex-shrink:0;}
.sk-lines{flex:1;display:flex;flex-direction:column;gap:10px;padding-top:6px;}
.sk-line{height:12px;border-radius:6px;}
.sk-line.w60{width:60%;}
.sk-line.w40{width:40%;}
.sk-line.w80{width:80%;}

/* preview card */
.preview-card{padding:20px;}
.preview-top{display:flex;gap:16px;}
.thumb-wrap{position:relative;width:110px;height:150px;flex-shrink:0;border-radius:var(--radius-md);overflow:hidden;box-shadow:var(--shadow-sm);background:color-mix(in srgb, var(--ink-1) 8%, transparent);}
.thumb-wrap img{width:100%;height:100%;object-fit:cover;display:block;}
.thumb-wrap .dur-badge{position:absolute;bottom:6px;right:6px;background:rgba(0,0,0,0.65);color:#fff;font-size:10.5px;font-weight:600;padding:2px 6px;border-radius:6px;backdrop-filter:blur(6px);}
.thumb-wrap .play-badge{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0);transition:background .2s ease;cursor:pointer;}
.thumb-wrap .play-badge:hover{background:rgba(0,0,0,0.25);}
.thumb-wrap .play-badge svg{width:34px;height:34px;color:#fff;filter:drop-shadow(0 4px 10px rgba(0,0,0,.4));opacity:0;transition:opacity .2s ease;}
.thumb-wrap .play-badge:hover svg{opacity:1;}
.meta{flex:1;min-width:0;display:flex;flex-direction:column;}
.meta .title{font-size:14.5px;font-weight:650;line-height:1.35;margin:0 0 6px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;}
.meta .author{font-size:12.5px;color:var(--ink-3);margin:0 0 10px;display:flex;align-items:center;gap:6px;}
.meta .author .avatar-dot{width:6px;height:6px;border-radius:50%;background:linear-gradient(135deg,var(--accent-1),var(--accent-3));}
.pill-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:auto;}
.pill{
  font-size:11px;font-weight:600;padding:4px 9px;border-radius:20px;
  background:color-mix(in srgb, var(--ink-1) 6%, transparent);
  border:1px solid var(--glass-border);color:var(--ink-2);
  display:inline-flex;align-items:center;gap:4px;
}
.pill.accent{background:color-mix(in srgb, var(--accent-1) 16%, transparent);color:color-mix(in srgb, var(--accent-1) 70%, var(--ink-1));border-color:transparent;}

.divider{height:1px;background:var(--glass-border);margin:18px 0;}

.action-row{display:flex;gap:10px;flex-wrap:wrap;}
.btn{
  border:none;cursor:pointer;font-family:inherit;font-weight:600;font-size:14px;
  border-radius:var(--radius-md);padding:13px 18px;display:flex;align-items:center;justify-content:center;gap:8px;
  transition:transform .15s ease, box-shadow .15s ease, opacity .15s ease;
}
.btn:active{transform:scale(.97);}
.btn-primary{
  flex:1;min-width:170px;color:#fff;
  background:linear-gradient(135deg,var(--accent-1),var(--accent-2));
  box-shadow:0 12px 26px -10px color-mix(in srgb, var(--accent-1) 60%, transparent);
}
.btn-primary:hover{transform:translateY(-2px);}
.btn-secondary{
  background:color-mix(in srgb, var(--ink-1) 6%, transparent);
  color:var(--ink-1);border:1px solid var(--glass-border);
}
.btn-secondary:hover{background:color-mix(in srgb, var(--ink-1) 10%, transparent);}
.btn-secondary svg,.btn-primary svg{width:16px;height:16px;flex-shrink:0;}
.btn:disabled{opacity:.55;cursor:not-allowed;transform:none;}

/* progress */
.progress-wrap{margin-top:16px;}
.progress-head{display:flex;justify-content:space-between;font-size:12.5px;color:var(--ink-2);margin-bottom:8px;font-weight:600;}
.progress-track{height:10px;border-radius:20px;background:color-mix(in srgb, var(--ink-1) 8%, transparent);overflow:hidden;position:relative;}
.progress-fill{
  height:100%;border-radius:20px;width:0%;
  background:linear-gradient(90deg,var(--accent-1),var(--accent-2),var(--accent-3));
  background-size:200% 100%;
  animation:progress-flow 2s linear infinite;
  transition:width .25s ease;
}
@keyframes progress-flow{0%{background-position:0% 0;}100%{background-position:200% 0;}}
.progress-sub{margin-top:8px;font-size:11.5px;color:var(--ink-3);display:flex;justify-content:space-between;}

/* success */
.success-box{padding:26px 20px;text-align:center;}
.success-icon{
  width:64px;height:64px;border-radius:50%;margin:0 auto 14px;
  background:linear-gradient(135deg,var(--success),#1fb968);
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 14px 30px -10px color-mix(in srgb, var(--success) 60%, transparent);
  animation:pop .5s cubic-bezier(.2,1.4,.4,1);
}
@keyframes pop{0%{transform:scale(0);}70%{transform:scale(1.12);}100%{transform:scale(1);}}
.success-icon svg{width:30px;height:30px;color:#fff;}
.success-box h3{margin:0 0 4px;font-size:16.5px;font-weight:700;}
.success-box p{margin:0 0 18px;color:var(--ink-3);font-size:13px;}

/* history */
.history-card{padding:18px 20px;margin-top:20px;}
.history-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}
.history-head h4{margin:0;font-size:13.5px;font-weight:700;color:var(--ink-2);}
.history-clear{font-size:12px;color:var(--ink-3);background:none;border:none;cursor:pointer;font-weight:600;}
.history-clear:hover{color:var(--danger);}
.history-list{display:flex;flex-direction:column;gap:8px;}
.history-item{
  display:flex;align-items:center;gap:12px;padding:9px 10px;border-radius:var(--radius-sm);
  cursor:pointer;transition:background .15s ease;
}
.history-item:hover{background:color-mix(in srgb, var(--ink-1) 5%, transparent);}
.history-item img{width:34px;height:46px;object-fit:cover;border-radius:6px;flex-shrink:0;background:color-mix(in srgb, var(--ink-1) 8%, transparent);}
.history-item .hi-meta{flex:1;min-width:0;}
.history-item .hi-title{font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.history-item .hi-sub{font-size:11px;color:var(--ink-3);}

footer{margin-top:auto;padding-top:36px;text-align:center;}
.footer-pill{
  display:inline-flex;align-items:center;gap:8px;
  padding:9px 16px;border-radius:20px;
  background:var(--glass);border:1px solid var(--glass-border);
  backdrop-filter:blur(20px) saturate(160%);-webkit-backdrop-filter:blur(20px) saturate(160%);
  font-size:11.5px;color:var(--ink-3);font-weight:500;box-shadow:var(--shadow-sm);
}
.footer-pill b{color:var(--ink-2);font-weight:700;}
.footer-pill .heart{color:var(--accent-2);}

@media (max-width:480px){
  .shell{padding:18px 14px 40px;}
  .input-row{flex-direction:column;}
  .fetch-btn{padding:14px;justify-content:center;}
  .preview-top{flex-direction:column;}
  .thumb-wrap{width:100%;height:220px;}
  .gate-card{padding:24px 18px 20px;}
}
:focus-visible{outline:2.5px solid var(--accent-1);outline-offset:2px;}

/* ---------- Follow gate ---------- */
.gate-overlay{
  position:fixed;inset:0;z-index:2000;
  display:flex;align-items:center;justify-content:center;
  padding:20px;
  background:
    radial-gradient(900px 700px at 20% 0%, color-mix(in srgb, var(--accent-1) 30%, transparent), transparent),
    radial-gradient(800px 700px at 100% 100%, color-mix(in srgb, var(--accent-2) 22%, transparent), transparent),
    color-mix(in srgb, var(--bg-0) 88%, black 12%);
  backdrop-filter:blur(6px);
  -webkit-backdrop-filter:blur(6px);
  animation:gate-fade-in .35s ease;
}
.gate-overlay.hidden{display:none;}
.gate-overlay.closing{animation:gate-fade-out .35s ease forwards;}
@keyframes gate-fade-in{from{opacity:0;}to{opacity:1;}}
@keyframes gate-fade-out{from{opacity:1;}to{opacity:0;visibility:hidden;}}
body.gate-locked{overflow:hidden;}

.gate-card{
  width:100%;max-width:380px;
  background:var(--glass-strong);
  border:1px solid var(--glass-border);
  backdrop-filter:blur(30px) saturate(180%);
  -webkit-backdrop-filter:blur(30px) saturate(180%);
  border-radius:var(--radius-xl);
  box-shadow:var(--shadow-lg);
  padding:28px 24px 24px;
  text-align:center;
  position:relative;
  animation:gate-pop .45s cubic-bezier(.2,.9,.3,1.2);
}
@keyframes gate-pop{from{opacity:0;transform:translateY(16px) scale(.96);}to{opacity:1;transform:translateY(0) scale(1);}}
.gate-icon{
  width:64px;height:64px;border-radius:20px;margin:0 auto 16px;
  background:linear-gradient(135deg,#1877f2,#42a5f5);
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 14px 30px -10px rgba(24,119,242,0.55);
}
.gate-icon svg{width:32px;height:32px;color:#fff;}
.gate-card h3{margin:0 0 8px;font-size:18px;font-weight:750;letter-spacing:-.01em;}
.gate-card p{margin:0 0 20px;color:var(--ink-2);font-size:13.5px;line-height:1.55;}
.gate-step{display:none;}
.gate-step.active{display:block;animation:fade-up .3s ease;}
.gate-follow-btn{
  display:flex;align-items:center;justify-content:center;gap:9px;
  width:100%;padding:14px 18px;border-radius:var(--radius-lg);
  background:linear-gradient(135deg,#1877f2,#3b8ef2);
  color:#fff;font-weight:700;font-size:15px;text-decoration:none;
  box-shadow:0 12px 26px -10px rgba(24,119,242,0.55);
  border:none;cursor:pointer;transition:transform .15s ease;
}
.gate-follow-btn:hover{transform:translateY(-2px);}
.gate-follow-btn:active{transform:translateY(0) scale(.97);}
.gate-follow-btn svg{width:18px;height:18px;flex-shrink:0;}
.gate-note{margin-top:14px;font-size:11.5px;color:var(--ink-3);}

.gate-check-row{
  display:flex;align-items:flex-start;gap:10px;text-align:left;
  background:color-mix(in srgb, var(--ink-1) 5%, transparent);
  border:1px solid var(--glass-border);border-radius:var(--radius-md);
  padding:12px 14px;margin-bottom:18px;cursor:pointer;
}
.gate-check-row input{margin-top:2px;width:17px;height:17px;flex-shrink:0;accent-color:var(--accent-1);cursor:pointer;}
.gate-check-row span{font-size:13px;color:var(--ink-1);line-height:1.4;}
.gate-continue-btn{
  width:100%;padding:14px 18px;border-radius:var(--radius-lg);
  border:none;font-weight:700;font-size:15px;cursor:pointer;
  background:linear-gradient(135deg,var(--accent-1),var(--accent-2));
  color:#fff;box-shadow:0 12px 26px -10px color-mix(in srgb, var(--accent-1) 60%, transparent);
  transition:transform .15s ease, opacity .15s ease;
  display:flex;align-items:center;justify-content:center;gap:8px;
}
.gate-continue-btn:hover:not(:disabled){transform:translateY(-2px);}
.gate-continue-btn:disabled{opacity:.5;cursor:not-allowed;}
.gate-fallback{
  display:none;margin-top:16px;font-size:11.5px;color:var(--ink-3);
  background:none;border:none;text-decoration:underline;cursor:pointer;
}
.gate-fallback.show{display:inline-block;}
</style>
</head>
<body>

<div class="gate-overlay" id="gateOverlay" role="dialog" aria-modal="true" aria-labelledby="gateTitle">
  <div class="gate-card">

    <div class="gate-step active" id="gateStepFollow">
      <div class="gate-icon">
        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M22 12a10 10 0 1 0-11.6 9.9v-7H7.9V12h2.5V9.8c0-2.5 1.5-3.9 3.8-3.9 1.1 0 2.2.2 2.2.2v2.5h-1.3c-1.2 0-1.6.8-1.6 1.6V12h2.8l-.4 2.9h-2.4v7A10 10 0 0 0 22 12z"/></svg>
      </div>
      <h3 id="gateTitle">One quick thing first</h3>
      <p>Give the GlassTok Facebook page a follow to unlock the downloader — it keeps this tool free and running.</p>
      <a class="gate-follow-btn" id="gateFollowBtn" href="https://www.facebook.com/dr.khertmd.em/" target="_blank" rel="noopener noreferrer">
        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M22 12a10 10 0 1 0-11.6 9.9v-7H7.9V12h2.5V9.8c0-2.5 1.5-3.9 3.8-3.9 1.1 0 2.2.2 2.2.2v2.5h-1.3c-1.2 0-1.6.8-1.6 1.6V12h2.8l-.4 2.9h-2.4v7A10 10 0 0 0 22 12z"/></svg>
        Follow on Facebook
      </a>
      <div class="gate-note">Opens in a new tab — come back here after.</div>
    </div>

    <div class="gate-step" id="gateStepConfirm">
      <div class="gate-icon" style="background:linear-gradient(135deg,var(--success),#1fb968);box-shadow:0 14px 30px -10px color-mix(in srgb, var(--success) 55%, transparent);">
        <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>
      </div>
      <h3>Thanks for stopping by!</h3>
      <p>Confirm you followed the page and you're in.</p>
      <label class="gate-check-row" for="gateCheckbox">
        <input type="checkbox" id="gateCheckbox" />
        <span>Yes, I followed the GlassTok Facebook page</span>
      </label>
      <button class="gate-continue-btn" id="gateContinueBtn" disabled>
        <span id="gateContinueLabel">Continue in 5s…</span>
      </button>
      <button class="gate-fallback" id="gateFallbackBtn">Having trouble? Continue anyway</button>
    </div>

  </div>
</div>

<div class="particles" aria-hidden="true">
  <div class="particle"></div><div class="particle"></div><div class="particle"></div>
  <div class="particle"></div><div class="particle"></div>
</div>

<div class="toast-stack" id="toastStack" aria-live="polite"></div>

<div class="shell">

  <div class="topbar">
    <div>
      <div class="brand">
        <span class="mark">
          <svg viewBox="0 0 24 24" fill="none"><path d="M16 3v9.5a4.5 4.5 0 1 1-3-4.24V3h3z" fill="#fff"/><path d="M16 3c.3 2.2 1.9 3.9 4 4.2V10c-1.5-.1-2.9-.6-4-1.5V3z" fill="#fff" opacity=".7"/></svg>
        </span>
        <div>
          GlassTok
          <div class="tagline">TikTok, without the watermark.</div>
        </div>
      </div>
    </div>
    <button class="theme-toggle" id="themeToggle" aria-label="Toggle dark mode" title="Toggle theme">
      <svg id="themeIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
    </button>
  </div>

  <!-- URL input -->
  <div class="panel input-card">
    <div class="input-row">
      <div class="url-field" id="urlField">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.07 0l2.83-2.83a5 5 0 0 0-7.07-7.07l-1.5 1.5"/><path d="M14 11a5 5 0 0 0-7.07 0L4.1 13.83a5 5 0 0 0 7.07 7.07l1.5-1.5"/></svg>
        <input id="urlInput" type="url" inputmode="url" autocomplete="off" spellcheck="false"
               placeholder="Paste a TikTok link… (Ctrl+V)" aria-label="TikTok video URL" />
      </div>
      <button class="fetch-btn" id="fetchBtn" aria-label="Fetch video">
        <span id="fetchBtnLabel">Fetch</span>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
      </button>
    </div>
    <div class="hint-row">
      <span class="hint">Works with tiktok.com, vm.tiktok.com &amp; vt.tiktok.com links</span>
      <span class="hint"><span class="kbd">Enter</span> to fetch</span>
    </div>
  </div>

  <!-- EMPTY STATE -->
  <div class="panel state active" id="stateEmpty">
    <div class="empty">
      <svg viewBox="0 0 200 160" fill="none">
        <ellipse cx="100" cy="140" rx="70" ry="10" fill="currentColor" opacity="0.06"/>
        <rect x="55" y="30" width="90" height="90" rx="18" fill="currentColor" opacity="0.07"/>
        <rect x="70" y="46" width="60" height="58" rx="10" fill="currentColor" opacity="0.10"/>
        <circle cx="100" cy="75" r="14" fill="none" stroke="currentColor" stroke-width="3" opacity="0.35"/>
        <path d="M96 75l6 4-6 4v-8z" fill="currentColor" opacity="0.45"/>
        <circle cx="146" cy="38" r="6" fill="#7c6cf6" opacity="0.5"/>
        <circle cx="54" cy="112" r="8" fill="#ff6ec4" opacity="0.4"/>
        <circle cx="150" cy="118" r="4" fill="#3ddad7" opacity="0.5"/>
      </svg>
      <h3>Paste a link to get started</h3>
      <p>Drop a TikTok URL above, or drag &amp; drop it anywhere on this card.<br>We'll grab the original, watermark-free file.</p>
    </div>
  </div>

  <!-- LOADING STATE -->
  <div class="panel state" id="stateLoading">
    <div class="skeleton-card">
      <div class="sk sk-thumb"></div>
      <div class="sk-lines">
        <div class="sk sk-line w80"></div>
        <div class="sk sk-line w60"></div>
        <div class="sk sk-line w40"></div>
        <div style="flex:1"></div>
        <div class="sk sk-line w60" style="height:34px;border-radius:10px;"></div>
      </div>
    </div>
  </div>

  <!-- PREVIEW STATE -->
  <div class="panel state" id="statePreview">
    <div class="preview-card">
      <div class="preview-top">
        <div class="thumb-wrap">
          <img id="pvThumb" alt="Video thumbnail" />
          <span class="dur-badge" id="pvDuration">0:00</span>
          <div class="play-badge" id="pvPlayToggle" title="Preview">
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
          </div>
        </div>
        <div class="meta">
          <p class="title" id="pvTitle">—</p>
          <p class="author"><span class="avatar-dot"></span><span id="pvAuthor">—</span></p>
          <div class="pill-row" id="pvPills"></div>
        </div>
      </div>

      <video id="pvVideo" controls playsinline style="display:none;width:100%;border-radius:14px;margin-top:16px;background:#000;"></video>

      <div class="divider"></div>

      <div class="action-row" id="actionRow">
        <button class="btn btn-primary" id="downloadBtn">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 19h16"/></svg>
          Download Video
        </button>
        <button class="btn btn-secondary" id="copyInfoBtn" title="Copy video info">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>
        </button>
        <button class="btn btn-secondary" id="anotherBtn" title="Download another video">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>
        </button>
      </div>

      <div class="progress-wrap" id="progressWrap" style="display:none;">
        <div class="progress-head">
          <span id="progressStatus">Downloading…</span>
          <span id="progressPct">0%</span>
        </div>
        <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
        <div class="progress-sub">
          <span id="progressSpeed">—</span>
          <span id="progressSize">—</span>
        </div>
      </div>
    </div>
  </div>

  <!-- SUCCESS STATE -->
  <div class="panel state" id="stateSuccess">
    <div class="success-box">
      <div class="success-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>
      </div>
      <h3>Download complete</h3>
      <p id="successSub">Saved to your downloads folder.</p>
      <div class="action-row">
        <button class="btn btn-secondary" id="successAgainBtn" style="flex:1;">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5"/></svg>
          Download another
        </button>
      </div>
    </div>
  </div>

  <!-- ERROR STATE -->
  <div class="panel state" id="stateError">
    <div class="empty">
      <svg viewBox="0 0 96 96" fill="none">
        <circle cx="48" cy="48" r="40" fill="none" stroke="#ff5470" stroke-width="4" opacity=".5"/>
        <path d="M48 30v22" stroke="#ff5470" stroke-width="5" stroke-linecap="round"/>
        <circle cx="48" cy="63" r="3.4" fill="#ff5470"/>
      </svg>
      <h3 id="errorTitle">Couldn't fetch that video</h3>
      <p id="errorSub">Something went wrong. Please check the link and try again.</p>
    </div>
  </div>

  <!-- HISTORY -->
  <div class="panel history-card state" id="historyCard">
    <div class="history-head">
      <h4>Recent this session</h4>
      <button class="history-clear" id="historyClear">Clear</button>
    </div>
    <div class="history-list" id="historyList"></div>
  </div>

  <footer>
    <span class="footer-pill">✦ Developed by <b>Dr. Khert Laguna Garde</b></span>
  </footer>

</div>

<script>
(() => {
  "use strict";

  // ---------------------------------------------------------------------
  // Follow gate — blocks use of the app until the person has (a) opened
  // the Facebook page and (b) confirmed + waited out a short timer.
  // We can't cryptographically verify a "follow" without Facebook OAuth
  // (which this project intentionally avoids), so this is an honest,
  // honor-system gate: real friction, not fake verification. It never
  // permanently traps anyone — a de-emphasized fallback always appears.
  // ---------------------------------------------------------------------
  (function initFollowGate() {
    const GATE_KEY = "glasstok_fb_gate_passed";
    const CONTINUE_DELAY_MS = 5000;
    const FALLBACK_DELAY_MS = 20000;

    const overlay = document.getElementById("gateOverlay");
    const stepFollow = document.getElementById("gateStepFollow");
    const stepConfirm = document.getElementById("gateStepConfirm");
    const followBtn = document.getElementById("gateFollowBtn");
    const checkbox = document.getElementById("gateCheckbox");
    const continueBtn = document.getElementById("gateContinueBtn");
    const continueLabel = document.getElementById("gateContinueLabel");
    const fallbackBtn = document.getElementById("gateFallbackBtn");

    let alreadyPassed = false;
    try { alreadyPassed = localStorage.getItem(GATE_KEY) === "true"; } catch (e) {}

    if (alreadyPassed) {
      overlay.classList.add("hidden");
      return;
    }

    document.body.classList.add("gate-locked");
    let timerDone = false;

    function evaluateContinueState() {
      continueBtn.disabled = !(timerDone && checkbox.checked);
    }

    function closeGate() {
      try { localStorage.setItem(GATE_KEY, "true"); } catch (e) {}
      document.body.classList.remove("gate-locked");
      overlay.classList.add("closing");
      setTimeout(() => overlay.classList.add("hidden"), 350);
      const urlInputEl = document.getElementById("urlInput");
      if (urlInputEl) setTimeout(() => urlInputEl.focus(), 400);
    }

    followBtn.addEventListener("click", () => {
      // The <a> tag's own href+target already opens the Facebook page in a
      // new tab natively (more reliable than window.open across mobile
      // browsers, since it's a direct user-gesture navigation rather than
      // a script-initiated popup that some browsers block).
      stepFollow.classList.remove("active");
      stepConfirm.classList.add("active");

      let secondsLeft = Math.ceil(CONTINUE_DELAY_MS / 1000);
      continueLabel.textContent = `Continue in ${secondsLeft}s…`;
      const tick = setInterval(() => {
        secondsLeft -= 1;
        if (secondsLeft <= 0) {
          clearInterval(tick);
          timerDone = true;
          continueLabel.textContent = "Continue to GlassTok";
          evaluateContinueState();
        } else {
          continueLabel.textContent = `Continue in ${secondsLeft}s…`;
        }
      }, 1000);

      // Absolute safety net: never let someone be stuck here indefinitely,
      // even if the checkbox/timer UI misbehaves on some device.
      setTimeout(() => fallbackBtn.classList.add("show"), FALLBACK_DELAY_MS);
    });

    checkbox.addEventListener("change", evaluateContinueState);
    continueBtn.addEventListener("click", () => {
      if (!continueBtn.disabled) closeGate();
    });
    fallbackBtn.addEventListener("click", closeGate);
  })();

  // ---------------------------------------------------------------------
  // Theme
  // ---------------------------------------------------------------------
  const root = document.documentElement;
  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");
  const SUN = '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>';
  const MOON = '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>';

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    themeIcon.innerHTML = theme === "dark" ? MOON : SUN;
    try { localStorage.setItem("glasstok_theme", theme); } catch (e) {}
  }
  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem("glasstok_theme"); } catch (e) {}
    if (saved) { applyTheme(saved); return; }
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(prefersDark ? "dark" : "light");
  }
  themeToggle.addEventListener("click", () => {
    applyTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
      let saved = null;
      try { saved = localStorage.getItem("glasstok_theme"); } catch (err) {}
      if (!saved) applyTheme(e.matches ? "dark" : "light");
    });
  }
  initTheme();

  // ---------------------------------------------------------------------
  // Toasts
  // ---------------------------------------------------------------------
  const toastStack = document.getElementById("toastStack");
  function toast(message, type = "info", duration = 3600) {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.innerHTML = `<span class="dot"></span><span>${escapeHtml(message)}</span>`;
    toastStack.appendChild(el);
    setTimeout(() => {
      el.style.animation = "toast-out .25s ease forwards";
      setTimeout(() => el.remove(), 250);
    }, duration);
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    return div.innerHTML;
  }

  // ---------------------------------------------------------------------
  // State machine
  // ---------------------------------------------------------------------
  const states = {
    empty: document.getElementById("stateEmpty"),
    loading: document.getElementById("stateLoading"),
    preview: document.getElementById("statePreview"),
    success: document.getElementById("stateSuccess"),
    error: document.getElementById("stateError"),
  };
  function showState(name) {
    Object.values(states).forEach((el) => el.classList.remove("active"));
    states[name].classList.add("active");
  }

  // ---------------------------------------------------------------------
  // Elements
  // ---------------------------------------------------------------------
  const urlField = document.getElementById("urlField");
  const urlInput = document.getElementById("urlInput");
  const fetchBtn = document.getElementById("fetchBtn");
  const fetchBtnLabel = document.getElementById("fetchBtnLabel");

  const pvThumb = document.getElementById("pvThumb");
  const pvDuration = document.getElementById("pvDuration");
  const pvTitle = document.getElementById("pvTitle");
  const pvAuthor = document.getElementById("pvAuthor");
  const pvPills = document.getElementById("pvPills");
  const pvVideo = document.getElementById("pvVideo");
  const pvPlayToggle = document.getElementById("pvPlayToggle");

  const downloadBtn = document.getElementById("downloadBtn");
  const copyInfoBtn = document.getElementById("copyInfoBtn");
  const anotherBtn = document.getElementById("anotherBtn");
  const successAgainBtn = document.getElementById("successAgainBtn");

  const progressWrap = document.getElementById("progressWrap");
  const progressFill = document.getElementById("progressFill");
  const progressPct = document.getElementById("progressPct");
  const progressStatus = document.getElementById("progressStatus");
  const progressSpeed = document.getElementById("progressSpeed");
  const progressSize = document.getElementById("progressSize");

  const errorTitle = document.getElementById("errorTitle");
  const errorSub = document.getElementById("errorSub");
  const successSub = document.getElementById("successSub");

  const historyCard = document.getElementById("historyCard");
  const historyList = document.getElementById("historyList");
  const historyClear = document.getElementById("historyClear");

  let currentVideo = null;   // last resolved video payload from /api/resolve
  let history = [];          // session-only history (never persisted)

  // ---------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------
  function fmtDuration(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
  }
  function fmtBytes(bytes) {
    if (!bytes || bytes <= 0) return null;
    const units = ["B", "KB", "MB", "GB"];
    let i = 0, val = bytes;
    while (val >= 1024 && i < units.length - 1) { val /= 1024; i++; }
    return `${val.toFixed(val >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
  }
  function looksLikeTikTokUrl(str) {
    return /tiktok\.com\//i.test(str || "");
  }

  // ---------------------------------------------------------------------
  // Resolve a URL
  // ---------------------------------------------------------------------
  async function resolveUrl(url, isRetry = false) {
    if (!url || !url.trim()) { toast("Paste a TikTok link first.", "error"); return; }
    if (!looksLikeTikTokUrl(url)) {
      toast("That doesn't look like a TikTok link.", "error");
      return;
    }

    showState("loading");
    fetchBtn.disabled = true;
    fetchBtnLabel.textContent = isRetry ? "Waking up server…" : "Fetching…";

    try {
      const res = await fetch("/api/resolve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim() }),
      });

      // On a free-tier cold start, the platform's own proxy can briefly
      // return a plain-text "Not Found" (not JSON) before the app is fully
      // awake. Parse defensively so that hiccup doesn't crash as an
      // "Unexpected token" error — instead, retry once automatically.
      const raw = await res.text();
      let data;
      try {
        data = JSON.parse(raw);
      } catch (parseErr) {
        if (!isRetry) {
          await new Promise((r) => setTimeout(r, 1800));
          return resolveUrl(url, true);
        }
        throw new Error("The server is still waking up — please tap Fetch again.");
      }

      if (!res.ok || !data.ok) {
        throw new Error(data.error || "Couldn't fetch that video.");
      }

      currentVideo = data.video;
      renderPreview(currentVideo);
      addToHistory(currentVideo);
      showState("preview");
      toast("Video found — ready to download.", "success", 2200);
    } catch (err) {
      errorTitle.textContent = "Couldn't fetch that video";
      errorSub.textContent = err.message || "Something went wrong. Please check the link and try again.";
      showState("error");
      toast(err.message || "Something went wrong.", "error");
    } finally {
      fetchBtn.disabled = false;
      fetchBtnLabel.textContent = "Fetch";
    }
  }

  function renderPreview(v) {
    pvThumb.src = v.cover || "";
    pvThumb.alt = v.title || "TikTok video thumbnail";
    pvDuration.textContent = fmtDuration(v.duration);
    pvTitle.textContent = v.title || "Untitled TikTok video";
    pvAuthor.textContent = v.authorNickname ? `@${v.author || v.authorNickname}` : "Unknown creator";

    pvVideo.style.display = "none";
    pvVideo.pause?.();
    pvVideo.removeAttribute("src");

    const pills = [];
    if (v.resolution && v.resolution !== "Unknown") pills.push(`<span class="pill">${escapeHtml(v.resolution)}</span>`);
    if (v.hasHd) pills.push(`<span class="pill accent">HD</span>`);
    const size = fmtBytes(v.hdSizeBytes || v.sizeBytes);
    if (size) pills.push(`<span class="pill">~${size}</span>`);
    pills.push(`<span class="pill">No watermark</span>`);
    pvPills.innerHTML = pills.join("");

    progressWrap.style.display = "none";
    progressFill.style.width = "0%";
  }

  pvPlayToggle.addEventListener("click", () => {
    if (!currentVideo) return;
    if (pvVideo.style.display === "none") {
      pvVideo.src = currentVideo.downloadUrls.sd;
      pvVideo.style.display = "block";
      pvVideo.play().catch(() => {});
    } else {
      pvVideo.style.display = "none";
      pvVideo.pause();
    }
  });

  // ---------------------------------------------------------------------
  // Download with progress
  // ---------------------------------------------------------------------
  async function downloadCurrent() {
    if (!currentVideo) return;
    const url = (currentVideo.hasHd && currentVideo.downloadUrls.hd) || currentVideo.downloadUrls.sd;

    downloadBtn.disabled = true;
    progressWrap.style.display = "block";
    progressStatus.textContent = "Connecting…";
    progressPct.textContent = "0%";
    progressFill.style.width = "0%";
    progressSpeed.textContent = "—";
    progressSize.textContent = "—";

    const startTime = performance.now();
    let lastLoaded = 0, lastTime = startTime;

    try {
      const res = await fetch(url);
      if (!res.ok) {
        let msg = "Download failed.";
        try { const j = await res.json(); msg = j.error || msg; } catch (e) {}
        throw new Error(msg);
      }

      const total = parseInt(res.headers.get("Content-Length") || "0", 10);
      const disposition = res.headers.get("Content-Disposition") || "";
      const match = /filename="([^"]+)"/.exec(disposition);
      const filename = match ? match[1] : `${(currentVideo.author || "tiktok")}_${currentVideo.id}.mp4`;

      progressStatus.textContent = "Downloading…";

      const reader = res.body.getReader();
      const chunks = [];
      let loaded = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;

        if (total) {
          const pct = Math.min(100, Math.round((loaded / total) * 100));
          progressFill.style.width = pct + "%";
          progressPct.textContent = pct + "%";
          progressSize.textContent = `${fmtBytes(loaded)} / ${fmtBytes(total)}`;
        } else {
          progressFill.style.width = "100%";
          progressPct.textContent = fmtBytes(loaded);
          progressSize.textContent = fmtBytes(loaded);
        }

        const now = performance.now();
        if (now - lastTime > 400) {
          const speed = ((loaded - lastLoaded) / ((now - lastTime) / 1000));
          progressSpeed.textContent = `${fmtBytes(speed)}/s`;
          lastLoaded = loaded;
          lastTime = now;
        }
      }

      progressFill.style.width = "100%";
      progressPct.textContent = "100%";
      progressStatus.textContent = "Finishing…";

      const blob = new Blob(chunks);
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 4000);

      successSub.textContent = `Saved as ${filename}`;
      showState("success");
      toast("Download complete!", "success");
    } catch (err) {
      toast(err.message || "Download failed.", "error");
      progressStatus.textContent = "Failed";
    } finally {
      downloadBtn.disabled = false;
    }
  }

  downloadBtn.addEventListener("click", downloadCurrent);

  // ---------------------------------------------------------------------
  // Copy info / reset / history
  // ---------------------------------------------------------------------
  copyInfoBtn.addEventListener("click", async () => {
    if (!currentVideo) return;
    const info = `${currentVideo.title}\nCreator: @${currentVideo.author}\nDuration: ${fmtDuration(currentVideo.duration)}\nResolution: ${currentVideo.resolution}`;
    try {
      await navigator.clipboard.writeText(info);
      toast("Video info copied.", "success", 2000);
    } catch (e) {
      toast("Couldn't copy — clipboard access blocked.", "error");
    }
  });

  function resetToEmpty() {
    currentVideo = null;
    urlInput.value = "";
    progressWrap.style.display = "none";
    showState("empty");
    urlInput.focus();
  }
  anotherBtn.addEventListener("click", resetToEmpty);
  successAgainBtn.addEventListener("click", resetToEmpty);

  function addToHistory(v) {
    history = history.filter((h) => h.id !== v.id);
    history.unshift(v);
    history = history.slice(0, 6);
    renderHistory();
  }
  function renderHistory() {
    if (!history.length) { historyCard.classList.remove("active"); return; }
    historyCard.classList.add("active");
    historyList.innerHTML = history.map((v) => `
      <div class="history-item" data-id="${escapeHtml(v.id)}">
        <img src="${escapeHtml(v.cover || "")}" alt="" loading="lazy" />
        <div class="hi-meta">
          <div class="hi-title">${escapeHtml(v.title || "Untitled TikTok video")}</div>
          <div class="hi-sub">@${escapeHtml(v.author || "unknown")} · ${fmtDuration(v.duration)}</div>
        </div>
      </div>
    `).join("");
    historyList.querySelectorAll(".history-item").forEach((el) => {
      el.addEventListener("click", () => {
        const v = history.find((h) => h.id === el.dataset.id);
        if (!v) return;
        currentVideo = v;
        renderPreview(v);
        showState("preview");
      });
    });
  }
  historyClear.addEventListener("click", () => {
    history = [];
    renderHistory();
    toast("History cleared.", "info", 1800);
  });

  // ---------------------------------------------------------------------
  // Input wiring: fetch button, Enter key, paste detection, drag & drop
  // ---------------------------------------------------------------------
  fetchBtn.addEventListener("click", () => resolveUrl(urlInput.value));
  urlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") resolveUrl(urlInput.value);
  });
  urlInput.addEventListener("paste", () => {
    setTimeout(() => {
      if (looksLikeTikTokUrl(urlInput.value)) resolveUrl(urlInput.value);
    }, 30);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    urlField.addEventListener(evt, (e) => { e.preventDefault(); urlField.classList.add("drag-over"); });
  });
  ["dragleave", "drop"].forEach((evt) => {
    urlField.addEventListener(evt, (e) => { e.preventDefault(); urlField.classList.remove("drag-over"); });
  });
  urlField.addEventListener("drop", (e) => {
    const text = e.dataTransfer.getData("text/plain") || e.dataTransfer.getData("text/uri-list");
    if (text) { urlInput.value = text; resolveUrl(text); }
  });

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v" && document.activeElement !== urlInput) {
      urlInput.focus();
    }
  });

  urlInput.focus();
})();
</script>
</body>
</html>
"""


# ==============================================================================
# Server bootstrap
# ==============================================================================

def find_free_port() -> int:
    for port in PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST, port))
                return port
            except OSError:
                continue
    # Fall back to an OS-assigned ephemeral port if the whole range is busy.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def open_browser_when_ready(url: str) -> None:
    def _wait_and_open():
        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                with socket.create_connection((HOST, int(url.rsplit(":", 1)[1])), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.15)
        webbrowser.open(url)

    threading.Thread(target=_wait_and_open, daemon=True).start()


def main() -> None:
    port = find_free_port()
    url = f"http://{HOST}:{port}"

    print("=" * 60)
    print(f"  {APP_NAME} — {APP_TAGLINE}")
    print(f"  Running at {url}")
    print("  Press CTRL+C to stop.")
    print("=" * 60)

    open_browser_when_ready(url)

    try:
        app.run(host=HOST, port=port, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
