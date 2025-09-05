#!/usr/bin/env python3
import io
import json
import logging
import os
import re
import shutil
import sys
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import requests
from dotenv import load_dotenv
from mutagen.id3 import APIC, ID3
from mutagen.id3 import error as ID3Error
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from PIL import Image, ImageTk, UnidentifiedImageError
from tqdm import tqdm

# ============== CONFIG & LOGGING ==============


def get_executable_directory() -> str:
    """
    Returns the folder where the running executable or script is located.
    """
    if getattr(sys, "frozen", False):
        # Running as compiled exe (onefile or folder)
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    else:
        # Running as normal script
        return os.path.dirname(os.path.abspath(__file__))


folder = get_executable_directory()
CONFIG_FILE = os.path.join(folder, "conf/config.json")
CACHE_FILE = os.path.join(folder, "data/artwork_cache.json")
DOTENV_FILE = os.path.join(folder, ".env")

DEFAULT_CONFIG = {
    "ENABLED_PROVIDERS": [
        "juno",
        "discogs",
        "itunes",
        "spotify",
        "soundcloud",
        "google",
    ],
    "MIN_RESOLUTION": 500,
    "REQUIRE_SQUARE": True,
    "SKIP_PROCESSED": True,
    "PROCESSED_DIR": "Processed",
    "UNPROCESSED_DIR": "Unprocessed",
    "FILE_ACTION": "copy",  # "copy" or "move"
    "GOOGLE_API_KEY": "",
    "GOOGLE_CX_ID": "",
    "GOOGLE_MAX_RESULTS": 30,
    "GOOGLE_MAX_WORKERS": 3,
    "DISCOGS_TOKEN": "",
    "SPOTIFY_CLIENT_ID": "",
    "SPOTIFY_CLIENT_SECRET": "",
    "SOUNDCLOUD_CLIENT_ID": "",
    "LOG_LEVEL": "INFO",
    "USE_CACHE": True,
}

# A polite browser-like user agent to avoid 403s from some CDNs
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
)


def load_config() -> dict:
    load_dotenv(dotenv_path=DOTENV_FILE)

    cfg = DEFAULT_CONFIG.copy()

    # If config.json exists, merge it in
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as fobj:
            user = json.load(fobj)
        cfg.update(user or {})

    # Override with environment variables if set
    for key in cfg.keys():
        env_val = os.getenv(key)
        if env_val is not None:
            # Cast certain values
            if isinstance(env_val, str) and env_val.lower() in ("true", "false"):
                cfg[key] = env_val.lower() == "true"
            elif isinstance(env_val, str) and env_val.isdigit():
                cfg[key] = int(env_val)
            else:
                cfg[key] = env_val

    # Validate FILE_ACTION
    action = str(cfg.get("FILE_ACTION", "copy")).lower()
    if action not in ("copy", "move"):
        logging.warning("Invalid FILE_ACTION '%s'; falling back to 'copy'", action)
        cfg["FILE_ACTION"] = "copy"
    else:
        cfg["FILE_ACTION"] = action

    # Normalize LOG_LEVEL name to expected key used below
    if "LOG_LEVEL" not in cfg and "LOG_LEVEL" in cfg:
        cfg["LOG_LEVEL"] = cfg.get("LOG_LEVEL", "INFO")

    return cfg


config = load_config()

logging.basicConfig(
    level=getattr(logging, config.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger("artwork-embedder")

# Cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as fobj:
        cache = json.load(fobj)
else:
    cache = {}


def save_cache():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as fobj:
        json.dump(cache, fobj, indent=2)


# ============== HELPERS ==============


def clean_query(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\[.*?\]|\(.*?\)", "", name)
    name = re.sub(
        r"\b(\d{3}kbps|hq|official video|audio|video|lyrics)\b", "", name, flags=re.I
    )
    return re.sub(r"\s+", " ", name).strip()


def fetch_image(url: str) -> Optional[Tuple[str, bytes, Tuple[int, int], str]]:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        w, h = img.size
        fmt = (img.format or "JPEG").upper()
        mime = "image/jpeg" if fmt == "JPEG" else "image/%s" % fmt.lower()
        # verify content is image (verify may close the image so we only used it to confirm)
        img.verify()
        return (url, r.content, (w, h), mime)
    except Exception as exc:
        logger.debug("🔍 fetch_image failed for %s: %s", url, exc)
        return None


def filter_and_order_candidates(
    results: List[Tuple[str, bytes, Tuple[int, int], str]],
) -> List[Tuple[str, bytes, Tuple[int, int], str]]:
    min_res = int(config["MIN_RESOLUTION"])
    require_square = bool(config["REQUIRE_SQUARE"])
    valid = []
    for url, content, (w, h), mime in results:
        if w >= min_res and h >= min_res:
            if not require_square or abs(w - h) < 50:
                valid.append((url, content, (w, h), mime))
    valid.sort(
        key=lambda x: (min(x[2][0], x[2][1]), max(x[2][0], x[2][1])), reverse=True
    )
    return valid


def preview_with_tkinter(image_bytes: bytes) -> bool:
    """
    Display image with Tkinter and ask user to choose yes/no.
    Shows original resolution under the preview.
    Grabs focus until input is given, then returns focus to console.
    Returns True if approved, False otherwise.
    """
    decision = {"value": None}

    def approve(event=None):
        decision["value"] = True
        try:
            root.destroy()
        except Exception:
            pass

    def reject(event=None):
        decision["value"] = False
        try:
            root.destroy()
        except Exception:
            pass

    # Create root window
    root = tk.Tk()
    root.title("Artwork Preview")

    # Bring window to front and grab focus
    try:
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        root.grab_set()  # Force all events to this window
    except Exception:
        pass

    # Load PIL image and capture original resolution
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        return False

    orig_w, orig_h = pil_img.size

    # Resize to max 800x800 for preview while preserving aspect ratio
    max_size = (800, 800)
    pil_preview = pil_img.copy()
    pil_preview.thumbnail(max_size, Image.LANCZOS)

    tk_img = ImageTk.PhotoImage(pil_preview)

    panel = tk.Label(root, image=tk_img)
    panel.image = tk_img  # keep reference
    panel.pack(padx=10, pady=(10, 4))

    # Show resolution label
    res_text = f"Resolution: {orig_w}×{orig_h}"
    res_label = tk.Label(root, text=res_text, font=("TkDefaultFont", 10))
    res_label.pack(pady=(0, 8))

    # Buttons frame
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=6)

    yes_btn = tk.Button(
        btn_frame, text="✅ Use this artwork", width=18, command=approve
    )
    yes_btn.pack(side="left", padx=6)

    no_btn = tk.Button(btn_frame, text="❌ Skip", width=12, command=reject)
    no_btn.pack(side="right", padx=6)

    # Keyboard shortcuts
    root.bind("<Return>", approve)
    root.bind("<KP_Enter>", approve)
    root.bind("<Escape>", reject)

    # Center the window
    try:
        root.update_idletasks()
        w = root.winfo_width()
        h = root.winfo_height()
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws // 2) - (w // 2)
        y = (hs // 2) - (h // 2)
        root.geometry(f"+{x}+{y}")
    except Exception:
        pass

    # Wait until the popup is closed (blocking)
    root.mainloop()

    # Attempt to give focus back to console (platform-specific)
    try:
        root.quit()
        root.destroy()
    except Exception:
        pass

    return decision["value"] is True


def select_image_from_candidates(
    urls: List[str],
) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
    if not urls:
        return None, None, None
    results: List[Tuple[str, bytes, Tuple[int, int], str]] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fetch_image, u): u for u in urls}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    candidates = filter_and_order_candidates(results)
    for url, content, (w, h), mime in candidates:
        logger.info("🖼️ Candidate: %s (%dx%d)", url, w, h)
        try:
            # Use Tkinter preview window with Yes / No buttons
            approved = preview_with_tkinter(content)
            if approved:
                logger.info("✅ User approved artwork: %s", url)
                return url, content, mime
            else:
                logger.info("⏭️ User skipped artwork: %s", url)
        except Exception as exc:
            logger.warning("❌ Could not preview image %s: %s", url, exc)

        logger.info("⏭️ Skipping candidate: %s", url)
    return None, None, None


# ============== PROVIDERS ==============


import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def provider_google(query: str) -> List[str]:
    """
    Threaded Google CSE provider:
    - Reads GOOGLE_MAX_RESULTS and GOOGLE_MAX_WORKERS from config
    - Tries sizes in descending order and pages up to what's necessary
    - Fetches pages in parallel (bounded by GOOGLE_MAX_WORKERS)
    - Uses image metadata (width/height) when available to stop early
    """
    key = config.get("GOOGLE_API_KEY")
    cx = config.get("GOOGLE_CX_ID")
    if not key or not cx:
        logger.debug(
            "🔍 Google provider skipped (missing GOOGLE_API_KEY/GOOGLE_CX_ID)."
        )
        return []

    max_results_cfg = int(config.get("GOOGLE_MAX_RESULTS", 30))
    max_workers = int(config.get("GOOGLE_MAX_WORKERS", 3))
    max_results = min(max(1, max_results_cfg), 100)
    min_res = int(
        config.get("MIN_RESOLUTION", DEFAULT_CONFIG.get("MIN_RESOLUTION", 500))
    )

    sizes = ["XLARGE", "LARGE"]
    page_size = 10
    gathered: List[str] = []
    seen = set()
    valid_count = 0  # number of metadata-validated items collected

    def fetch_page(img_size: str, start_index: int):
        """
        Fetch one page of results for img_size at start_index.
        Returns list of items (raw JSON) or empty list on error.
        Build the service inside the worker to avoid sharing client across threads.
        """
        try:
            from googleapiclient.discovery import build

            service = build(
                "customsearch", "v1", developerKey=key
            )  # pylint: disable=no-member
            res = (
                service.cse()
                .list(
                    q=query,
                    cx=cx,
                    searchType="image",
                    imgSize=img_size,
                    num=page_size,
                    start=start_index,
                )
                .execute()
            )
            return res.get("items", []) or []
        except Exception as exc:
            logger.warning(
                "🔍 Google page fetch error (size=%s start=%d): %s",
                img_size,
                start_index,
                exc,
            )
            return []

    # Outer loop sizes — but we will submit page fetch jobs in batches per size
    try:
        for img_size in sizes:
            if valid_count >= max_results:
                break

            # estimate how many pages we might need (cap at 10 pages per size)
            pages_needed = (max_results + page_size - 1) // page_size
            pages_needed = min(
                pages_needed, 10
            )  # don't try more than ~10 pages per size

            # Build list of start indices for this size: 1, 11, 21, ...
            starts = [1 + (i * page_size) for i in range(pages_needed)]

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(fetch_page, img_size, s): s for s in starts}

                for fut in as_completed(futures):
                    items = fut.result()
                    if not items:
                        continue

                    # iterate items in returned order
                    for it in items:
                        link = it.get("link")
                        if not link or link in seen:
                            continue

                        seen.add(link)
                        gathered.append(link)

                        # use Google image metadata when present
                        img_meta = it.get("image", {}) or {}
                        try:
                            w = (
                                int(img_meta.get("width"))
                                if img_meta.get("width")
                                else None
                            )
                            h = (
                                int(img_meta.get("height"))
                                if img_meta.get("height")
                                else None
                            )
                        except Exception:
                            w = h = None

                        if w and h and w >= min_res and h >= min_res:
                            valid_count += 1

                        # stop conditions:
                        if valid_count >= max_results:
                            break
                        # safety: if we have gathered plenty of items (even without metadata), stop to avoid huge results
                        if len(gathered) >= max_results * 2:
                            break

                    logger.debug(
                        "🔍 Google(size=%s) batch returned %d items (gathered=%d, valid=%d)",
                        img_size,
                        len(items),
                        len(gathered),
                        valid_count,
                    )

                    if valid_count >= max_results or len(gathered) >= max_results * 2:
                        break

                    # polite tiny sleep to avoid bursts when many futures complete quickly
                    time.sleep(0.1)

            # after finishing this size, check stop condition
            if valid_count >= max_results:
                break

            # small gap before trying next size
            time.sleep(0.2)

        # Return up to max_results deduplicated urls
        return gathered[:max_results]

    except Exception as exc:
        logger.warning("🔍 ❌ Google provider error for query %s: %s", query, exc)
        return []


def provider_itunes(query: str) -> List[str]:
    try:
        url = "https://itunes.apple.com/search"
        r = requests.get(
            url, params={"term": query, "entity": "song", "limit": 3}, timeout=8
        )
        r.raise_for_status()
        data = r.json()
        urls: List[str] = []

        min_res = int(config.get("MIN_RESOLUTION", DEFAULT_CONFIG["MIN_RESOLUTION"]))
        size_str = f"{min_res}x{min_res}bb"

        for item in data.get("results", []):
            art = item.get("artworkUrl100")
            if art:
                urls.append(art.replace("100x100bb", size_str))

        logger.debug("🍎 iTunes returned %d results for query: %s", len(urls), query)
        return urls
    except Exception as exc:
        logger.warning("🍎 ❌ iTunes provider error for query %s: %s", query, exc)
        return []


def provider_discogs(query: str) -> List[str]:
    token = config.get("DISCOGS_TOKEN")
    if not token:
        logger.debug("💿 Discogs provider skipped (missing DISCOGS_TOKEN).")
        return []
    try:
        url = "https://api.discogs.com/database/search"
        r = requests.get(
            url, params={"q": query, "token": token, "per_page": 10}, timeout=10
        )
        r.raise_for_status()
        data = r.json()
        urls = [
            it["cover_image"] for it in data.get("results", []) if it.get("cover_image")
        ]
        logger.debug("💿 Discogs returned %d results for query: %s", len(urls), query)
        return urls
    except Exception as exc:
        logger.warning("💿 ❌ Discogs provider error for query %s: %s", query, exc)
        return []


def provider_juno(query: str) -> List[str]:
    try:
        search_url = (
            "https://www.junodownload.com/search/?q[all][]=%s"
            % requests.utils.quote(query)
        )
        html = requests.get(search_url, timeout=10).text
        urls = re.findall(r"https://images\.junodownload\.com/[^\"']+\.jpg", html)
        seen = set()
        uniq = []
        for u in urls:
            if u not in seen:
                uniq.append(u)
                seen.add(u)
        logger.debug(
            "🎧 Juno returned %d candidate URLs for query: %s", len(uniq), query
        )
        return uniq[:12]
    except Exception as exc:
        logger.warning("🎧 ❌ Juno provider error for query %s: %s", query, exc)
        return []


def provider_spotify(query: str) -> List[str]:
    cid = config.get("SPOTIFY_CLIENT_ID")
    secret = config.get("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        logger.debug("🎵 Spotify provider skipped (missing client id/secret).")
        return []
    try:
        tok_resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(cid, secret),
            timeout=8,
        )
        tok_resp.raise_for_status()
        tok = tok_resp.json().get("access_token")
        if not tok:
            logger.debug("🎵 Spotify token missing for query: %s", query)
            return []
        headers = {"Authorization": "Bearer %s" % tok}
        r = requests.get(
            "https://api.spotify.com/v1/search",
            params={"q": query, "type": "track", "limit": 3},
            headers=headers,
            timeout=8,
        )
        r.raise_for_status()
        items = r.json().get("tracks", {}).get("items", [])
        urls: List[str] = []
        for it in items:
            for img in it.get("album", {}).get("images", []):
                if img.get("url"):
                    urls.append(img["url"])
        out, seen = [], set()
        for u in urls:
            if u not in seen:
                out.append(u)
                seen.add(u)
        logger.debug("🎵 Spotify returned %d results for query: %s", len(out), query)
        return out
    except Exception as exc:
        logger.warning("🎵 ❌ Spotify provider error for query %s: %s", query, exc)
        return []


def provider_soundcloud(query: str) -> List[str]:
    """
    Primary: query SoundCloud API using SOUNDCLOUD_CLIENT_ID (if present).
    Fallback: if no client id is configured, use Google CSE restricted to site:soundcloud.com.
    """
    cid = config.get("SOUNDCLOUD_CLIENT_ID")
    if not cid:
        logger.debug(
            "☁️ SoundCloud client id missing; falling back to Google (site:soundcloud.com) for query: %s",
            query,
        )
        try:
            # Reuse provider_google to search only within soundcloud.com
            return provider_google(f"{query} site:soundcloud.com")
        except Exception as exc:
            logger.warning(
                "☁️ ❌ SoundCloud → Google fallback error for query %s: %s", query, exc
            )
            return []

    # If we have a client id, use the SoundCloud API as before
    try:
        r = requests.get(
            "https://api.soundcloud.com/tracks",
            params={"q": query, "client_id": cid, "limit": 5},
            timeout=10,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        tracks = r.json()
        urls: List[str] = []
        for t in tracks if isinstance(tracks, list) else []:
            art = t.get("artwork_url")
            if art:
                urls.extend(
                    [
                        art.replace("large", "t2000x2000"),
                        art.replace("large", "t1200x1200"),
                        art.replace("large", "t1000x1000"),
                        art.replace("large", "t800x800"),
                        art.replace("large", "t500x500"),
                        art,
                    ]
                )
        out, seen = [], set()
        for u in urls:
            if u not in seen:
                out.append(u)
                seen.add(u)
        logger.debug("☁️ SoundCloud returned %d results for query: %s", len(out), query)
        return out
    except Exception as exc:
        logger.warning("☁️ ❌ SoundCloud provider error for query %s: %s", query, exc)
        return []


PROVIDER_MAP = {
    "google": provider_google,
    "itunes": provider_itunes,
    "discogs": provider_discogs,
    "juno": provider_juno,
    "spotify": provider_spotify,
    "soundcloud": provider_soundcloud,
}


# ============== SEARCH & EMBED ==============


def search_artwork(query: str) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
    if config.get("USE_CACHE", True) and query in cache:
        cached_url = cache[query]
        logger.info("💾 Using cached artwork for: %s", query)
        res = fetch_image(cached_url)
        if res:
            _, content, _, mime = res
            return cached_url, content, mime
        logger.info("💾 ❌ Cached URL invalid; re-searching for: %s", query)

    enabled = config.get("ENABLED_PROVIDERS") or DEFAULT_CONFIG["ENABLED_PROVIDERS"]
    for name in enabled:
        provider = PROVIDER_MAP.get(name.lower())
        if not provider:
            logger.warning("❓ Unknown provider in config: %s", name)
            continue
        logger.info("🔎 Trying provider: %s", name)
        urls = provider(query)
        if not urls:
            logger.debug("🔎 Provider %s returned no URLs for query: %s", name, query)
            continue
        chosen_url, image_bytes, mime = select_image_from_candidates(urls)
        if chosen_url and image_bytes:
            logger.info("✅ Artwork selected via %s", name)
            if config.get("USE_CACHE", True):  # ✅ save only if enabled
                cache[query] = chosen_url
                save_cache()
            return chosen_url, image_bytes, mime

    logger.info("❌ No approved artwork found from any provider for: %s", query)
    return None, None, None


def embed_into_existing_file(target_path: str, image_bytes: bytes, mime: str) -> bool:
    """
    Embed artwork into the file at target_path (modifies file in place).
    Returns True on success, False on failure.
    """
    try:
        ext = os.path.splitext(target_path)[1].lower()
        if ext == ".mp3":
            audio = MP3(target_path, ID3=ID3)
            try:
                audio.add_tags()
            except ID3Error:
                pass

            # Remove existing APIC (artwork) frames
            if audio.tags:
                for key in list(audio.tags.keys()):
                    if key.startswith("APIC"):
                        del audio.tags[key]

            # Add new artwork
            audio.tags.add(
                APIC(
                    encoding=3,
                    mime=mime or "image/jpeg",
                    type=3,  # cover (front)
                    desc="Cover",
                    data=image_bytes,
                )
            )
            audio.save(target_path, v2_version=3)
            return True
        elif ext == ".wav":
            audio = WAVE(target_path)
            if audio.tags is None:
                audio.add_tags()

            # Remove existing APIC (artwork) frames
            if audio.tags:
                for key in list(audio.tags.keys()):
                    if key.startswith("APIC"):
                        del audio.tags[key]

            # Add new artwork
            audio.tags.add(
                APIC(
                    encoding=3,
                    mime=mime or "image/jpeg",
                    type=3,
                    desc="Cover",
                    data=image_bytes,
                )
            )
            audio.save(target_path)
            return True
        logger.warning("❓ Unsupported file type: %s", target_path)
        return False
    except Exception as exc:
        logger.error("💥 Embed error for %s: %s", target_path, exc)
        return False


def transfer_original_post_success(
    src_path: str, processed_dir: str
) -> Tuple[bool, Optional[str]]:
    """
    After success: if FILE_ACTION == 'move' move original into processed_dir/originals/
    if FILE_ACTION == 'copy' do nothing (return True, None).
    """
    action = config.get("FILE_ACTION", "copy")
    basename = os.path.basename(src_path)
    if action == "copy":
        return True, None
    # action == move
    try:
        originals_dir = os.path.join(processed_dir, "originals")
        os.makedirs(originals_dir, exist_ok=True)
        dst = os.path.join(originals_dir, basename)
        os.replace(src_path, dst)
        return True, dst
    except Exception as exc:
        logger.error(
            "📁 ❌ Move original to processed/originals failed for %s: %s",
            src_path,
            exc,
        )
        return False, None


def transfer_original_post_failure(
    src_path: str, unprocessed_dir: str
) -> Tuple[bool, Optional[str]]:
    """
    After failure: if FILE_ACTION == 'move' move original to unprocessed_dir
    if FILE_ACTION == 'copy' copy original to unprocessed_dir
    """
    action = config.get("FILE_ACTION", "copy")
    basename = os.path.basename(src_path)
    dst = os.path.join(unprocessed_dir, basename)
    try:
        if action == "move":
            os.replace(src_path, dst)
        else:
            shutil.copy2(src_path, dst)
        return True, dst
    except Exception as exc:
        logger.error(
            "📁 ❌ Transfer to Unprocessed failed for %s -> %s: %s", src_path, dst, exc
        )
        return False, None


# ============== MAIN ==============


def main():
    # folder = os.path.dirname(os.path.abspath(__file__))
    logger.info("📂 Working folder: %s", folder)
    processed_dir = os.path.join(folder, str(config["PROCESSED_DIR"]))
    unprocessed_dir = os.path.join(folder, str(config["UNPROCESSED_DIR"]))
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(unprocessed_dir, exist_ok=True)

    audio_files = [
        f for f in os.listdir(folder) if f.lower().endswith((".mp3", ".wav"))
    ]
    if not audio_files:
        logger.warning("🔍 ❌ No MP3/WAV files found in current folder.")
        return

    total = len(audio_files)
    success = 0
    failed = 0
    skipped = 0
    failed_list: List[str] = []

    logger.info("🎵 Found %d audio files to process.", total)

    for i, fname in enumerate(
        tqdm(audio_files, desc="Processing tracks", unit="file"), start=1
    ):
        print("\n")
        src_path = os.path.join(folder, fname)

        if config.get("SKIP_PROCESSED", True):
            # if os.path.exists(os.path.join(processed_dir, fname)) or os.path.exists(
            #     os.path.join(unprocessed_dir, fname)
            # ):
            if os.path.exists(os.path.join(processed_dir, fname)):
                logger.info("⏭️ Skipping already processed: %s", fname)
                logger.info("📊 Progress: %d/%d files completed", i, total)
                skipped += 1
                continue

        query = clean_query(fname)
        logger.info("🎶 Processing: %s", fname)
        logger.info("🔍 Query: %s", query)

        url, image_bytes, mime = search_artwork(query)

        # Copy original to Processed folder first (always)
        processed_copy_path = os.path.join(processed_dir, fname)
        try:
            shutil.copy2(src_path, processed_copy_path)
            logger.debug("📁 Copied original to %s", processed_copy_path)
        except Exception as exc:
            logger.error(
                "📁 ❌ Failed to copy original to Processed: %s -> %s: %s",
                src_path,
                processed_copy_path,
                exc,
            )
            # If cannot copy to processed, treat as failure and transfer original to Unprocessed per FILE_ACTION
            transferred, tpath = transfer_original_post_failure(
                src_path, unprocessed_dir
            )
            if transferred:
                logger.warning(
                    "⚠️ Could not copy to Processed; original transferred to Unprocessed: %s",
                    tpath,
                )
            else:
                logger.error(
                    "💥 Could not copy to Processed and failed to transfer original for %s",
                    src_path,
                )
            failed += 1
            failed_list.append(fname)
            logger.info("📊 Progress: %d/%d files completed", i, total)
            continue

        if url and image_bytes:
            # Try embedding into the processed copy (in-place)
            ok = embed_into_existing_file(processed_copy_path, image_bytes, mime)
            if ok:
                logger.info("✅ Saved with artwork → %s", processed_copy_path)
                transferred_ok, tpath = transfer_original_post_success(
                    src_path, processed_dir
                )
                if transferred_ok and tpath:
                    logger.info("📁 Original moved to: %s", tpath)
                elif transferred_ok:
                    logger.debug("📁 Original left in place (FILE_ACTION=copy).")
                else:
                    logger.warning(
                        "⚠️ Original transfer after success failed for %s", src_path
                    )
                success += 1
            else:
                # embedding failed -> remove processed copy, transfer original to Unprocessed
                try:
                    if os.path.exists(processed_copy_path):
                        os.remove(processed_copy_path)
                        logger.debug(
                            "🗑️ Removed failed processed copy: %s", processed_copy_path
                        )
                except Exception as exc:
                    logger.warning(
                        "🗑️ ❌ Could not remove failed processed copy %s: %s",
                        processed_copy_path,
                        exc,
                    )
                transferred, tpath = transfer_original_post_failure(
                    src_path, unprocessed_dir
                )
                if transferred:
                    logger.error(
                        "❌ Failed to embed → original transferred to %s", tpath
                    )
                else:
                    logger.error(
                        "💥 Failed to embed and failed to transfer original for %s",
                        src_path,
                    )
                failed += 1
                failed_list.append(fname)
        else:
            # No artwork chosen/found -> delete processed copy and transfer original to Unprocessed
            try:
                if os.path.exists(processed_copy_path):
                    os.remove(processed_copy_path)
                    logger.debug(
                        "🗑️ Removed processed copy (no artwork): %s", processed_copy_path
                    )
            except Exception as exc:
                logger.warning(
                    "🗑️ ❌ Could not remove processed copy %s: %s",
                    processed_copy_path,
                    exc,
                )
            transferred, tpath = transfer_original_post_failure(
                src_path, unprocessed_dir
            )
            if transferred:
                logger.warning(
                    "⚠️ No artwork approved → original transferred to %s", tpath
                )
            else:
                logger.error(
                    "💥 No artwork approved and failed to transfer original for %s",
                    src_path,
                )
            failed += 1
            failed_list.append(fname)

        logger.info("📊 Progress: %d/%d files completed\n", i, total)

    print("\n")
    logger.info("📋 ===== SUMMARY =====")
    logger.info("📊 Total files: %d", total)
    logger.info("⏭️ Skipped (already processed): %d", skipped)
    logger.info("✅ Successfully processed: %d", success)
    logger.info("❌ Unprocessed: %d", failed)
    if failed_list:
        logger.info("⚠️ Files needing attention:")
        for failed_file in failed_list:
            logger.info(" - %s", failed_file)
    logger.info("📋 ====================")


if __name__ == "__main__":
    main()
