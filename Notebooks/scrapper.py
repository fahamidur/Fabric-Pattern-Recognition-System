# ============================================================
# Textile Pattern Image Scraper for Local Device
# DDGS + Wikimedia fallback, resilient retries, duplicate checks
# ============================================================

# Install first (terminal):
# pip uninstall -y duckduckgo_search
# pip install -U ddgs requests pillow tqdm pillow-heif

import io
import re
import os
import time
import json
import random
import hashlib
import warnings
from pathlib import Path
from urllib.parse import quote_plus

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Optional HEIF support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# DDGS package
try:
    from ddgs import DDGS
    try:
        from ddgs.exceptions import DDGSException
    except Exception:
        DDGSException = Exception
except Exception as e:
    raise ImportError("ddgs import failed. Run: pip install -U ddgs") from e

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------
# 1) Configuration
# ---------------------------
PROJECT_NAME = "textile_pattern_classifier"

# Change if needed
BASE_DIR = Path.cwd() / "TextilePatternScraper"

RAW_DIR = BASE_DIR / "raw_images"
LOG_DIR = BASE_DIR / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_PER_CLASS_TARGET = 300
MAX_DOWNLOAD_ATTEMPTS_PER_CLASS = 3000

# Search controls
DDG_BATCH_SIZE = 50
DDG_MAX_CLASS_FETCH_ATTEMPTS = 10
DDG_PAUSE_BETWEEN_BATCHES = (1.5, 3.0)
DDG_PAUSE_BETWEEN_CLASSES = (2.0, 5.0)
DDG_BACKOFF_ON_ERROR = (3.0, 6.0)
DDG_CONSECUTIVE_EMPTY_LIMIT = 4

# Wikimedia fallback controls
WIKI_MAX_RESULTS_PER_QUERY = 80

# Download controls
REQUEST_TIMEOUT = (10, 25)
MAX_BYTES = 20 * 1024 * 1024
MIN_DIM = 128
MAX_DIM = 7000
JPEG_QUALITY = 92

FORCE_RGB_SAVE = True

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

CLASSES = [
    "Damask",
    "Matelasse",
    "Quatrefoil",
    "Houndstooth",
    "Suzani",
    "Chevron",
    "Paisley",
    "Ogee",
    "Jacobean",
    "Ikat",
    "Animal_Print",
    "Dot_Polka_Dot",
    "Herringbone",
    "Plaid_Checkered",
    "Gingham",
]

SEARCH_QUERIES = {
    "Damask": "damask fabric pattern textile",
    "Matelasse": "matelasse fabric pattern textile",
    "Quatrefoil": "quatrefoil fabric pattern textile",
    "Houndstooth": "houndstooth fabric textile pattern",
    "Suzani": "suzani textile pattern fabric",
    "Chevron": "chevron fabric pattern textile",
    "Paisley": "paisley fabric pattern textile",
    "Ogee": "ogee fabric pattern textile",
    "Jacobean": "jacobean floral fabric pattern textile",
    "Ikat": "ikat fabric textile pattern",
    "Animal_Print": "animal print fabric textile pattern",
    "Dot_Polka_Dot": "polka dot fabric textile pattern",
    "Herringbone": "herringbone fabric textile pattern",
    "Plaid_Checkered": "plaid checkered fabric textile pattern",
    "Gingham": "gingham fabric textile pattern",
}

NEGATIVE_HINTS = {
    "Damask": ["wallpaper", "vector", "clipart"],
    "Matelasse": ["amazon listing screenshot", "bedspread listing"],
    "Quatrefoil": ["logo", "icon", "svg"],
    "Houndstooth": ["logo", "illustration", "vector"],
    "Suzani": ["wall art frame", "embroidery hoop"],
    "Chevron": ["road sign", "military insignia"],
    "Paisley": ["logo", "tattoo"],
    "Ogee": ["architecture", "tile"],
    "Jacobean": ["wall art print mockup", "painting"],
    "Ikat": ["vector", "clipart"],
    "Animal_Print": ["wildlife photography", "real animal photo"],
    "Dot_Polka_Dot": ["icon", "cartoon"],
    "Herringbone": ["brick", "floor tile"],
    "Plaid_Checkered": ["shirt model", "product mockup"],
    "Gingham": ["picnic scene", "tablecloth mockup"],
}

# Shared DDGS client
DDGS_CLIENT = DDGS()


# ---------------------------
# 2) Helpers
# ---------------------------
def random_sleep(rng_tuple):
    time.sleep(random.uniform(*rng_tuple))


def slugify(text, max_len=80):
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return text[:max_len].strip("_") or "img"


def hash_bytes(b):
    return hashlib.sha256(b).hexdigest()


def session_with_retries():
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Connection": "keep-alive",
    })
    return s


def is_likely_bad_url(url):
    if not url or not isinstance(url, str):
        return True
    u = url.lower()
    return u.startswith(("data:", "blob:", "javascript:"))


def contains_negative_hints(text, hints):
    if not text or not hints:
        return False
    t = text.lower()
    return any(h.lower() in t for h in hints)


def normalise_result_url(item):
    return (
        item.get("image")
        or item.get("url")
        or item.get("thumbnail")
        or item.get("src")
    )


def validate_and_load_image(raw_bytes):
    if not raw_bytes or len(raw_bytes) < 1024:
        raise ValueError("File too small")
    if len(raw_bytes) > MAX_BYTES:
        raise ValueError("File too large")

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except UnidentifiedImageError as e:
        raise ValueError("Unidentified image") from e

    w, h = img.size
    if w < MIN_DIM or h < MIN_DIM:
        raise ValueError(f"Image too small ({w}x{h})")
    if w > MAX_DIM or h > MAX_DIM:
        pass

    return img


def save_image_standardised(img, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if FORCE_RGB_SAVE:
        if img.mode in ("RGBA", "LA"):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        elif img.mode == "P":
            tmp = img.convert("RGBA")
            bg = Image.new("RGB", tmp.size, (255, 255, 255))
            bg.paste(tmp, mask=tmp.split()[-1] if "A" in tmp.getbands() else None)
            img = bg
        else:
            img = img.convert("RGB")

        final_path = out_path.with_suffix(".jpg")
        img.save(final_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return final_path

    final_path = out_path.with_suffix(".png")
    img.save(final_path)
    return final_path


def load_existing_hashes(class_dir):
    hash_file = class_dir / "_hashes.txt"
    hashes = set()
    if hash_file.exists():
        with open(hash_file, "r", encoding="utf-8") as f:
            hashes = {line.strip() for line in f if line.strip()}
    return hashes


def append_hash(class_dir, h):
    with open(class_dir / "_hashes.txt", "a", encoding="utf-8") as f:
        f.write(h + "\n")


def load_seen_urls(class_dir):
    url_file = class_dir / "_seen_urls.txt"
    urls = set()
    if url_file.exists():
        with open(url_file, "r", encoding="utf-8") as f:
            urls = {line.strip() for line in f if line.strip()}
    return urls


def append_seen_url(class_dir, u):
    with open(class_dir / "_seen_urls.txt", "a", encoding="utf-8") as f:
        f.write(u + "\n")


def count_valid_images(class_dir):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
    return sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


# ---------------------------
# 3) Search wrappers
# ---------------------------
def ddg_image_search(query, max_results=50, region="wt-wt", safesearch="off"):
    """
    Robust DDGS wrapper.
    Returns [] when DDGS says 'No results found' instead of raising.
    """
    results = []

    call_attempts = [
        lambda: DDGS_CLIENT.images(query, max_results=max_results, region=region, safesearch=safesearch),
        lambda: DDGS_CLIENT.images(query, region=region, safesearch=safesearch, max_results=max_results),
        lambda: DDGS_CLIENT.images(query, max_results=max_results),
        lambda: DDGS_CLIENT.images(query),
    ]

    last_error = None
    for attempt in call_attempts:
        try:
            gen = attempt()
            for item in gen:
                if isinstance(item, dict):
                    results.append(item)
            return results
        except DDGSException as e:
            # ddgs often throws "No results found." as an exception
            msg = str(e).lower()
            if "no results found" in msg:
                return []
            last_error = e
        except TypeError as e:
            last_error = e
            continue
        except Exception as e:
            # Similar handling if the library wraps exceptions differently
            msg = str(e).lower()
            if "no results found" in msg:
                return []
            last_error = e

    if last_error:
        raise last_error
    return results


def build_query_variants(base_query, class_name):
    plain_cls = class_name.replace("_", " ")
    variants = [
        base_query,
        f"{plain_cls} fabric",
        f"{plain_cls} textile",
        f"{plain_cls} pattern fabric",
        f"{base_query} swatch",
        f"{base_query} close up",
        f"{base_query} upholstery",
        f"{base_query} seamless",
        f"{base_query} woven",
        f"{base_query} printed",
    ]
    seen = set()
    out = []
    for q in variants:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def fetch_ddgs_results(query, class_name, target_fetch):
    collected = []
    seen_result_urls = set()
    variants = build_query_variants(query, class_name)
    consecutive_empty = 0

    for attempt_num in range(1, DDG_MAX_CLASS_FETCH_ATTEMPTS + 1):
        active_query = variants[(attempt_num - 1) % len(variants)]
        try:
            batch = ddg_image_search(active_query, max_results=DDG_BATCH_SIZE)

            added_this_round = 0
            for item in batch:
                url = normalise_result_url(item)
                title = item.get("title") or item.get("source") or item.get("description") or ""

                if is_likely_bad_url(url):
                    continue

                if contains_negative_hints(f"{title} {url}", NEGATIVE_HINTS.get(class_name, [])):
                    continue

                if url not in seen_result_urls:
                    seen_result_urls.add(url)
                    collected.append(item)
                    added_this_round += 1

            print(
                f"  DDG attempt {attempt_num} | query '{active_query}' | "
                f"added {added_this_round} | collected {len(collected)}"
            )

            if added_this_round == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            if len(collected) >= target_fetch:
                break

            if consecutive_empty >= DDG_CONSECUTIVE_EMPTY_LIMIT:
                print("  DDG produced repeated empty batches. Switching to fallback source.")
                break

            random_sleep(DDG_PAUSE_BETWEEN_BATCHES)

        except Exception as e:
            print(f"  DDG fetch error on attempt {attempt_num} with query '{active_query}': {e}")
            random_sleep(DDG_BACKOFF_ON_ERROR)

    return collected


def wikimedia_image_search(query, max_results=80):
    """
    Wikimedia Commons API fallback.
    Returns DDG-like dicts with 'image' and 'title'.
    """
    # Commons search works better with broader terms
    api = "https://commons.wikimedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,           # File namespace
        "gsrlimit": min(max_results, 50),
        "prop": "imageinfo",
        "iiprop": "url|mime",
    }

    out = []
    try:
        r = requests.get(api, params=params, timeout=(10, 20), headers={"User-Agent": random.choice(USER_AGENTS)})
        r.raise_for_status()
        data = r.json()

        pages = (data.get("query") or {}).get("pages") or {}
        for _, page in pages.items():
            title = page.get("title", "")
            ii = page.get("imageinfo", [])
            if not ii:
                continue
            url = ii[0].get("url")
            mime = ii[0].get("mime", "")
            if not url:
                continue
            if "image/" not in str(mime).lower():
                continue

            out.append({
                "title": title,
                "image": url,
                "source": "wikimedia",
            })
    except Exception:
        return []

    return out


def fetch_search_results_with_fallback(query, class_name, target_fetch):
    collected = []
    seen_urls = set()

    # First source DDGS
    ddg_results = fetch_ddgs_results(query, class_name, target_fetch=target_fetch)
    for item in ddg_results:
        u = normalise_result_url(item)
        if not is_likely_bad_url(u) and u not in seen_urls:
            seen_urls.add(u)
            collected.append(item)

    # If DDGS is weak or empty, use Wikimedia fallback
    if len(collected) < target_fetch:
        variants = build_query_variants(query, class_name)
        for v in variants[:6]:
            wiki_batch = wikimedia_image_search(v, max_results=WIKI_MAX_RESULTS_PER_QUERY)
            added = 0
            for item in wiki_batch:
                u = normalise_result_url(item)
                title = item.get("title") or ""
                if is_likely_bad_url(u):
                    continue
                if contains_negative_hints(f"{title} {u}", NEGATIVE_HINTS.get(class_name, [])):
                    continue
                if u not in seen_urls:
                    seen_urls.add(u)
                    collected.append(item)
                    added += 1

            print(f"  Wikimedia fallback | query '{v}' | added {added} | collected {len(collected)}")

            if len(collected) >= target_fetch:
                break

            time.sleep(random.uniform(0.5, 1.5))

    return collected


# ---------------------------
# 4) Download pipeline
# ---------------------------
def download_one_image(url, dest_dir, class_name, img_index, existing_hashes, seen_urls, logs):
    if url in seen_urls:
        logs.append({"status": "skip_seen_url", "class": class_name, "url": url})
        return False

    session = session_with_retries()

    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True) as r:
            if r.status_code != 200:
                logs.append({"status": "http_error", "class": class_name, "url": url, "code": r.status_code})
                append_seen_url(dest_dir, url)
                seen_urls.add(url)
                return False

            content_type = (r.headers.get("Content-Type") or "").lower()
            if "image" not in content_type:
                logs.append({"status": "not_image", "class": class_name, "url": url, "content_type": content_type})
                append_seen_url(dest_dir, url)
                seen_urls.add(url)
                return False

            data = bytearray()
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    logs.append({"status": "too_large", "class": class_name, "url": url})
                    append_seen_url(dest_dir, url)
                    seen_urls.add(url)
                    return False

        raw_bytes = bytes(data)
        content_hash = hash_bytes(raw_bytes)
        if content_hash in existing_hashes:
            logs.append({"status": "duplicate_hash", "class": class_name, "url": url})
            append_seen_url(dest_dir, url)
            seen_urls.add(url)
            return False

        img = validate_and_load_image(raw_bytes)

        final_base = dest_dir / slugify(f"{class_name}_{img_index:05d}")
        saved_path = save_image_standardised(img, final_base)

        existing_hashes.add(content_hash)
        append_hash(dest_dir, content_hash)
        append_seen_url(dest_dir, url)
        seen_urls.add(url)

        logs.append({
            "status": "saved",
            "class": class_name,
            "url": url,
            "saved_path": str(saved_path),
            "size": list(img.size),
        })
        return True

    except (requests.exceptions.RequestException, ValueError, OSError) as e:
        logs.append({"status": "download_failed", "class": class_name, "url": url, "error": repr(e)})
        try:
            append_seen_url(dest_dir, url)
            seen_urls.add(url)
        except Exception:
            pass
        return False
    except Exception as e:
        logs.append({"status": "unexpected_error", "class": class_name, "url": url, "error": repr(e)})
        try:
            append_seen_url(dest_dir, url)
            seen_urls.add(url)
        except Exception:
            pass
        return False


# ---------------------------
# 5) Main run
# ---------------------------
def run_scraper():
    all_download_logs = []
    summary = []

    print(f"Saving dataset to: {BASE_DIR.resolve()}")

    for cls in CLASSES:
        query = SEARCH_QUERIES.get(cls, cls.replace("_", " "))
        class_raw_dir = RAW_DIR / cls
        class_raw_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 90)
        print(f"Class: {cls}")
        print(f"Query: {query}")

        existing_hashes = load_existing_hashes(class_raw_dir)
        seen_urls = load_seen_urls(class_raw_dir)
        current_count = count_valid_images(class_raw_dir)

        print(f"Existing valid images: {current_count}")

        if current_count >= IMAGES_PER_CLASS_TARGET:
            print("Target already met. Skipping.")
            summary.append({
                "class": cls,
                "query": query,
                "candidates": 0,
                "new_saved": 0,
                "saved_total": current_count,
                "attempts": 0,
                "status": "skipped_target_met"
            })
            random_sleep(DDG_PAUSE_BETWEEN_CLASSES)
            continue

        target_fetch = max(IMAGES_PER_CLASS_TARGET * 4, 400)

        collected_results = fetch_search_results_with_fallback(
            query=query,
            class_name=cls,
            target_fetch=target_fetch
        )

        candidate_urls = []
        for item in collected_results:
            u = normalise_result_url(item)
            if not is_likely_bad_url(u):
                candidate_urls.append(u)

        # Dedupe preserving order
        dedup_urls = []
        seen_tmp = set()
        for u in candidate_urls:
            if u not in seen_tmp:
                seen_tmp.add(u)
                dedup_urls.append(u)

        print(f"Unique candidate URLs: {len(dedup_urls)}")

        new_saved = 0
        attempts = 0
        img_index = current_count + 1
        needed = max(0, IMAGES_PER_CLASS_TARGET - current_count)

        with tqdm(total=needed, desc=cls, leave=False) as pbar:
            for url in dedup_urls:
                if current_count + new_saved >= IMAGES_PER_CLASS_TARGET:
                    break
                if attempts >= MAX_DOWNLOAD_ATTEMPTS_PER_CLASS:
                    print("Reached max download attempts for this class.")
                    break

                attempts += 1

                ok = download_one_image(
                    url=url,
                    dest_dir=class_raw_dir,
                    class_name=cls,
                    img_index=img_index,
                    existing_hashes=existing_hashes,
                    seen_urls=seen_urls,
                    logs=all_download_logs
                )

                if ok:
                    new_saved += 1
                    img_index += 1
                    pbar.update(1)

                time.sleep(random.uniform(0.2, 0.8))

        final_count = count_valid_images(class_raw_dir)
        print(f"Finished {cls} | new saved {new_saved} | final count {final_count}")

        summary.append({
            "class": cls,
            "query": query,
            "candidates": len(dedup_urls),
            "new_saved": new_saved,
            "saved_total": final_count,
            "attempts": attempts,
            "status": "done"
        })

        random_sleep(DDG_PAUSE_BETWEEN_CLASSES)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = LOG_DIR / f"scrape_summary_{timestamp}.json"
    logs_path = LOG_DIR / f"download_logs_{timestamp}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(all_download_logs, f, ensure_ascii=False, indent=2)

    print("\nSaved summary:", summary_path)
    print("Saved logs:", logs_path)

    print("\nFinal image counts by class")
    print("-" * 60)
    for row in summary:
        print(f"{row['class']:<20} {row['saved_total']:>5} images (new {row['new_saved']})")

    return summary, all_download_logs


# ---------------------------
# 6) Optional global dedupe
# ---------------------------
def global_deduplicate(root_dir):
    seen_hashes = {}
    removed = []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}

    for cls_dir in sorted([d for d in Path(root_dir).iterdir() if d.is_dir()]):
        for fp in sorted(cls_dir.iterdir()):
            if not fp.is_file() or fp.suffix.lower() not in exts:
                continue
            try:
                b = fp.read_bytes()
                h = hash_bytes(b)
                if h in seen_hashes:
                    removed.append((str(fp), seen_hashes[h]))
                    fp.unlink()
                else:
                    seen_hashes[h] = str(fp)
            except Exception:
                continue
    return removed


# ---------------------------
# 7) Optional zip dataset
# ---------------------------
def make_zip():
    import shutil
    zip_base = str(BASE_DIR / f"{PROJECT_NAME}_raw_images")
    zip_file = shutil.make_archive(zip_base, "zip", root_dir=str(RAW_DIR))
    print("ZIP created:", zip_file)
    return zip_file


if __name__ == "__main__":
    try:
        run_scraper()
    finally:
        try:
            DDGS_CLIENT.close()
        except Exception:
            pass

    # Optional post-processing
    # removed = global_deduplicate(RAW_DIR)
    # print("Global duplicates removed:", len(removed))
    # make_zip()
