"""
memory_manager.py — Mark-XXX + Mark-XXXV Merged Memory System
==============================================================
Changes merged from XXXV:
  - Added 'projects' and 'wishes' memory categories
  - Memory entries now store an 'updated' timestamp
  - format_memory_for_prompt: richer output with grouped sections
  - should_extract_memory / extract_memory: broader, more aggressive extraction
  - remember() / forget() helpers added
  - MAX_VALUE_LENGTH raised to 400
  - load_memory auto-migrates old JSON by adding missing categories
Kept from XXX:
  - process_memory_update_async() wrapper (called by main.py)
  - _memory_turn_counter / _MEMORY_EVERY_N_TURNS throttling
"""

import json
import re
import threading
from datetime import datetime
from threading import Lock
from pathlib import Path
import sys


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR         = get_base_dir()
MEMORY_PATH      = BASE_DIR / "memory" / "long_term.json"
_lock            = Lock()
MAX_VALUE_LENGTH = 400

# Turn-counter throttle (used by process_memory_update_async)
_memory_turn_lock     = threading.Lock()
_memory_turn_counter  = 0
_MEMORY_EVERY_N_TURNS = 5
_last_memory_input    = ""


# ── Schema ────────────────────────────────────────────────────────────────────

def _empty_memory() -> dict:
    return {
        "identity":      {},
        "preferences":   {},
        "projects":      {},
        "relationships": {},
        "wishes":        {},
        "notes":         {}
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return _empty_memory()

    with _lock:
        try:
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Auto-migrate: add any missing categories
                base = _empty_memory()
                for key in base:
                    if key not in data:
                        data[key] = {}
                return data
            return _empty_memory()
        except Exception as e:
            print(f"[Memory] ⚠️ Load error: {e}")
            return _empty_memory()


def save_memory(memory: dict) -> None:
    if not isinstance(memory, dict):
        return
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        MEMORY_PATH.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _truncate_value(val: str) -> str:
    if isinstance(val, str) and len(val) > MAX_VALUE_LENGTH:
        return val[:MAX_VALUE_LENGTH].rstrip() + "…"
    return val


def _recursive_update(target: dict, updates: dict) -> bool:
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        if isinstance(value, dict) and "value" not in value:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
                changed = True
            if _recursive_update(target[key], value):
                changed = True
        else:
            if isinstance(value, dict) and "value" in value:
                new_val = _truncate_value(str(value["value"]))
            else:
                new_val = _truncate_value(str(value))

            entry    = {"value": new_val, "updated": datetime.now().strftime("%Y-%m-%d")}
            existing = target.get(key, {})
            if not isinstance(existing, dict) or existing.get("value") != new_val:
                target[key] = entry
                changed = True

    return changed


def update_memory(memory_update: dict) -> dict:
    if not isinstance(memory_update, dict) or not memory_update:
        return load_memory()

    memory = load_memory()
    if _recursive_update(memory, memory_update):
        save_memory(memory)
        print(f"[Memory] 💾 Saved: {list(memory_update.keys())}")
    return memory


# ── Extraction (2-stage) ──────────────────────────────────────────────────────

def should_extract_memory(user_text: str, jarvis_text: str, api_key: str) -> bool:
    """Stage 1: quick YES/NO check with broad criteria."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        combined = f"User: {user_text[:300]}\nJarvis: {jarvis_text[:200]}"
        check = model.generate_content(
            f"Does this conversation contain ANY of the following?\n"
            f"- Personal facts (name, age, city, job, birthday, nationality)\n"
            f"- Preferences or favorites (food, color, music, sport, game, film, book, etc.)\n"
            f"- Active projects or goals the user is working on\n"
            f"- People in the user's life (friends, family, partner, colleagues)\n"
            f"- Things the user wants to do or buy in the future\n"
            f"- Any other fact worth remembering long-term\n\n"
            f"Reply only YES or NO.\n\nConversation:\n{combined}"
        )
        return "YES" in check.text.upper()
    except Exception as e:
        print(f"[Memory] ⚠️ Stage1 check failed: {e}")
        return False


def extract_memory(user_text: str, jarvis_text: str, api_key: str) -> dict:
    """Stage 2: detailed extraction from both user and Jarvis turns."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        combined = f"User: {user_text[:500]}\nJarvis: {jarvis_text[:300]}"
        raw = model.generate_content(
            f"Extract ALL memorable personal facts from this conversation. Any language.\n"
            f"Return ONLY valid JSON. Use {{}} if truly nothing is worth saving.\n\n"
            f"Category guide:\n"
            f"  identity      → name, age, birthday, city, country, job, school, nationality, language\n"
            f"  preferences   → ANY favorite or preferred thing:\n"
            f"                  favorite_food, favorite_color, favorite_music, favorite_film,\n"
            f"                  favorite_game, favorite_sport, favorite_book, favorite_artist,\n"
            f"                  favorite_country, hobbies, interests, dislikes, etc.\n"
            f"  projects      → projects being built, ongoing work, goals, ideas in progress\n"
            f"  relationships → people mentioned: friends, family, partner, colleagues\n"
            f"  wishes        → future plans, things to buy, travel plans, dreams\n"
            f"  notes         → anything else worth remembering (habits, schedule, etc.)\n\n"
            f"IMPORTANT:\n"
            f"- Be LIBERAL: if something MIGHT be worth remembering, include it.\n"
            f"- Extract from BOTH user and Jarvis turns.\n"
            f"- Skip: weather, reminders, search results, one-time commands.\n"
            f"- Use concise English values regardless of conversation language.\n\n"
            f"Format:\n"
            f'{{"identity":{{"name":{{"value":"Ali"}}}},\n'
            f' "preferences":{{"favorite_color":{{"value":"blue"}}, "hobby":{{"value":"gaming"}}}},\n'
            f' "projects":{{"mark_xxx":{{"value":"JARVIS-like AI assistant on Windows"}}}},\n'
            f' "relationships":{{"friend_yusuf":{{"value":"close friend"}}}},\n'
            f' "wishes":{{"buy_guitar":{{"value":"wants an acoustic guitar"}}}},\n'
            f' "notes":{{"works_at_night":{{"value":"usually active late at night"}}}}}}\n\n'
            f"Conversation:\n{combined}\n\nJSON:"
        ).text.strip()

        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        if not raw or raw == "{}":
            return {}
        return json.loads(raw)

    except json.JSONDecodeError:
        return {}
    except Exception as e:
        if "429" not in str(e):
            print(f"[Memory] ⚠️ Extract failed: {e}")
        return {}


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_memory_for_prompt(memory: dict | None) -> str:
    if not memory:
        return ""

    lines = []

    # Identity
    identity  = memory.get("identity", {})
    id_fields = ["name", "age", "birthday", "city", "job", "language", "school", "nationality"]
    for field in id_fields:
        entry = identity.get(field)
        if entry:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"{field.title()}: {val}")
    for key, entry in identity.items():
        if key in id_fields:
            continue
        val = entry.get("value") if isinstance(entry, dict) else entry
        if val:
            lines.append(f"{key.replace('_', ' ').title()}: {val}")

    # Preferences
    prefs = memory.get("preferences", {})
    if prefs:
        lines.append("")
        lines.append("Preferences:")
        for key, entry in list(prefs.items())[:15]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    # Projects
    projects = memory.get("projects", {})
    if projects:
        lines.append("")
        lines.append("Active Projects / Goals:")
        for key, entry in list(projects.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    # Relationships
    rels = memory.get("relationships", {})
    if rels:
        lines.append("")
        lines.append("People in their life:")
        for key, entry in list(rels.items())[:10]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    # Wishes
    wishes = memory.get("wishes", {})
    if wishes:
        lines.append("")
        lines.append("Wishes / Plans / Wants:")
        for key, entry in list(wishes.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    # Notes
    notes = memory.get("notes", {})
    if notes:
        lines.append("")
        lines.append("Other notes:")
        for key, entry in list(notes.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key}: {val}")

    if not lines:
        return ""

    header = "[WHAT YOU KNOW ABOUT THIS PERSON — use naturally, never recite like a list]\n"
    result = header + "\n".join(lines)
    if len(result) > 2000:
        result = result[:1997] + "…"

    return result + "\n"


# ── Public helpers ────────────────────────────────────────────────────────────

def remember(key: str, value: str, category: str = "notes") -> str:
    """Manually store a fact in memory."""
    valid = {"identity", "preferences", "projects", "relationships", "wishes", "notes"}
    if category not in valid:
        category = "notes"
    update_memory({category: {key: {"value": value}}})
    return f"Remembered: {category}/{key} = {value}"


def forget(key: str, category: str = "notes") -> str:
    """Remove a fact from memory."""
    memory = load_memory()
    cat    = memory.get(category, {})
    if key in cat:
        del cat[key]
        memory[category] = cat
        save_memory(memory)
        return f"Forgotten: {category}/{key}"
    return f"Not found: {category}/{key}"


forget_memory = forget  # backwards-compat alias


# ── Async wrapper (called by main.py every N turns) ───────────────────────────

def process_memory_update_async(user_text: str, jarvis_text: str, api_key: str) -> None:
    """
    Throttled, 2-stage memory updater.
    main.py calls this in a daemon thread every turn;
    actual extraction only runs every _MEMORY_EVERY_N_TURNS turns.
    """
    global _memory_turn_counter, _last_memory_input

    with _memory_turn_lock:
        _memory_turn_counter += 1
        current_count = _memory_turn_counter

    if current_count % _MEMORY_EVERY_N_TURNS != 0:
        return

    text = (user_text or "").strip()
    if len(text) < 10 or text == _last_memory_input:
        return
    _last_memory_input = text

    if not should_extract_memory(user_text, jarvis_text, api_key):
        return

    data = extract_memory(user_text, jarvis_text, api_key)
    if data:
        update_memory(data)
        print(f"[Memory] ✅ Updated: {list(data.keys())}")
