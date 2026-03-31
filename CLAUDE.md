# CLAUDE.md — Mark-XXXV Project Context

## Project Overview
Mark-XXXV is a Windows-only JARVIS-like voice AI assistant built on Google Gemini 2.5 Flash Native Audio. It listens for the wake word "Jarvis", executes tool calls via the Gemini API, and controls the PC through a set of action modules.

This version merges the original Mark-XXX (with gesture control, agent system, separate tools_config.py) with the Mark-XXXV upstream (game_updater, enhanced memory system with projects/wishes categories, F4 mute shortcut).

## Architecture
- **main.py** — Entry point. `JarvisLive` class runs an asyncio loop with 4 concurrent tasks: mic input, audio send, response receive, audio playback. Tool calls are dispatched in `_execute_tool()` using `loop.run_in_executor()` to avoid blocking.
- **ui.py** — Tkinter UI with real-time log and face avatar. Runs on the main thread.
- **core/tools_config.py** — All Gemini tool declarations (`TOOL_DECLARATIONS` list).
- **core/prompt.txt** — System prompt (JARVIS personality).
- **actions/** — One module per tool. Each exposes a callable that accepts `parameters: dict` and `player` (UI ref).
- **agent/** — Task queue + planner for multi-step autonomous tasks.
- **memory/** — Persistent long-term memory (JSON), updated every 5 turns in a background thread.
- **config/api_keys.json** — API keys and camera index (gitignored).

## Adding a New Tool (standard pattern)
1. Create `actions/my_tool.py` with a function `my_tool(parameters: dict, player=None) -> str`
2. Add declaration to `core/tools_config.py` → `TOOL_DECLARATIONS`
3. Import in `main.py` and add `elif name == "my_tool":` branch in `_execute_tool()`

## Key Conventions
- All blocking actions use `await loop.run_in_executor(None, lambda: ...)` in `_execute_tool`.
- Long-running background tasks: spawn a `threading.Thread(daemon=True)` and return immediately (see `screen_process`).
- Camera index is stored in `config/api_keys.json` under `"camera_index"`.
- `pyautogui.FAILSAFE = True` — moving mouse to screen corner stops PyAutoGUI as a safety net.

## Dependencies
Install with `pip install -r requirements.txt` then `playwright install chromium`.

Key packages: `google-genai`, `pyaudio`, `pyautogui`, `opencv-python`, `playwright`, `pillow`, `numpy`, `mss`, `pycaw`, `comtypes`, `mediapipe`, `pywinauto`.

---

## Features Implemented

### Mute Button (`ui.py`)
Button in the input bar that completely silences the microphone feed to Gemini.
- `⏸ MUTE` → stops sending audio; turns green with `▶ UNMUTE`
- Calls `mute_callback(bool)` → sets `JarvisLive._muted` in `main.py`
- Audio is still read from mic (to drain the buffer) but not enqueued

### Hand Gesture Control (`actions/gesture_control.py`)
MediaPipe Tasks API (hand_landmarker, VIDEO mode) gesture controller running on a daemon thread.
Triggered with: `gesture_control({"action": "start"})` / `gesture_control({"action": "stop"})`.

| Gesto | Trigger | Acción |
|---|---|---|
| POINT | Solo índice extendido | Mueve cursor (EMA α=0.25) |
| PINCH | Pulgar + índice juntos (dist < 0.07) | Click izquierdo |
| RIGHT_CLICK | Pulgar + medio juntos (dist < 0.07) | Click derecho |
| SCROLL | Índice + medio arriba, anular + meñique abajo | Scroll vertical |
| PALM | 5 dedos abiertos | Captura de pantalla → Desktop |
| THUMB_UP | Solo pulgar arriba | Toggle mute/unmute Jarvis (`player.mute_callback`) |
| THREE_FINGERS | Índice + medio + anular arriba | Siguiente canción (`nexttrack`) |
| PINKY_UP | Solo meñique arriba | Play / Pause (`playpause`) |
| FIST (2 s) | Puño cerrado 2 segundos | Detiene control + notifica en log |

**Notas de implementación:**
- Modelo `hand_landmarker.task` (~9 MB) se descarga automáticamente en `config/hand_landmarker.task`.
- El preview OpenCV corre en el mismo daemon thread (no bloquea Tkinter ni asyncio).
- THUMB_UP llama a `player.mute_callback(bool)` — el mismo mecanismo que el botón de la UI.
- Al detenerse (fist o stop), restaura el micrófono si estaba silenciado por gesture.
- Cooldowns: click 0.6 s, screenshot 2 s, media 1.5 s, mute toggle 1.0 s.

### Game Updater (`actions/game_updater.py`) — from Mark-XXXV
Steam and Epic Games integration. Parses app manifests, queries the Steam registry, automates update dialogs via pywinauto/pyautogui.
Triggered with `game_updater({"action": "update"})` etc.

| Action | Description |
|---|---|
| `update` | Update all/specific Steam or Epic games |
| `install` | Install a game by name or Steam AppID |
| `list` | List all installed games |
| `download_status` | Check current download progress |
| `schedule` | Schedule daily auto-updates |
| `cancel_schedule` | Remove scheduled task |
| `schedule_status` | Check if a schedule exists |

**Notes:** Requires `pywinauto`. The `shutdown_when_done` flag powers off the PC when the download finishes.

### Enhanced Memory System (`memory/memory_manager.py`) — merged from Mark-XXXV
Memory now has 6 categories: `identity`, `preferences`, **`projects`**, `relationships`, **`wishes`**, `notes`.
- Memory entries store an `updated` date stamp.
- `format_memory_for_prompt` produces a richer grouped prompt section (up to 2000 chars).
- `remember(key, value, category)` / `forget(key, category)` helper functions available.
- Stage-1 YES/NO check is broader and catches projects, favorites, wishes in addition to identity facts.

### F4 Mute Shortcut (`ui.py`) — from Mark-XXXV
Press **F4** anywhere in the JARVIS window to toggle microphone mute (same as clicking the MUTE button).
