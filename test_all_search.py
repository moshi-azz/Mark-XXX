# test_all_search.py
import json
import sys
from pathlib import Path

def get_base_dir() -> Path:
    return Path(__file__).resolve().parent

BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

def _get_api_key() -> str:
    try:
        with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)["gemini_api_key"]
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None

def test_gemini(query):
    print(f"Testing Gemini Search for: {query}")
    try:
        from google import genai
        # Try a more standard model name if the current one fails
        models_to_try = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]
        
        api_key = _get_api_key()
        if not api_key: return

        client = genai.Client(api_key=api_key)
        
        for model in models_to_try:
            print(f"  Trying model: {model}")
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=query,
                    config={"tools": [{"google_search": {}}]}
                )
                text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
                if text.strip():
                    print(f"  [SUCCESS] with {model}: {text[:100]}...")
                    return
                else:
                    print(f"  [EMPTY] Empty response from {model}")
            except Exception as e:
                print(f"  [FAILED] with {model}: {e}")
    except Exception as e:
        print(f"Gemini test failed drastically: {e}")

def test_ddg(query):
    print(f"Testing DuckDuckGo for: {query}")
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # Try a different method if 'text' returns 0
            results = list(ddgs.text(query, max_results=3))
            print(f"  DDG text results: {len(results)}")
            if len(results) == 0:
                print("  Trying alternative DDG method...")
    except Exception as e:
        print(f"  DDG failed: {e}")

if __name__ == "__main__":
    q = "current weather in New York"
    test_gemini(q)
    test_ddg(q)
