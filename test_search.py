# test_search.py
import sys
from pathlib import Path

# Add current directory to path if needed
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from duckduckgo_search import DDGS
    print("DDGS imported successfully.")
except ImportError:
    print("duckduckgo_search not found.")

def test_ddg():
    print("Testing DuckDuckGo...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text("python programming", max_results=3))
            print(f"Results: {len(results)}")
            for r in results:
                print(f" - {r.get('title')}")
    except Exception as e:
        print(f"DDG failed: {e}")

if __name__ == "__main__":
    test_ddg()
