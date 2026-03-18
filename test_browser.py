# test_browser.py
import asyncio
from actions.browser_control import browser_control

class MockUI:
    def write_log(self, text):
        print(f"UI Log: {text}")

def test_browser():
    ui = MockUI()
    params = {"action": "search", "query": "python programming"}
    print("Executing browser search...")
    try:
        # browser_control is synchronous, it starts its own thread
        result = browser_control(parameters=params, player=ui)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Browser control failed: {e}")

if __name__ == "__main__":
    test_browser()
