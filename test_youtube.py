# test_youtube.py
import sys
from unittest.mock import MagicMock

# Mocking modules that might not be in the environment or cause issues in headful mode
pyautogui_mock = MagicMock()
pyautogui_mock.size.return_value = (1920, 1080)
sys.modules['pyautogui'] = pyautogui_mock
sys.modules['numpy'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['PIL'] = MagicMock()

from actions.youtube_video import youtube_video

class MockPlayer:
    def write_log(self, text):
        print(f"[LOG] {text}")

def test_play_video():
    player = MockPlayer()
    params = {"action": "play", "query": "never gonna give you up"}
    
    print("Testing youtube_video(action='play')...")
    try:
        # This should call webbrowser.open and print logs
        result = youtube_video(parameters=params, player=player)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_play_video()
