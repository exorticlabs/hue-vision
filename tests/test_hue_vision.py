"""
Unit and integration tests for Hue Vision multi-device perception and automation.
"""

import unittest
import numpy as np
from PIL import Image

from hue_vision.platforms import PlatformManager, WindowsAdapter, MacOSAdapter, LinuxAdapter, AndroidAdapter, UniversalCVAdapter
from hue_vision.cv_detector import CVDetector
from hue_vision.context_builder import ContextBuilder, ScreenContext, UIElement, WindowContext, UIGroup
from hue_vision.dom_extractor import RawDOMGroup, RawDOMElement
from hue_vision.tracker import LiveScreenTracker, ScreenDelta
from hue_vision.engine import VisionEngine


class TestPlatforms(unittest.TestCase):
    def test_platform_manager_resolution(self):
        pm = PlatformManager.get_instance()
        self.assertIsNotNone(pm.get_adapter())
        self.assertEqual(pm.get_adapter("android").platform_name, "android")
        self.assertEqual(pm.get_adapter("macos").platform_name, "macos")
        self.assertEqual(pm.get_adapter("linux").platform_name, "linux")
        self.assertEqual(pm.get_adapter("universal").platform_name, "universal_cv")

    def test_universal_adapter_capture(self):
        adapter = UniversalCVAdapter()
        img, sz = adapter.capture_screen()
        self.assertIsInstance(img, Image.Image)
        self.assertGreater(sz[0], 0)
        self.assertGreater(sz[1], 0)

    def test_android_adapter_instantiation(self):
        adapter = AndroidAdapter(device_serial="emulator-5554")
        self.assertEqual(adapter.device_serial, "emulator-5554")
        self.assertEqual(adapter.platform_name, "android")


class TestCVDetector(unittest.TestCase):
    def test_candidate_detection(self):
        # Create a synthetic image with a high-contrast button
        img = Image.new("RGB", (400, 300), color=(240, 240, 240))
        arr = np.array(img)
        # Draw a simulated dark button rectangle
        arr[100:150, 150:250] = [30, 30, 30]
        test_img = Image.fromarray(arr)

        candidates = CVDetector.detect_visual_candidates(test_img, [0, 0, 400, 300], min_size=10, max_boxes=20)
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        first = candidates[0]
        self.assertIn("bbox", first)
        self.assertEqual(len(first["bbox"]), 4)


class TestTrackerAndDelta(unittest.TestCase):
    def test_delta_computation(self):
        tracker = LiveScreenTracker()
        win_meta1 = {
            "title": "App Window 1",
            "process_name": "app.exe",
            "pid": 1234,
            "rect": [0, 0, 800, 600],
            "is_maximized": False,
            "hwnd": 100
        }
        ctx1 = ContextBuilder.build_hybrid_context(
            window_meta=win_meta1,
            screen_size=(800, 600),
            raw_groups=[],
            raw_elements=[],
            model_predictions=[],
            source="pure_cv"
        )
        d1 = tracker.update(ctx1)
        self.assertFalse(d1.has_changed)

        win_meta2 = dict(win_meta1, title="App Window 2")
        ctx2 = ContextBuilder.build_hybrid_context(
            window_meta=win_meta2,
            screen_size=(800, 600),
            raw_groups=[],
            raw_elements=[],
            model_predictions=[],
            source="pure_cv"
        )
        d2 = tracker.update(ctx2)
        self.assertTrue(d2.has_changed)
        self.assertTrue(d2.window_changed)


class TestVisionEngine(unittest.TestCase):
    def test_engine_singleton(self):
        e1 = VisionEngine.get_instance()
        e2 = VisionEngine.get_instance()
        self.assertIs(e1, e2)

    def test_live_perceive(self):
        engine = VisionEngine.get_instance()
        md = engine.perceive(target="device", max_elements=10)
        self.assertIsInstance(md, str)
        self.assertIn("Active Screen", md)


if __name__ == "__main__":
    unittest.main()
