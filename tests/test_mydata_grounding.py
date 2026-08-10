from __future__ import annotations

import unittest

from mydata_bench.grounding.sam3 import _clip_bbox_to_image


class SAM3BBoxTests(unittest.TestCase):
    def test_clips_subpixel_overshoot_at_image_edges(self) -> None:
        self.assertEqual(
            _clip_bbox_to_image([577.9, 336.7, 640.375, 405.1], 640, 480),
            [577.9, 336.7, 640.0, 405.1],
        )
        self.assertEqual(_clip_bbox_to_image([-0.25, 2, 10, 12], 640, 480), [0.0, 2.0, 10.0, 12.0])
        self.assertIsNone(_clip_bbox_to_image([641, 2, 642, 12], 640, 480))


if __name__ == "__main__":
    unittest.main()
