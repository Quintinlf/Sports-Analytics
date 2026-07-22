"""Regression: math placeholders must survive Markdown underscore emphasis."""
from __future__ import annotations

import re
import unittest


def simulate_marked_underscore_strong(text: str) -> str:
    """Approximate marked/CommonMark treatment of __…__ as strong emphasis."""
    return re.sub(r"__([\s\S]+?)__", r"<strong>\1</strong>", text)


class TestKatexPlaceholders(unittest.TestCase):
    def test_legacy_underscore_tokens_are_corrupted_by_markdown(self) -> None:
        legacy_block = "__MATH_BLOCK_0__"
        legacy_inline = "__MATH_INLINE_1__"
        corrupted_block = simulate_marked_underscore_strong(legacy_block)
        corrupted_inline = simulate_marked_underscore_strong(legacy_inline)
        self.assertNotEqual(corrupted_block, legacy_block)
        self.assertNotEqual(corrupted_inline, legacy_inline)
        self.assertIn("<strong>", corrupted_block)
        self.assertIn("<strong>", corrupted_inline)

    def test_inert_percent_tokens_survive_markdown_emphasis(self) -> None:
        # Contract mirrored by frontend/feedback/script.js MATH_*_TOKEN helpers.
        block = "%%MATH_BLOCK_0%%"
        inline = "%%MATH_INLINE_1%%"
        sample = f"Intro {block} and {inline} outro **bold**"
        after = simulate_marked_underscore_strong(sample)
        self.assertIn(block, after)
        self.assertIn(inline, after)
        self.assertEqual(after.count(block), 1)
        self.assertEqual(after.count(inline), 1)


if __name__ == "__main__":
    unittest.main()
