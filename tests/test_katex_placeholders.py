"""Regression: math placeholders must survive Markdown underscore emphasis."""
from __future__ import annotations

import re
import unittest


def simulate_marked_underscore_strong(text: str) -> str:
    """Approximate marked/CommonMark treatment of __…__ as strong emphasis."""
    return re.sub(r"__([\s\S]+?)__", r"<strong>\1</strong>", text)


def extract_math_placeholders(body: str) -> tuple[str, list[str], list[str]]:
    """Mirror frontend/feedback/script.js renderMarkdownMath extraction order."""
    displays: list[str] = []
    inlines: list[str] = []

    def _block(match: re.Match[str]) -> str:
        idx = len(displays) + len(inlines)
        displays.append(match.group(1).strip())
        return f"%%MATH_BLOCK_{idx}%%"

    def _inline(match: re.Match[str]) -> str:
        idx = len(displays) + len(inlines)
        inlines.append(match.group(1).strip())
        return f"%%MATH_INLINE_{idx}%%"

    work = re.sub(r"\$\$([\s\S]+?)\$\$", _block, body)
    # Same lookbehind/lookahead intent as the JS inline regex.
    work = re.sub(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)", _inline, work)
    return work, displays, inlines


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

    def test_nash_equilibrium_seed_extracts_expected_math(self) -> None:
        from backend.routes.feedback import RESEARCH_SEED_QUESTIONS

        nash = next(
            q for q in RESEARCH_SEED_QUESTIONS if q["question_id"] == "research-nash-equilibrium"
        )
        body = nash["body_markdown"]
        work, displays, inlines = extract_math_placeholders(body)

        self.assertEqual(len(displays), 1)
        self.assertEqual(len(inlines), 3)
        self.assertIn("\\ge", displays[0])
        self.assertEqual(set(inlines), {"u_i", "s_i", "s_{-i}"})
        self.assertNotIn("$", work)

        after = simulate_marked_underscore_strong(work)
        self.assertIn("%%MATH_BLOCK_0%%", after)
        # Inline indices are 1,2,3 because block consumed index 0.
        self.assertIn("%%MATH_INLINE_1%%", after)
        self.assertIn("%%MATH_INLINE_2%%", after)
        self.assertIn("%%MATH_INLINE_3%%", after)


if __name__ == "__main__":
    unittest.main()
