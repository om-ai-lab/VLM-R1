# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the image-size preprocessing helper.

Run with::

    PYTHONPATH=src python3 -m pytest src/tests/test_image_utils.py
"""

import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from open_r1.utils.image_utils import ensure_min_image_size  # noqa: E402


def test_normal_image_is_left_untouched():
    """Normal images must not be resized (and must not raise)."""
    img = Image.new("RGB", (100, 80))
    result = ensure_min_image_size(img)
    assert result is img
    assert result.size == (100, 80)


def test_small_image_is_upscaled_preserving_aspect_ratio():
    w, h = 20, 40
    img = Image.new("RGB", (w, h))
    result = ensure_min_image_size(img)
    assert result is not img
    assert result.size[0] == 28
    assert result.size[1] == int(h * (28 / w))


def test_small_image_wider_than_tall():
    w, h = 40, 20
    img = Image.new("RGB", (w, h))
    result = ensure_min_image_size(img)
    assert result.size[1] == 28
    assert result.size[0] == int(w * (28 / h))


def test_exact_boundary_is_unchanged():
    img = Image.new("RGB", (28, 28))
    assert ensure_min_image_size(img) is img


def test_custom_min_size():
    img = Image.new("RGB", (10, 30))
    result = ensure_min_image_size(img, min_size=64)
    assert result.size == (64, int(30 * (64 / 10)))
