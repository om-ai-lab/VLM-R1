# SPDX-License-Identifier: Apache-2.0

"""Small image helpers that do not require the training stack.

Keeping this logic free of ``torch`` / ``transformers`` imports makes it
possible to unit-test the preprocessing behaviour without a GPU or the full
training environment.
"""

from PIL import Image

MIN_IMAGE_SIZE = 28


def ensure_min_image_size(
    img: Image.Image, min_size: int = MIN_IMAGE_SIZE
) -> Image.Image:
    """Upscale ``img`` when either dimension is smaller than ``min_size``.

    The aspect ratio is preserved by scaling the shorter side up to
    ``min_size``. Images that are already large enough are returned unchanged
    (same object, no copy and no ``resize`` call).

    This is a no-op for degenerate (zero-sized) images, which cannot be
    resized meaningfully.
    """
    w, h = img.size
    if w >= min_size and h >= min_size:
        return img
    if w <= 0 or h <= 0:
        return img

    if w < h:
        new_w = min_size
        new_h = int(h * (min_size / w))
    else:
        new_h = min_size
        new_w = int(w * (min_size / h))

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
