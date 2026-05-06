import PIL.Image


def resize_image_min_size(image: PIL.Image.Image, min_size: int = 28) -> PIL.Image.Image:
    """Resize an image to ensure its shortest side is at least ``min_size``."""
    width, height = image.size
    if width >= min_size and height >= min_size:
        return image

    if width < height:
        new_width = min_size
        new_height = int(height * (min_size / width))
    else:
        new_height = min_size
        new_width = int(width * (min_size / height))

    return image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
