import PIL.Image

from open_r1.utils.image import resize_image_min_size


def test_resize_image_min_size_keeps_large_images():
    image = PIL.Image.new("RGB", (224, 224))

    resized = resize_image_min_size(image, min_size=28)

    assert resized.size == (224, 224)


def test_resize_image_min_size_keeps_boundary_images():
    image = PIL.Image.new("RGB", (28, 28))

    resized = resize_image_min_size(image, min_size=28)

    assert resized.size == (28, 28)


def test_resize_image_min_size_expands_small_width():
    image = PIL.Image.new("RGB", (20, 50))

    resized = resize_image_min_size(image, min_size=28)

    assert resized.size == (28, 70)


def test_resize_image_min_size_expands_small_height():
    image = PIL.Image.new("RGB", (50, 20))

    resized = resize_image_min_size(image, min_size=28)

    assert resized.size == (70, 28)


def test_resize_image_min_size_expands_equal_small_sides():
    image = PIL.Image.new("RGB", (20, 20))

    resized = resize_image_min_size(image, min_size=28)

    assert resized.size == (28, 28)
