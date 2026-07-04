"""
Generates a handful of synthetic placeholder images into edge/camera_storage/ so the
pipeline is runnable end-to-end before you have real satellite imagery. These are NOT
real satellite photos and NOT training data — purely so run_edge_pipeline.py and the
Streamlit dashboard have something to chew on for a demo.

    python edge/generate_sample_frames.py
"""
import os
import random

from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = os.path.join(os.path.dirname(__file__), "camera_storage")


def make_overcast(size=256):
    """Bright, flat, low-contrast frame — simulates thick cloud cover."""
    base = random.randint(200, 235)
    img = Image.new("RGB", (size, size), (base, base, base + random.randint(-5, 10)))
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return img


def make_clear_ground(size=256):
    """Textured green/brown patchwork — simulates clear farmland/terrain."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    for _ in range(40):
        x0, y0 = random.randint(0, size), random.randint(0, size)
        w, h = random.randint(10, 40), random.randint(10, 40)
        color = random.choice([(60, 110, 40), (90, 130, 60), (120, 100, 70), (70, 90, 50)])
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=color)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def make_wildfire(size=256):
    """Ground scene with a bright orange/red hotspot — simulates an active fire."""
    img = make_clear_ground(size)
    draw = ImageDraw.Draw(img)
    cx, cy = random.randint(size // 4, 3 * size // 4), random.randint(size // 4, 3 * size // 4)
    for r, color in [(40, (255, 140, 0)), (25, (255, 60, 0)), (12, (255, 220, 100))]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def make_oil_spill(size=256):
    """Dark water scene with a dull sheen patch — simulates an oil slick on ocean."""
    img = Image.new("RGB", (size, size), (20, 50, 80))
    draw = ImageDraw.Draw(img)
    for _ in range(15):
        x0, y0 = random.randint(0, size), random.randint(0, size)
        w, h = random.randint(30, 80), random.randint(10, 25)
        draw.ellipse([x0, y0, x0 + w, y0 + h], fill=(35, 40, 35))
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img


GENERATORS = {
    "overcast": make_overcast,
    "clear_ground": make_clear_ground,
    "wildfire": make_wildfire,
    "oil_spill": make_oil_spill,
}


def main(n_per_class: int = 5):
    os.makedirs(OUT_DIR, exist_ok=True)
    count = 0
    for label, gen_fn in GENERATORS.items():
        for i in range(n_per_class):
            img = gen_fn()
            fname = f"{label}_{i:02d}.jpg"
            img.save(os.path.join(OUT_DIR, fname), quality=90)
            count += 1
    print(f"Wrote {count} synthetic sample frames to {OUT_DIR}")
    print("Remember: these are placeholders for pipeline testing, not real training data.")


if __name__ == "__main__":
    main()
