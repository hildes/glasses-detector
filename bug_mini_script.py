import os
import sys
from PIL import Image

# Make the package importable when running this script directly
repo_root = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(repo_root, "src"))

from glasses_detector.segmenter import GlassesSegmenter


def main():
    img_path = "data/demo/6_rect_not_square.png"

    # Load image and print its size (W, H)
    img = Image.open(img_path).convert("RGB")
    print(f"Input image size (W, H): {img.size}")
    # Input image size (W, H): (705, 803)

    w, h = img.size
    print(f"Input image size (H, W): {(h, w)}")
    # Input image size (H, W): (803, 705) <-- golden truth: height is larger than width

    # Instantiate the segmenter without downloading weights
    seg = GlassesSegmenter(weights=False, device="cpu")

    # Predict in two formats with no output_size
    mask = seg.predict(img_path, format="mask")   # PIL Image (W, H)
    proba = seg.predict(img_path, format="proba")  # torch.Tensor

    # Print dimensions
    print(f'"mask" output size (W, H): {mask.size}')
    # "mask" output size (W, H): (705, 803)

    # Tensor shape is printed as (H, W)
    print(f'"proba" output shape (H, W): {tuple(proba.shape)}')
    # BEFORE FIX: "proba" output shape (H, W): (705, 803)
    # AFTER FIX: "proba" output shape (H, W): (803, 705)

    proba = seg.predict(img_path, format="proba", output_size=(w, h))
    print(f'"proba" with output_size (H, W): {tuple(proba.shape)}')
    # BEFORE FIX: "proba" with output_size (H, W): (705, 803)
    # AFTER FIX: "proba" with output_size (H, W): (803, 705)


if __name__ == "__main__":
    main()
