# combine_5UTR_AB.py
# Combines the two already-exported PNGs (Part A on top, Part B on bottom)
# without modifying either plotting script.
# Alignment: uses the LEFT EDGE of visible content (not centering).

import os

OUT_DIR = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Luciferase Unit\final figures"

# ---- input PNGs (from your existing scripts) ----
A_PNG = os.path.join(OUT_DIR, "IL-33_5-UTR second part A.png")           # Part A output
B_PNG = os.path.join(OUT_DIR, "IL-33_5-UTR_second_part_B.png")    # Part B output

# ---- outputs ----
OUT_PNG = os.path.join(OUT_DIR, "IL-33_5-UTR second.png")
OUT_PDF = os.path.join(OUT_DIR, "IL-33_5-UTR second.pdf")

GAP_PX = 0  # optional vertical gap between A and B (try 20–40 if you want spacing)

try:
    from PIL import Image
    import numpy as np

    if not os.path.exists(A_PNG):
        raise FileNotFoundError(f"Missing Part A PNG: {A_PNG}")
    if not os.path.exists(B_PNG):
        raise FileNotFoundError(f"Missing Part B PNG: {B_PNG}")

    imgA = Image.open(A_PNG).convert("RGBA")
    imgB = Image.open(B_PNG).convert("RGBA")

    def content_left_x(img, white_thresh=250):
        """
        Detect the left-most x coordinate of non-white, non-transparent content.
        This aligns the actual plotted content rather than the PNG canvas.
        """
        arr = np.array(img)
        rgb = arr[..., :3]
        alpha = arr[..., 3]

        # Content pixels: alpha present AND not near-white
        mask = (alpha > 0) & (np.min(rgb, axis=2) < white_thresh)

        if not mask.any():
            return 0  # fallback if nothing detected

        ys, xs = np.where(mask)
        return int(xs.min())

    leftA = content_left_x(imgA)
    leftB = content_left_x(imgB)

    # Align both so their content-left starts at the same x position
    target_left = max(leftA, leftB)
    xA = target_left - leftA
    xB = target_left - leftB

    W = max(xA + imgA.width, xB + imgB.width)
    H = imgA.height + GAP_PX + imgB.height

    white = (255, 255, 255, 255)
    canvas = Image.new("RGBA", (W, H), white)

    canvas.paste(imgA, (xA, 0), imgA)
    canvas.paste(imgB, (xB, imgA.height + GAP_PX), imgB)

    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(OUT_PNG, "PNG")
    canvas_rgb.save(OUT_PDF, "PDF", resolution=600)

    print("Saved combined outputs (content-left aligned):")
    print("PNG:", OUT_PNG)
    print("PDF:", OUT_PDF)

except ImportError:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    if not os.path.exists(A_PNG):
        raise FileNotFoundError(f"Missing Part A PNG: {A_PNG}")
    if not os.path.exists(B_PNG):
        raise FileNotFoundError(f"Missing Part B PNG: {B_PNG}")

    imgA = mpimg.imread(A_PNG)
    imgB = mpimg.imread(B_PNG)

    def to_rgba(img):
        if img.shape[-1] == 4:
            return img
        if img.shape[-1] == 3:
            alpha = np.ones((*img.shape[:2], 1), dtype=img.dtype)
            return np.concatenate([img, alpha], axis=-1)
        raise ValueError("Unsupported image format")

    imgA = to_rgba(imgA)
    imgB = to_rgba(imgB)

    W = max(imgA.shape[1], imgB.shape[1])

    def pad_center(img, W):
        h, w, c = img.shape
        if w == W:
            return img
        pad_left = (W - w) // 2
        pad_right = W - w - pad_left
        white = np.ones((h, W, c), dtype=img.dtype)
        white[..., :3] = 1.0
        white[..., 3] = 1.0
        white[:, pad_left:pad_left + w, :] = img
        return white

    imgA = pad_center(imgA, W)
    imgB = pad_center(imgB, W)

    gap = np.ones((GAP_PX, W, 4), dtype=imgA.dtype)
    gap[..., :3] = 1.0
    gap[..., 3] = 1.0

    combined = np.vstack([imgA, gap, imgB])

    plt.figure(figsize=(W / 300, combined.shape[0] / 300), dpi=300)
    plt.axis("off")
    plt.imshow(combined)
    plt.tight_layout(pad=0)
    plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.close()

    print("Saved combined PNG (fallback centering):", OUT_PNG)
