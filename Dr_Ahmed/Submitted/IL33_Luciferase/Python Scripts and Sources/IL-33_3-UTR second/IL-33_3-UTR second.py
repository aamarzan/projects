# Combines the two already-exported PNGs (Part A on top, Part B on bottom)
# without modifying either plotting script.
# Fix: align panels using the LEFT EDGE of their visible content (not center).

import os

OUT_DIR = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Luciferase Unit\final figures"

A_PNG = os.path.join(OUT_DIR, "IL-33_3-UTR second part A.png")  # from Part A
B_PNG = os.path.join(OUT_DIR, "IL-33_3-UTR second part B.png")  # from Part B

OUT_PNG = os.path.join(OUT_DIR, "IL-33_3-UTR second.png")
OUT_PDF = os.path.join(OUT_DIR, "IL-33_3-UTR second.pdf")

GAP_PX = 0  # optional vertical gap between A and B

try:
    from PIL import Image
    import numpy as np

    if not os.path.exists(A_PNG):
        raise FileNotFoundError(f"Missing Part A PNG: {A_PNG}")
    if not os.path.exists(B_PNG):
        raise FileNotFoundError(f"Missing Part B PNG: {B_PNG}")

    imgA = Image.open(A_PNG).convert("RGBA")
    imgB = Image.open(B_PNG).convert("RGBA")

    def content_bbox_left(img, white_thresh=250):
        """
        Returns the left x of the non-white visible content.
        Works even if bbox_inches='tight' still leaves uneven padding.
        """
        arr = np.array(img)
        rgb = arr[..., :3]
        alpha = arr[..., 3]

        # content if not transparent and not near-white
        mask = (alpha > 0) & (np.min(rgb, axis=2) < white_thresh)
        if not mask.any():
            return 0

        ys, xs = np.where(mask)
        return int(xs.min())

    leftA = content_bbox_left(imgA)
    leftB = content_bbox_left(imgB)

    # Align both images so their content-left starts at the same x
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

    def pad_to_width(img, W):
        h, w, c = img.shape
        if w == W:
            return img
        pad_left = (W - w) // 2
        white = np.ones((h, W, c), dtype=img.dtype)
        white[..., :3] = 1.0
        white[..., 3] = 1.0
        white[:, pad_left:pad_left+w, :] = img
        return white

    imgA = pad_to_width(imgA, W)
    imgB = pad_to_width(imgB, W)

    gap = np.ones((GAP_PX, W, 4), dtype=imgA.dtype)
    gap[..., :3] = 1.0
    gap[..., 3] = 1.0

    combined = np.vstack([imgA, gap, imgB])

    plt.figure(figsize=(W/300, combined.shape[0]/300), dpi=300)
    plt.axis("off")
    plt.imshow(combined)
    plt.tight_layout(pad=0)
    plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.close()

    print("Saved combined PNG:", OUT_PNG)
