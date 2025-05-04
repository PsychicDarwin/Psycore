#!/usr/bin/env python3
"""
new_image_extraction.py
───────────────────────
Extract every figure (vector artwork + nearby labels / legend / caption) from a
PDF and save each as a cropped PDF page (or PNG).

Hard-wired paths:
  INPUT_PDF  = /Users/arun/Desktop/Darwin/Psycore/scripting/jupyter_testing/image_extract_new/22-036458-01_GIS_early_process_evaluation_Accessible_CLIENT_USE.pdf
  OUTPUT_DIR = /Users/arun/Desktop/Darwin/Psycore/scripting/jupyter_testing/image_extract_new/figures
"""

# ── std-lib
import sys
from pathlib import Path
from typing import List, Dict

# ── third-party
try:
    import fitz                              # PyMuPDF
except ImportError:
    sys.exit("❌  PyMuPDF missing – run  pip install pymupdf")

try:
    from tqdm import tqdm                    # progress bar
except ImportError:                          # fallback if tqdm not installed
    tqdm = lambda x, **k: x                  # type: ignore[assignment,arg-type]


# ───────────────────────────────────────────────────────────────────────────────
# USER SETTINGS
# ───────────────────────────────────────────────────────────────────────────────
INPUT_PDF  = Path("/Users/arun/Desktop/Darwin/Psycore/scripting/jupyter_testing/BDUK_Annual_Reports-Accounts_2024_-_Certified_copy.pdf")

OUTPUT_DIR = Path("/Users/arun/Desktop/Darwin/Psycore/scripting/jupyter_testing/"
                  "image_extract_new/figures")

VECTOR_PDF    = True          # True ➜ export cropped PDFs  ·  False ➜ PNGs
DPI           = 300           # raster DPI (PNG mode only)
MIN_OBJECTS   = 15            # cluster dismissal threshold
MARGIN_PT     = 3.0           # white-space added round graphic
LABEL_PAD     = 18.0          # how far away a text block may sit & still count
MAX_CHARS     = 350           # ignore text blocks longer than this (≃ paragraphs)


# ───────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ───────────────────────────────────────────────────────────────────────────────
def _inflate(rect: fitz.Rect, d: float) -> fitz.Rect:
    return fitz.Rect(rect.x0 - d, rect.y0 - d, rect.x1 + d, rect.y1 + d)


def _rect_union(rects: List[fitz.Rect]) -> fitz.Rect:
    r = rects[0]
    for rc in rects[1:]:
        r |= rc
    return r


def _touches(a: fitz.Rect, b: fitz.Rect, gap: float = 0.7) -> bool:
    return bool(_inflate(a, gap) & b)


def _cluster_rects(rects: List[fitz.Rect]) -> List[List[fitz.Rect]]:
    groups: List[List[fitz.Rect]] = []
    for r in rects:
        for g in groups:
            if any(_touches(r, rc) for rc in g):
                g.append(r)
                break
        else:
            groups.append([r])
    return groups


# ───────────────────────────────────────────────────────────────────────────────
# Figure extraction
# ───────────────────────────────────────────────────────────────────────────────
def _figure_bbox(page: fitz.Page,
                 draw_rects: List[fitz.Rect]) -> fitz.Rect:
    """
    Given the drawing-cluster rectangles on *page*, return a bounding box that
    also incorporates small text blocks (labels, legend, caption) that sit
    within LABEL_PAD pts.
    """
    # initial box around artwork only
    box = _rect_union(draw_rects)
    box = _inflate(box, MARGIN_PT)

    # examine every text block on the page
    text_dict: Dict = page.get_text("dict")
    for block in text_dict["blocks"]:
        if block["type"] != 0:     # type 0 = text
            continue
        chars = sum(len(span["text"]) for line in block["lines"]
                                        for span in line["spans"])
        if chars > MAX_CHARS:      # long paragraph – ignore
            continue
        rect = fitz.Rect(block["bbox"])
        near = _inflate(box, LABEL_PAD)
        if near & rect:            # intersects padded region → include
            box |= rect            # enlarge figure box

    return box


def extract_figures(pdf: Path, out_dir: Path) -> int:
    if not pdf.is_file():
        raise FileNotFoundError(pdf)

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    counter = 0

    for pno in tqdm(range(doc.page_count), desc="Pages"):
        page = doc[pno]
        drawings = page.get_drawings()
        if not drawings:
            continue

        # 1. cluster vector-drawing elements
        drects = [fitz.Rect(d["rect"]) for d in drawings]
        clusters = _cluster_rects(drects)

        # 2. export each sufficiently large cluster
        for cluster in clusters:
            if len(cluster) < MIN_OBJECTS:
                continue

            bbox = _figure_bbox(page, cluster)

            if VECTOR_PDF:
                # vector-preserving export
                mini = fitz.open()
                tgt = mini.new_page(width=bbox.width, height=bbox.height)
                tgt.show_pdf_page(fitz.Rect(0, 0, bbox.width, bbox.height),
                                  doc, pno, clip=bbox)
                fname = f"p{pno+1:03d}_fig{counter+1:02d}.pdf"
                mini.save(out_dir / fname)
                mini.close()
            else:
                # raster export
                zoom = DPI / 72.0
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                      clip=bbox, alpha=False)
                fname = f"p{pno+1:03d}_fig{counter+1:02d}.png"
                pix.save(out_dir / fname)

            counter += 1

    return counter


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main() -> None:
    try:
        n = extract_figures(INPUT_PDF, OUTPUT_DIR)
    except Exception as e:
        sys.exit(f"❌  Extraction failed: {e}")

    print(f"✓ Extracted {n} figure(s) → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()



# import fitz  # PyMuPDF
# from pathlib import Path
# from PIL import Image
# from io import BytesIO
# import numpy as np

# # === CONFIGURATION ===
# pdf_path = "/Users/arun/Desktop/Darwin/Psycore/scripting/jupyter_testing/image_extract_new/22-036458-01_GIS_early_process_evaluation_Accessible_CLIENT_USE.pdf"
# output_dir = Path("/Users/arun/Desktop/Darwin/Psycore/scripting/jupyter_testing/image_extract_new/figures")
# output_dir.mkdir(parents=True, exist_ok=True)

# # === PROCESSING ===
# doc = fitz.open(pdf_path)
# drawing_count = 0

# for page_num, page in enumerate(doc):
#     drawings = page.get_drawings()
#     if not drawings:
#         continue

#     for draw_index, path in enumerate(drawings):
#         rect = path.get("rect")
#         if rect is None:
#             continue

#         pix = page.get_pixmap(clip=rect, dpi=300)
#         img = Image.open(BytesIO(pix.tobytes("ppm")))

#         gray = img.convert("L")
#         if np.array(gray).std() < 5:
#             continue  # Skip mostly blank

#         output_path = output_dir / f"vector_page{page_num+1}_drawing{draw_index+1}.jpg"
#         img.convert("RGB").save(output_path, "JPEG")
#         drawing_count += 1

# doc.close()
# print(f"✅ Extracted {drawing_count} vector graphic(s).")
