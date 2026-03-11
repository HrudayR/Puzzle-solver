import numpy as np
from PIL import Image, ImageDraw
import math
import random
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import argparse


# ---------------------------------------------------------------------------
# Bezier helper
# ---------------------------------------------------------------------------

def cubic_bezier(p0, p1, p2, p3, steps=24):
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
        y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts


# ---------------------------------------------------------------------------
# Tab edge builder
# ---------------------------------------------------------------------------

def tab_edge(x0, y0, x1, y1, direction, tab_ratio=0.28):
    if direction == 0:
        return [(x0, y0), (x1, y1)]

    dx, dy   = x1 - x0, y1 - y0
    length   = math.hypot(dx, dy)
    ux, uy   = dx / length, dy / length
    nx, ny   = -uy, ux

    tab_h    = length * tab_ratio * direction
    neck_w   = length * 0.10

    def pt(t, n=0):
        return (x0 + ux*length*t + nx*n,
                y0 + uy*length*t + ny*n)

    A    = pt(0.00)
    B    = pt(0.30)
    NL   = pt(0.38,  tab_h * 0.05)
    HL   = pt(0.38,  tab_h)
    HR   = pt(0.62,  tab_h)
    NR   = pt(0.62,  tab_h * 0.05)
    D    = pt(0.70)
    E    = pt(1.00)

    pts  = [A]
    pts += cubic_bezier(A, B, B, NL, steps=12)[1:]
    pts += cubic_bezier(NL, (NL[0]+nx*tab_h*0.6, NL[1]+ny*tab_h*0.6),
                            (HL[0]-ux*neck_w,    HL[1]-uy*neck_w),
                            HL, steps=14)[1:]
    pts += cubic_bezier(HL, pt(0.50, tab_h), pt(0.50, tab_h), HR, steps=10)[1:]
    pts += cubic_bezier(HR, (HR[0]+ux*neck_w,    HR[1]+uy*neck_w),
                            (NR[0]+nx*tab_h*0.6, NR[1]+ny*tab_h*0.6),
                            NR, steps=14)[1:]
    pts += cubic_bezier(NR, D, D, E, steps=12)[1:]
    return pts


# ---------------------------------------------------------------------------
# Deterministic tab layout
# ---------------------------------------------------------------------------

def make_tab_grid(rows, cols, seed=42):
    rng = random.Random(seed)
    h_tabs = [[rng.choice([1, -1]) for _ in range(cols)] for _ in range(rows - 1)]
    v_tabs = [[rng.choice([1, -1]) for _ in range(cols - 1)] for _ in range(rows)]
    return h_tabs, v_tabs


# ---------------------------------------------------------------------------
# Full piece outline (local coordinates, tile = [0,0]->[pw,ph])
# ---------------------------------------------------------------------------

def piece_outline(r, c, pw, ph, rows, cols, h_tabs, v_tabs):
    top    = 0 if r == 0        else -h_tabs[r-1][c]
    bottom = 0 if r == rows-1   else  h_tabs[r][c]
    left   = 0 if c == 0        else -v_tabs[r][c-1]
    right  = 0 if c == cols-1   else  v_tabs[r][c]

    pts = []
    pts += tab_edge(0,  0,  pw, 0,  top   )[:-1]
    pts += tab_edge(pw, 0,  pw, ph, right )[:-1]
    pts += tab_edge(pw, ph, 0,  ph, bottom)[:-1]
    pts += tab_edge(0,  ph, 0,  0,  left  )[:-1]
    return pts


# ---------------------------------------------------------------------------
# Render one piece onto a padded canvas
# ---------------------------------------------------------------------------

def render_piece(img_array, r, c, pw, ph, rows, cols, h_tabs, v_tabs):
    pad = int(max(pw, ph) * 0.35)

    cw = pw + 2 * pad
    ch = ph + 2 * pad

    mask = Image.new('L', (cw, ch), 0)
    draw = ImageDraw.Draw(mask)
    outline = piece_outline(r, c, pw, ph, rows, cols, h_tabs, v_tabs)
    shifted = [(x + pad, y + pad) for x, y in outline]
    draw.polygon(shifted, fill=255)

    img_h, img_w = img_array.shape[:2]
    sx1 = max(0, c * pw - pad);  sx2 = min(img_w, c * pw + pw + pad)
    sy1 = max(0, r * ph - pad);  sy2 = min(img_h, r * ph + ph + pad)

    src_crop = img_array[sy1:sy2, sx1:sx2]

    dst_x = pad - (c * pw - sx1)
    dst_y = pad - (r * ph - sy1)

    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
    sh, sw = src_crop.shape[:2]
    canvas[dst_y:dst_y+sh, dst_x:dst_x+sw, :3] = src_crop[:, :, :3]
    canvas[dst_y:dst_y+sh, dst_x:dst_x+sw,  3] = 255

    canvas[:, :, 3] = np.array(mask)

    return canvas, pad


# ---------------------------------------------------------------------------
# Normalize all pieces to the same canvas size
# ---------------------------------------------------------------------------

def normalize_piece_sizes(pieces):
    """
    Pad all pieces to the same canvas size (max_w x max_h), centering each
    piece's content. Mutates pieces in-place and stores 'canvas_offset' so
    that assembly previews can correctly re-position each piece.
    """
    max_w = max(p['piece_img'].shape[1] for p in pieces)
    max_h = max(p['piece_img'].shape[0] for p in pieces)

    for piece in pieces:
        h, w = piece['piece_img'].shape[:2]
        x_off = (max_w - w) // 2
        y_off = (max_h - h) // 2

        if w == max_w and h == max_h:
            piece['canvas_offset'] = (0, 0)
            continue

        padded = np.zeros((max_h, max_w, 4), dtype=np.uint8)
        padded[y_off:y_off+h, x_off:x_off+w] = piece['piece_img']
        piece['piece_img'] = padded
        piece['canvas_offset'] = (x_off, y_off)

    return pieces, max_w, max_h


# ---------------------------------------------------------------------------
# Shattered / Voronoi polygon pieces
# ---------------------------------------------------------------------------

def clip_polygon_to_box(polygon, x0, y0, x1, y1):
    def inside(p, edge):
        if edge == 0: return p[0] >= x0
        if edge == 1: return p[0] <= x1
        if edge == 2: return p[1] >= y0
        if edge == 3: return p[1] <= y1

    def intersect(p1, p2, edge):
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        if edge == 0:
            t = (x0 - p1[0]) / dx if dx else 0
        elif edge == 1:
            t = (x1 - p1[0]) / dx if dx else 0
        elif edge == 2:
            t = (y0 - p1[1]) / dy if dy else 0
        else:
            t = (y1 - p1[1]) / dy if dy else 0
        return (p1[0] + t*dx, p1[1] + t*dy)

    output = list(polygon)
    for edge in range(4):
        if not output:
            return []
        inp = output
        output = []
        for i in range(len(inp)):
            cur  = inp[i]
            prev = inp[i-1]
            if inside(cur, edge):
                if not inside(prev, edge):
                    output.append(intersect(prev, cur, edge))
                output.append(cur)
            elif inside(prev, edge):
                output.append(intersect(prev, cur, edge))
    return output


def create_shattered(image_path, num_pieces, output_dir='puzzle_shattered', seed=42):
    """
    Shatter an image into irregular Voronoi polygon pieces.
    All saved pieces share the same canvas dimensions (max bounding box),
    with each piece centered and surrounded by transparency.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    img       = Image.open(image_path).convert('RGBA')
    img_array = np.array(img)
    img_w, img_h = img.width, img.height

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    grid_cols = math.ceil(math.sqrt(num_pieces * img_w / img_h))
    grid_rows = math.ceil(num_pieces / grid_cols)
    cell_w    = img_w / grid_cols
    cell_h    = img_h / grid_rows

    points = []
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            jitter = 0.65
            px = (gc + 0.5 + rng.uniform(-jitter/2, jitter/2)) * cell_w
            py = (gr + 0.5 + rng.uniform(-jitter/2, jitter/2)) * cell_h
            px = max(cell_w*0.1, min(img_w - cell_w*0.1, px))
            py = max(cell_h*0.1, min(img_h - cell_h*0.1, py))
            points.append([px, py])

    margin = max(img_w, img_h) * 3
    for x in [-margin, img_w/2, img_w+margin]:
        for y in [-margin, img_h/2, img_h+margin]:
            if not (0 < x < img_w and 0 < y < img_h):
                points.append([x, y])

    pts_array = np.array(points)
    vor = Voronoi(pts_array)

    n_real = grid_rows * grid_cols
    region_polys = []
    for point_idx in range(n_real):
        region_idx = vor.point_region[point_idx]
        region     = vor.regions[region_idx]
        if not region or -1 in region:
            poly = _infinite_region_polygon(vor, point_idx, img_w, img_h)
        else:
            poly = [tuple(vor.vertices[v]) for v in region]

        clipped = clip_polygon_to_box(poly, 0, 0, img_w - 1, img_h - 1)
        if len(clipped) >= 3:
            region_polys.append((point_idx, clipped))

    # --- First pass: build all piece arrays at their natural bounding-box size ---
    pieces = []
    for piece_id, (point_idx, poly) in enumerate(region_polys):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bx1, by1 = max(0, int(min(xs))),       max(0, int(min(ys)))
        bx2, by2 = min(img_w, int(max(xs))+1), min(img_h, int(max(ys))+1)

        bw = bx2 - bx1
        bh = by2 - by1
        if bw <= 0 or bh <= 0:
            continue

        mask = Image.new('L', (bw, bh), 0)
        draw = ImageDraw.Draw(mask)
        local_poly = [(x - bx1, y - by1) for x, y in poly]
        draw.polygon(local_poly, fill=255)

        src = img_array[by1:by2, bx1:bx2].copy()
        piece_array = np.zeros((bh, bw, 4), dtype=np.uint8)
        piece_array[:, :, :3] = src[:, :, :3]
        piece_array[:, :,  3] = np.array(mask)

        pieces.append({
            'id':        piece_id,
            'path':      os.path.join(output_dir, f'piece_{piece_id:03d}.png'),
            'bbox':      (bx1, by1, bx2, by2),
            'poly':      poly,
            'piece_img': piece_array,
            'pad':       0,
            'canvas_offset': (0, 0),
        })

    # --- Second pass: normalize all pieces to the same canvas size ---
    pieces, max_w, max_h = normalize_piece_sizes(pieces)

    # --- Save normalized pieces ---
    for piece in pieces:
        Image.fromarray(piece['piece_img'], 'RGBA').save(piece['path'])

    print(f"Generated {len(pieces)} shattered pieces -> '{output_dir}' "
          f"(uniform canvas {max_w}x{max_h})")
    return pieces


def _infinite_region_polygon(vor, point_idx, img_w, img_h):
    margin = max(img_w, img_h) * 2
    poly = [(-margin, -margin), (img_w+margin, -margin),
            (img_w+margin, img_h+margin), (-margin, img_h+margin)]

    seed = vor.points[point_idx]
    for other_idx in range(len(vor.points)):
        if other_idx == point_idx:
            continue
        other = vor.points[other_idx]
        mid   = (seed + other) / 2
        diff  = other - seed
        poly  = _clip_halfplane(poly, mid, diff)
        if len(poly) < 3:
            break

    return [tuple(p) for p in poly]


def _clip_halfplane(polygon, mid, normal):
    def inside(p):
        return (p[0]-mid[0])*normal[0] + (p[1]-mid[1])*normal[1] <= 0

    def intersect(p1, p2):
        d1 = (p1[0]-mid[0])*normal[0] + (p1[1]-mid[1])*normal[1]
        d2 = (p2[0]-mid[0])*normal[0] + (p2[1]-mid[1])*normal[1]
        t  = d1 / (d1 - d2) if (d1 - d2) != 0 else 0
        return (p1[0] + t*(p2[0]-p1[0]), p1[1] + t*(p2[1]-p1[1]))

    out = []
    for i in range(len(polygon)):
        cur, prev = polygon[i], polygon[i-1]
        if inside(cur):
            if not inside(prev):
                out.append(intersect(prev, cur))
            out.append(cur)
        elif inside(prev):
            out.append(intersect(prev, cur))
    return out


def preview_assembled_shattered(pieces, img_w, img_h,
                                 save_path='preview_shattered_assembled.png'):
    """
    Composite shattered pieces back onto canvas using canvas_offset to correctly
    account for the uniform-size padding added during normalization.
    """
    canvas = Image.new('RGBA', (img_w, img_h), (180, 180, 180, 255))
    for piece in pieces:
        bx1, by1     = piece['bbox'][0], piece['bbox'][1]
        x_off, y_off = piece['canvas_offset']
        p = Image.fromarray(piece['piece_img'], 'RGBA')
        # Subtract the centering offset so pixels land in their original position
        canvas.paste(p, (bx1 - x_off, by1 - y_off), p)
    canvas.save(save_path)
    print(f"Assembled shattered preview -> '{save_path}'")
    return canvas


def preview_grid_shattered(pieces, save_path='preview_shattered_grid.png',
                            bg=(30, 30, 30, 255)):
    """Show shattered pieces in a grid -- all cells are the same size."""
    if not pieces:
        return
    n  = len(pieces)
    nc = math.ceil(math.sqrt(n))
    nr = math.ceil(n / nc)

    # All pieces are already the same size after normalization
    cell_w = pieces[0]['piece_img'].shape[1]
    cell_h = pieces[0]['piece_img'].shape[0]
    gap = max(6, cell_w // 15)

    cw = nc * (cell_w + gap) + gap
    ch = nr * (cell_h + gap) + gap
    canvas = Image.new('RGBA', (cw, ch), bg)

    for i, piece in enumerate(pieces):
        col_i = i % nc
        row_i = i // nc
        cx = gap + col_i * (cell_w + gap)
        cy = gap + row_i * (cell_h + gap)
        p = Image.fromarray(piece['piece_img'], 'RGBA')
        canvas.paste(p, (cx, cy), p)

    canvas.save(save_path)
    print(f"Grid shattered preview -> '{save_path}'")


# ---------------------------------------------------------------------------
# Main puzzle creator
# ---------------------------------------------------------------------------

def create_jigsaw(image_path, num_pieces, shape_type='curved',
                  output_dir='puzzle_pieces', seed=42):
    """
    Create interlocking jigsaw puzzle pieces from an image.
    All pieces are saved at the same canvas size.
    """
    img       = Image.open(image_path).convert('RGBA')
    img_array = np.array(img)

    rows = max(1, int(math.sqrt(num_pieces)))
    cols = max(1, math.ceil(num_pieces / rows))
    pw   = img.width  // cols
    ph   = img.height // rows

    h_tabs, v_tabs = make_tab_grid(rows, cols, seed=seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pieces = []
    for piece_id, (r, c) in enumerate((r, c)
                                       for r in range(rows)
                                       for c in range(cols)):
        if shape_type == 'regular':
            x1, y1 = c * pw, r * ph
            x2, y2 = min(x1+pw, img.width), min(y1+ph, img.height)
            piece_img = img.crop((x1, y1, x2, y2))
            piece_array = np.array(piece_img)
            pad = 0
        else:
            piece_array, pad = render_piece(
                img_array, r, c, pw, ph, rows, cols, h_tabs, v_tabs)
            x1, y1 = c * pw, r * ph
            x2, y2 = min(x1+pw, img.width), min(y1+ph, img.height)

        pieces.append({
            'id':        piece_id,
            'path':      os.path.join(output_dir, f'piece_{piece_id:03d}.png'),
            'bbox':      (x1, y1, x2, y2),
            'piece_img': piece_array,
            'grid_pos':  (r, c),
            'pad':       pad,
            'pw': pw, 'ph': ph,
            'canvas_offset': (0, 0),
        })

    # Normalize to uniform canvas size (handles edge pieces that may be slightly
    # smaller due to integer division of image dimensions)
    pieces, max_w, max_h = normalize_piece_sizes(pieces)

    for piece in pieces:
        Image.fromarray(piece['piece_img'], 'RGBA').save(piece['path'])

    print(f"Generated {len(pieces)} puzzle pieces -> '{output_dir}' "
          f"(uniform canvas {max_w}x{max_h})")
    return pieces


# ---------------------------------------------------------------------------
# Preview: composite pieces back onto canvas to verify they tile perfectly
# ---------------------------------------------------------------------------

def preview_assembled(pieces, save_path='preview_assembled.png'):
    """
    Stamp all pieces back in place -- should perfectly reconstruct the original.
    Uses canvas_offset to account for uniform-size padding.
    """
    if not pieces:
        return
    pw  = pieces[0]['pw']
    ph  = pieces[0]['ph']
    img_w = max(p['bbox'][2] for p in pieces)
    img_h = max(p['bbox'][3] for p in pieces)

    canvas = Image.new('RGBA', (img_w, img_h), (180, 180, 180, 255))
    for piece in sorted(pieces, key=lambda p: p['grid_pos']):
        r, c         = piece['grid_pos']
        x_off, y_off = piece['canvas_offset']
        piece_pil    = Image.fromarray(piece['piece_img'], 'RGBA')
        paste_x      = c * pw - piece['pad'] - x_off
        paste_y      = r * ph - piece['pad'] - y_off
        canvas.paste(piece_pil, (paste_x, paste_y), piece_pil)

    canvas.save(save_path)
    print(f"Assembled preview -> '{save_path}'")
    return canvas


def preview_grid(pieces, save_path='preview_grid.png'):
    """Show all pieces laid out in a grid -- all cells are the same size."""
    if not pieces:
        return
    n  = len(pieces)
    nc = math.ceil(math.sqrt(n))
    nr = math.ceil(n / nc)

    # All pieces are already the same size after normalization
    cell_w = pieces[0]['piece_img'].shape[1]
    cell_h = pieces[0]['piece_img'].shape[0]
    gap = max(4, cell_w // 20)

    cw = nc * (cell_w + gap) + gap
    ch = nr * (cell_h + gap) + gap
    canvas = Image.new('RGBA', (cw, ch), (40, 40, 40, 255))

    for i, piece in enumerate(pieces):
        col_i = i % nc
        row_i = i // nc
        x = gap + col_i * (cell_w + gap)
        y = gap + row_i * (cell_h + gap)
        p = Image.fromarray(piece['piece_img'], 'RGBA')
        canvas.paste(p, (x, y), p)

    canvas.save(save_path)
    print(f"Grid preview -> '{save_path}'")


# ---------------------------------------------------------------------------
# Derive a deterministic seed from an image path
# ---------------------------------------------------------------------------

def seed_from_image(image_path):
    """
    Produce a stable integer seed derived from the image filename.
    This means every unique filename gets a unique cut pattern, but
    re-running on the same file always yields the same pieces.
    """
    name = os.path.basename(image_path)
    return hash(name) & 0x7FFFFFFF   # positive 31-bit int


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Puzzle Generator: Create jigsaw or shattered pieces.")

    parser.add_argument("image", help="Path to the source image file")
    parser.add_argument("-n", "--num_pieces", type=int, default=20,
                        help="Approximate number of pieces to generate (default: 20)")
    parser.add_argument("-s", "--style", choices=['curved', 'shattered', 'regular'], default='curved',
                        help="Puzzle style: 'curved' (jigsaw), 'shattered' (Voronoi), or 'regular' (rectangles)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for deterministic piece shapes "
                             "(default: derived from image filename so each image gets unique cuts)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable generation of assembly and grid preview images")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File {args.image} not found.")
        exit(1)

    # Use caller-supplied seed if given; otherwise derive one from the filename
    # so that different images always produce different cut patterns.
    seed = args.seed if args.seed is not None else seed_from_image(args.image)

    img_tmp = Image.open(args.image)
    w, h = img_tmp.size
    output_dir_name = f"./{args.style}_{args.num_pieces}"

    print(f"Processing '{args.image}' ({w}x{h}) into {args.num_pieces} {args.style} pieces... (seed={seed})")

    if args.style == 'shattered':
        pieces = create_shattered(args.image, args.num_pieces,
                                  output_dir=output_dir_name + '/pieces', seed=seed)
        if not args.no_preview:
            preview_assembled_shattered(pieces, w, h,
                                        save_path=output_dir_name + '/preview_assembled.png')
            preview_grid_shattered(pieces, save_path=output_dir_name + '/preview_grid.png')
    else:
        pieces = create_jigsaw(args.image, args.num_pieces, shape_type=args.style,
                               output_dir=output_dir_name + '/pieces', seed=seed)
        if not args.no_preview:
            preview_assembled(pieces, save_path=output_dir_name + '/preview_assembled.png')
            preview_grid(pieces, save_path=output_dir_name + '/preview_grid.png')

    print("Done!")