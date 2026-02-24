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
    """
    Build points for one edge from (x0,y0) to (x1,y1).
    direction: +1 = tab sticks outward (outie), -1 = inward (innie), 0 = flat.

    IMPORTANT: tabs extend OUTSIDE the [0,pw]x[0,ph] tile box.
    This is what makes pieces interlock — the outie tab of one piece
    fills the innie cutout of its neighbour.
    """
    if direction == 0:
        return [(x0, y0), (x1, y1)]

    dx, dy   = x1 - x0, y1 - y0
    length   = math.hypot(dx, dy)
    ux, uy   = dx / length, dy / length       # unit along edge
    nx, ny   = -uy, ux                         # unit perpendicular (left of travel)

    tab_h    = length * tab_ratio * direction  # signed: + = outie
    neck_w   = length * 0.10                   # half-width of tab neck

    def pt(t, n=0):
        return (x0 + ux*length*t + nx*n,
                y0 + uy*length*t + ny*n)

    # Key points
    A    = pt(0.00)
    B    = pt(0.30)
    NL   = pt(0.38,  tab_h * 0.05)   # neck left base
    HL   = pt(0.38,  tab_h)           # head left
    HR   = pt(0.62,  tab_h)           # head right
    NR   = pt(0.62,  tab_h * 0.05)   # neck right base
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
    """
    Assign a random +1/-1 to every shared edge.
    h_tabs[r][c] = direction of the bottom edge of tile (r,c)  [rows-1 rows]
    v_tabs[r][c] = direction of the right  edge of tile (r,c)  [cols-1 cols]
    """
    rng = random.Random(seed)
    h_tabs = [[rng.choice([1, -1]) for _ in range(cols)] for _ in range(rows - 1)]
    v_tabs = [[rng.choice([1, -1]) for _ in range(cols - 1)] for _ in range(rows)]
    return h_tabs, v_tabs


# ---------------------------------------------------------------------------
# Full piece outline (local coordinates, tile = [0,0]->[pw,ph])
# ---------------------------------------------------------------------------

def piece_outline(r, c, pw, ph, rows, cols, h_tabs, v_tabs):
    top    = 0 if r == 0        else -h_tabs[r-1][c]   # opposite of row above's bottom
    bottom = 0 if r == rows-1   else  h_tabs[r][c]
    left   = 0 if c == 0        else -v_tabs[r][c-1]
    right  = 0 if c == cols-1   else  v_tabs[r][c]

    pts = []
    pts += tab_edge(0,  0,  pw, 0,  top   )[:-1]   # top:    L→R
    pts += tab_edge(pw, 0,  pw, ph, right )[:-1]   # right:  T→B
    pts += tab_edge(pw, ph, 0,  ph, bottom)[:-1]   # bottom: R→L
    pts += tab_edge(0,  ph, 0,  0,  left  )[:-1]   # left:   B→T
    return pts


# ---------------------------------------------------------------------------
# Render one piece onto a padded canvas
# ---------------------------------------------------------------------------

def render_piece(img_array, r, c, pw, ph, rows, cols, h_tabs, v_tabs):
    pad = int(max(pw, ph) * 0.35)   # large enough to contain any tab overhang

    cw = pw + 2 * pad
    ch = ph + 2 * pad

    # Build mask
    mask = Image.new('L', (cw, ch), 0)
    draw = ImageDraw.Draw(mask)
    outline = piece_outline(r, c, pw, ph, rows, cols, h_tabs, v_tabs)
    shifted = [(x + pad, y + pad) for x, y in outline]
    draw.polygon(shifted, fill=255)

    # Copy source pixels — grab a padded region of the original image
    img_h, img_w = img_array.shape[:2]
    sx1 = max(0, c * pw - pad);  sx2 = min(img_w, c * pw + pw + pad)
    sy1 = max(0, r * ph - pad);  sy2 = min(img_h, r * ph + ph + pad)

    src_crop = img_array[sy1:sy2, sx1:sx2]

    # Where this crop lands on the padded canvas
    dst_x = pad - (c * pw - sx1)
    dst_y = pad - (r * ph - sy1)

    canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
    sh, sw = src_crop.shape[:2]
    canvas[dst_y:dst_y+sh, dst_x:dst_x+sw, :3] = src_crop[:, :, :3]
    canvas[dst_y:dst_y+sh, dst_x:dst_x+sw,  3] = 255

    # Apply mask
    canvas[:, :, 3] = np.array(mask)

    return canvas, pad


# ---------------------------------------------------------------------------
# Shattered / Voronoi polygon pieces
# ---------------------------------------------------------------------------

def clip_polygon_to_box(polygon, x0, y0, x1, y1):
    """
    Sutherland–Hodgman algorithm: clip a polygon to an axis-aligned box.
    polygon: list of (x,y) tuples.  Returns clipped list (may be empty).
    """
    def inside(p, edge):
        # edge: 0=left, 1=right, 2=bottom, 3=top
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


def voronoi_finite_polygons(vor, img_w, img_h, margin=500):
    """
    Convert a scipy Voronoi diagram into finite polygons clipped to [0,img_w]×[0,img_h].
    Returns list of (region_index, polygon_points) pairs.
    """
    center   = np.array([img_w / 2, img_h / 2])
    ptp_bound = max(img_w, img_h) * 2

    # Map from ridge index to far point for infinite ridges
    ridge_dict = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_dict[(min(p1,p2), max(p1,p2))] = (v1, v2)

    new_vertices = list(vor.vertices)

    region_polys = []
    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region:
            continue

        # Reconstruct finite polygon, replacing -1 vertices with far points
        poly = []
        verts = list(region)
        for i, v in enumerate(verts):
            if v >= 0:
                poly.append(tuple(vor.vertices[v]))
            else:
                # Find the ridge that contains this -1 vertex
                # by checking adjacent vertices in the region
                v_prev = verts[i-1]
                v_next = verts[(i+1) % len(verts)]

                # Find two ridges sharing this infinite vertex
                tangents = []
                for (p1, p2), (rv1, rv2) in zip(vor.ridge_points, vor.ridge_vertices):
                    if point_idx not in (p1, p2):
                        continue
                    if -1 not in (rv1, rv2):
                        continue
                    other_p = p2 if p1 == point_idx else p1
                    t = vor.points[point_idx] - vor.points[other_p]
                    t /= np.linalg.norm(t)
                    tangents.append(t)

                if tangents:
                    tangent = np.mean(tangents, axis=0)
                else:
                    tangent = vor.points[point_idx] - center
                    norm = np.linalg.norm(tangent)
                    if norm > 0:
                        tangent /= norm

                # Normal pointing away from center
                midpoint = vor.points[point_idx]
                direction = np.sign(np.dot(midpoint - center, tangent))
                far_point = vor.vertices[v_prev if v_prev >= 0 else v_next] if (v_prev >= 0 or v_next >= 0) else midpoint
                # Use the finite neighbour vertex as anchor
                anchor = None
                for va in (v_prev, v_next):
                    if va >= 0:
                        anchor = vor.vertices[va]
                        break
                if anchor is None:
                    anchor = vor.points[point_idx]
                far = anchor + direction * tangent * ptp_bound
                poly.append(tuple(far))

        # Clip to image bounds
        clipped = clip_polygon_to_box(poly, 0, 0, img_w, img_h)
        if len(clipped) >= 3:
            region_polys.append((point_idx, clipped))

    return region_polys


def create_shattered(image_path, num_pieces, output_dir='puzzle_shattered', seed=42):
    """
    Shatter an image into irregular Voronoi polygon pieces.
    Every piece together perfectly reconstructs the original — no gaps, no overlaps.

    Returns list of piece dicts (same format as create_jigsaw).
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    img       = Image.open(image_path).convert('RGBA')
    img_array = np.array(img)
    img_w, img_h = img.width, img.height

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Generate Voronoi seed points ---
    # Scatter points randomly but with slight regularity (jittered grid looks more natural)
    grid_cols = math.ceil(math.sqrt(num_pieces * img_w / img_h))
    grid_rows = math.ceil(num_pieces / grid_cols)
    cell_w    = img_w / grid_cols
    cell_h    = img_h / grid_rows

    points = []
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            # Jitter within cell, biased toward center to avoid tiny slivers at border
            jitter = 0.65
            px = (gc + 0.5 + rng.uniform(-jitter/2, jitter/2)) * cell_w
            py = (gr + 0.5 + rng.uniform(-jitter/2, jitter/2)) * cell_h
            px = max(cell_w*0.1, min(img_w - cell_w*0.1, px))
            py = max(cell_h*0.1, min(img_h - cell_h*0.1, py))
            points.append([px, py])

    # Add mirror points far outside to bound all infinite ridges cleanly
    margin = max(img_w, img_h) * 3
    for x in [-margin, img_w/2, img_w+margin]:
        for y in [-margin, img_h/2, img_h+margin]:
            if not (0 < x < img_w and 0 < y < img_h):
                points.append([x, y])

    pts_array = np.array(points)
    vor = Voronoi(pts_array)

    # --- Get finite, clipped polygons for the real interior points only ---
    n_real = grid_rows * grid_cols
    region_polys = []
    for point_idx in range(n_real):
        region_idx = vor.point_region[point_idx]
        region     = vor.regions[region_idx]
        if not region or -1 in region:
            # Infinite region — use convex hull fallback clipped to box
            poly = _infinite_region_polygon(vor, point_idx, img_w, img_h)
        else:
            poly = [tuple(vor.vertices[v]) for v in region]

        clipped = clip_polygon_to_box(poly, 0, 0, img_w - 1, img_h - 1)
        if len(clipped) >= 3:
            region_polys.append((point_idx, clipped))

    # --- Render each piece ---
    pieces = []
    for piece_id, (point_idx, poly) in enumerate(region_polys):
        seed_pt = pts_array[point_idx]

        # Bounding box of polygon
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bx1, by1 = max(0, int(min(xs))),     max(0, int(min(ys)))
        bx2, by2 = min(img_w, int(max(xs))+1), min(img_h, int(max(ys))+1)

        bw = bx2 - bx1
        bh = by2 - by1
        if bw <= 0 or bh <= 0:
            continue

        # Build mask in bounding-box local coords
        mask = Image.new('L', (bw, bh), 0)
        draw = ImageDraw.Draw(mask)
        local_poly = [(x - bx1, y - by1) for x, y in poly]
        draw.polygon(local_poly, fill=255)

        # Crop image to bounding box and apply mask
        src = img_array[by1:by2, bx1:bx2].copy()
        piece_array = np.zeros((bh, bw, 4), dtype=np.uint8)
        piece_array[:, :, :3] = src[:, :, :3]
        piece_array[:, :,  3] = np.array(mask)

        piece_pil  = Image.fromarray(piece_array, 'RGBA')
        piece_path = os.path.join(output_dir, f'piece_{piece_id:03d}.png')
        piece_pil.save(piece_path)

        pieces.append({
            'id':        piece_id,
            'path':      piece_path,
            'bbox':      (bx1, by1, bx2, by2),   # position in original image
            'poly':      poly,                     # absolute polygon coords
            'piece_img': piece_array,
            'pad':       0,                        # no padding needed for voronoi
        })

    print(f"Generated {len(pieces)} shattered pieces → '{output_dir}'")
    return pieces


def _infinite_region_polygon(vor, point_idx, img_w, img_h):
    """
    Fallback for Voronoi regions with infinite ridges:
    build a large bounding polygon and clip it.
    Computes the half-plane intersection for this seed point.
    """
    # Use a large bounding box polygon and intersect with all half-planes
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
        # Clip poly to the half-plane: (p - mid) · diff <= 0
        poly  = _clip_halfplane(poly, mid, diff)
        if len(poly) < 3:
            break

    return [tuple(p) for p in poly]


def _clip_halfplane(polygon, mid, normal):
    """Clip polygon to half-plane: points where (p-mid)·normal <= 0."""
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
    """Composite shattered pieces back onto canvas — should recreate the original."""
    canvas = Image.new('RGBA', (img_w, img_h), (180, 180, 180, 255))
    for piece in pieces:
        bx1, by1 = piece['bbox'][0], piece['bbox'][1]
        p = Image.fromarray(piece['piece_img'], 'RGBA')
        canvas.paste(p, (bx1, by1), p)
    canvas.save(save_path)
    print(f"Assembled shattered preview → '{save_path}'")
    return canvas


def preview_grid_shattered(pieces, save_path='preview_shattered_grid.png',
                            bg=(30, 30, 30, 255)):
    """Show shattered pieces in a grid — each on its own dark background cell."""
    if not pieces:
        return
    n  = len(pieces)
    nc = math.ceil(math.sqrt(n))
    nr = math.ceil(n / nc)

    # Uniform cell size = max bounding box
    cell_w = max(p['piece_img'].shape[1] for p in pieces)
    cell_h = max(p['piece_img'].shape[0] for p in pieces)
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
        # Centre piece within its cell
        ox = (cell_w - p.width)  // 2
        oy = (cell_h - p.height) // 2
        canvas.paste(p, (cx + ox, cy + oy), p)

    canvas.save(save_path)
    print(f"Grid shattered preview → '{save_path}'")


# ---------------------------------------------------------------------------
# Main puzzle creator
# ---------------------------------------------------------------------------

def create_jigsaw(image_path, num_pieces, shape_type='curved',
                  output_dir='puzzle_pieces', seed=42):
    """
    Create interlocking jigsaw puzzle pieces from an image.

    shape_type:
        'regular' — plain rectangles
        'curved'  — interlocking jigsaw tabs (default)
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

        piece_pil  = Image.fromarray(piece_array, 'RGBA')
        piece_path = os.path.join(output_dir, f'piece_{piece_id:03d}.png')
        piece_pil.save(piece_path)

        pieces.append({
            'id':        piece_id,
            'path':      piece_path,
            'bbox':      (x1, y1, x2, y2),
            'piece_img': piece_array,
            'grid_pos':  (r, c),
            'pad':       pad,
            'pw': pw, 'ph': ph,
        })

    print(f"Generated {len(pieces)} puzzle pieces → '{output_dir}'")
    return pieces


# ---------------------------------------------------------------------------
# Preview: composite pieces back onto canvas to verify they tile perfectly
# ---------------------------------------------------------------------------

def preview_assembled(pieces, save_path='preview_assembled.png'):
    """Stamp all pieces back in place — should perfectly reconstruct the original."""
    if not pieces:
        return
    pw  = pieces[0]['pw']
    ph  = pieces[0]['ph']
    pad = pieces[0]['pad']
    img_w = max(p['bbox'][2] for p in pieces)
    img_h = max(p['bbox'][3] for p in pieces)

    canvas = Image.new('RGBA', (img_w, img_h), (180, 180, 180, 255))
    for piece in sorted(pieces, key=lambda p: p['grid_pos']):
        r, c      = piece['grid_pos']
        piece_pil = Image.fromarray(piece['piece_img'], 'RGBA')
        paste_x   = c * pw - pad
        paste_y   = r * ph - pad
        canvas.paste(piece_pil, (paste_x, paste_y), piece_pil)

    canvas.save(save_path)
    print(f"Assembled preview → '{save_path}'")
    return canvas


def preview_grid(pieces, save_path='preview_grid.png'):
    """Show all pieces laid out in a grid with transparency."""
    if not pieces:
        return
    n  = len(pieces)
    nc = math.ceil(math.sqrt(n))
    nr = math.ceil(n / nc)

    pw = pieces[0]['piece_img'].shape[1]
    ph = pieces[0]['piece_img'].shape[0]
    gap = max(4, pw // 20)

    cw = nc * pw + (nc + 1) * gap
    ch = nr * ph + (nr + 1) * gap
    canvas = Image.new('RGBA', (cw, ch), (40, 40, 40, 255))

    for i, piece in enumerate(pieces):
        col_i = i % nc
        row_i = i // nc
        x = gap + col_i * (pw + gap)
        y = gap + row_i * (ph + gap)
        p = Image.fromarray(piece['piece_img'], 'RGBA')
        canvas.paste(p, (x, y), p)

    canvas.save(save_path)
    print(f"Grid preview → '{save_path}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # IMAGE = './tard-58b8d17f3df78c353c22729d.jpg'

    # # --- Classic curved jigsaw ---
    # pieces = create_jigsaw(IMAGE, num_pieces=20, shape_type='curved',
    #                        output_dir='puzzle_curved', seed=7)
    # preview_assembled(pieces, save_path='preview_curved_assembled.png')
    # preview_grid(pieces,      save_path='preview_curved_grid.png')

    # # --- Shattered glass / Voronoi polygons ---
    # img_tmp  = Image.open(IMAGE)
    # img_w, img_h = img_tmp.width, img_tmp.height

    # shards = create_shattered(IMAGE, num_pieces=20,
    #                            output_dir='puzzle_shattered', seed=7)
    # preview_assembled_shattered(shards, img_w, img_h,
    #                              save_path='preview_shattered_assembled.png')
    # preview_grid_shattered(shards, save_path='preview_shattered_grid.png')

    parser = argparse.ArgumentParser(description="Image Puzzle Generator: Create jigsaw or shattered pieces.")
    
    # Required arguments
    parser.add_argument("image", help="Path to the source image file")
    
    # Optional parameters
    parser.add_argument("-n", "--num_pieces", type=int, default=20, 
                        help="Approximate number of pieces to generate (default: 20)")
    parser.add_argument("-s", "--style", choices=['curved', 'shattered', 'regular'], default='curved',
                        help="Puzzle style: 'curved' (jigsaw), 'shattered' (Voronoi), or 'regular' (rectangles)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for deterministic piece shapes")
    parser.add_argument("--no-preview", action="store_true", 
                        help="Disable generation of assembly and grid preview images")

    args = parser.parse_args()

    # Load image to get dimensions
    if not os.path.exists(args.image):
        print(f"Error: File {args.image} not found.")
        exit(1)

    img_tmp = Image.open(args.image)
    w, h = img_tmp.size
    output_dir_name = f"./{args.style}_{args.num_pieces}"

    print(f"Processing '{args.image}' ({w}x{h}) into {args.num_pieces} {args.style} pieces...")

    if args.style == 'shattered':
        pieces = create_shattered(args.image, args.num_pieces, output_dir=output_dir_name + '/pieces', seed=args.seed)
        if not args.no_preview:
            preview_assembled_shattered(pieces, w, h, save_path=output_dir_name + '/preview_assembled.png')
            preview_grid_shattered(pieces, save_path=output_dir_name + '/preview_grid.png')
    else:
        # Handles 'curved' and 'regular'
        pieces = create_jigsaw(args.image, args.num_pieces, shape_type=args.style, 
                               output_dir=output_dir_name + '/pieces', seed=args.seed)
        if not args.no_preview:
            preview_assembled(pieces, save_path=output_dir_name + '/preview_assembled.png')
            preview_grid(pieces, save_path=output_dir_name + '/preview_grid.png')

    print("Done!")