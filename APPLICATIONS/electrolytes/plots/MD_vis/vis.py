import numpy as np
import py3Dmol
import ipywidgets as widgets
from IPython.display import display, clear_output

ELEMENT_COLORS = {
    'Na': '#ab5cf2',
    'P':  '#ff8000',
    'F':  '#90e050',
    'O':  '#ff0d0d',
    'C':  '#909090',
    'H':  '#dddddd',
}
ELEMENT_RADII = {
    'Na': 0.60, 'P': 0.55, 'F': 0.35, 'O': 0.40, 'C': 0.40, 'H': 0.25,
}


def atoms_to_xyz(atoms):
    lines = [str(len(atoms)), '']
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append(f'{sym}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}')
    return '\n'.join(lines)


def _draw_box(view, cell):
    a, b, c = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    corners = np.array([
        [0, 0, 0], [a, 0, 0], [0, b, 0], [0, 0, c],
        [a, b, 0], [a, 0, c], [0, b, c], [a, b, c],
    ])
    for i, j in [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),
                 (2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]:
        view.addLine({
            'start': {'x': corners[i,0], 'y': corners[i,1], 'z': corners[i,2]},
            'end':   {'x': corners[j,0], 'y': corners[j,1], 'z': corners[j,2]},
            'color': 'white', 'linewidth': 1.5,
        })
    return a, b, c


def _apply_style(view, style, hide_H, model_sel=None):
    sel_base = {'model': -1} if model_sel == 'all_frames' else {}
    if style == 'sphere':
        for elem, color in ELEMENT_COLORS.items():
            if hide_H and elem == 'H':
                continue
            view.setStyle({**sel_base, 'elem': elem},
                          {'sphere': {'color': color,
                                      'radius': ELEMENT_RADII.get(elem, 0.40)}})
    elif style == 'stick':
        view.setStyle(sel_base, {'stick': {'colorscheme': 'Jmol', 'radius': 0.15}})
        if hide_H:
            view.setStyle({**sel_base, 'elem': 'H'}, {})
    elif style == 'ball+stick':
        view.setStyle(sel_base, {'stick': {'colorscheme': 'Jmol', 'radius': 0.12}})
        for elem, color in ELEMENT_COLORS.items():
            if hide_H and elem == 'H':
                continue
            view.setStyle({**sel_base, 'elem': elem},
                          {'sphere': {'color': color,
                                      'radius': ELEMENT_RADII.get(elem, 0.35) * 0.6},
                           'stick':  {'colorscheme': 'Jmol', 'radius': 0.12}})


def view_frame(traj, frame_idx=0, width=900, height=650,
               style='sphere', hide_H=False):
    """Render a single frame. style: 'sphere' | 'stick' | 'ball+stick'"""
    atoms = traj[frame_idx].copy()
    atoms.set_pbc([True, True, True])
    atoms.wrap()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(atoms_to_xyz(atoms), 'xyz')
    _apply_style(view, style, hide_H)

    cell = atoms.get_cell()
    a, b, c = _draw_box(view, cell)

    view.setBackgroundColor('#1a1a2e')
    view.zoomTo()
    view.show()
    print(f'Frame {frame_idx} | {len(atoms)} atoms | '
          f'cell {a:.2f} x {b:.2f} x {c:.2f} Ang')
    return view


def animate_frames(traj, start=0, end=9, stride=1,
                   width=900, height=650, style='sphere', hide_H=False):
    """
    Animate a range of frames [start, end] with given stride.
    NPT note: box size changes per frame — py3Dmol has no per-frame shape support
    so the box outline is omitted. Atoms are wrapped per-frame.
    style: 'sphere' | 'ball+stick'
    """
    frame_indices = range(start, end + 1, stride)

    models = ""
    for i in frame_indices:
        atoms = traj[i].copy()
        atoms.set_pbc([True, True, True])
        atoms.wrap()
        models += f"{len(atoms)}\nframe {i}\n"
        for sym, p in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            models += f"{sym}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}\n"

    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(models, 'xyz')
    _apply_style(view, style, hide_H, model_sel='all_frames')

    view.setBackgroundColor('#1a1a2e')
    view.zoomTo()
    view.animate({'loop': 'forward', 'reps': 0})
    view.show()

    cells = [traj[i].get_cell() for i in frame_indices]
    vols  = [c[0,0]*c[1,1]*c[2,2] for c in cells]
    n_frames = len(list(frame_indices))
    print(f'Frames {start}:{end}:{stride} ({n_frames} frames) | '
          f'{len(traj[start])} atoms | vol {min(vols):.1f}–{max(vols):.1f} Å³')
    return view


def save_gif(traj, output_path, start=0, end=9, stride=1,
             hide_H=False, fps=5, size=500):
    """
    Render frames with PIL (top-down XY projection) and save as GIF.
    Much faster than matplotlib — draws circles directly in memory, no temp files.

    size : int — pixel width and height of each frame
    """
    from PIL import Image, ImageDraw
    import os

    def _hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    COLORS_RGB = {k: _hex_to_rgb(v) for k, v in ELEMENT_COLORS.items()}
    BG = (26, 26, 46)   # #1a1a2e
    MARGIN = 20

    frame_indices = list(range(start, end + 1, stride))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pil_frames = []
    for i in frame_indices:
        atoms = traj[i].copy()
        atoms.set_pbc([True, True, True])
        atoms.wrap()

        pos  = atoms.get_positions()
        syms = atoms.get_chemical_symbols()
        cell = atoms.get_cell()
        a, b = float(cell[0, 0]), float(cell[1, 1])

        scale = (size - 2 * MARGIN) / max(a, b)

        img  = Image.new('RGB', (size, size), BG)
        draw = ImageDraw.Draw(img)

        # Box outline
        bx = MARGIN + a * scale
        by = MARGIN + b * scale
        draw.rectangle([MARGIN, MARGIN, bx, by], outline=(180, 180, 180), width=1)

        # Atoms — sort by z so front atoms draw on top
        order = np.argsort(pos[:, 2])
        for idx in order:
            sym = syms[idx]
            if hide_H and sym == 'H':
                continue
            p = pos[idx]
            color  = COLORS_RGB.get(sym, (255, 255, 255))
            radius = ELEMENT_RADII.get(sym, 0.4) * scale * 0.9
            cx = MARGIN + p[0] * scale
            cy = MARGIN + p[1] * scale   # PIL y increases downward
            draw.ellipse([cx - radius, cy - radius,
                          cx + radius, cy + radius], fill=color)

        pil_frames.append(img)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=int(1000 / fps),
        optimize=True,
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f'GIF saved → {output_path}  '
          f'({len(frame_indices)} frames, {fps} fps, {size_mb:.2f} MB)')


def _render_frame_3d(args):
    """Render one frame as a PIL Image using perspective projection. Multiprocessing-safe."""
    from PIL import Image, ImageDraw, ImageFont

    pos, syms, cell_abc, size, hide_H, elev, azim, atom_scale, time_label, label = args
    a, b, c = cell_abc

    # Camera: orbit around box centre
    center   = np.array([a / 2, b / 2, c / 2])
    r        = max(a, b, c) * 2.2
    er, ar   = np.radians(elev), np.radians(azim)
    camera   = center + r * np.array([np.cos(er) * np.cos(ar),
                                       np.cos(er) * np.sin(ar),
                                       np.sin(er)])
    forward  = center - camera;  forward /= np.linalg.norm(forward)
    right    = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
    up_cam   = np.cross(right, forward)

    pos = np.asarray(pos)
    rel = pos - camera
    xc  = rel @ right
    yc  = rel @ up_cam
    zc  = rel @ forward          # positive = in front of camera

    # Perspective divide
    fov   = 45.0
    f     = 1.0 / np.tan(np.radians(fov / 2))
    scale = size / 2
    sx    = size / 2 + f * xc / zc * scale
    sy    = size / 2 - f * yc / zc * scale   # PIL y flipped

    # Draw back-to-front (painter's algorithm)
    order = np.argsort(-zc)

    BG  = (26, 26, 46)
    img = Image.new('RGB', (size, size), BG)
    draw = ImageDraw.Draw(img)

    def _hex(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    for idx in order:
        sym = syms[idx]
        if hide_H and sym == 'H':
            continue
        if zc[idx] <= 0:          # behind camera
            continue
        base_r = ELEMENT_RADII.get(sym, 0.4)
        pr     = base_r * f * scale / zc[idx] * atom_scale
        cx, cy = sx[idx], sy[idx]
        color  = _hex(ELEMENT_COLORS.get(sym, '#ffffff'))
        draw.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=color)

    if time_label or cell_abc:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12, size // 22))
        except Exception:
            font = ImageFont.load_default()
        margin = max(6, size // 60)
        try:
            line_h = font.getbbox("A")[3] + 2
        except Exception:
            line_h = 14
        y = margin
        for text, color in [
            (time_label,                                                            (255, 220, 80)),
            (f"a={cell_abc[0]:.1f} b={cell_abc[1]:.1f} c={cell_abc[2]:.1f} Å",  (160, 210, 255)),
        ]:
            if not text:
                continue
            draw.text((margin + 1, y + 1), text, fill=(0, 0, 0), font=font)
            draw.text((margin,     y),     text, fill=color,      font=font)
            y += line_h

    if label:
        try:
            font_lbl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(11, size // 26))
        except Exception:
            font_lbl = ImageFont.load_default()
        try:
            lbl_w = font_lbl.getbbox(label)[2]
            lbl_h = font_lbl.getbbox(label)[3]
        except Exception:
            lbl_w, lbl_h = len(label) * 7, 12
        margin = max(6, size // 60)
        x_lbl = (size - lbl_w) // 2
        y_lbl = size - lbl_h - margin
        draw.text((x_lbl + 1, y_lbl + 1), label, fill=(0, 0, 0),       font=font_lbl)
        draw.text((x_lbl,     y_lbl),     label, fill=(200, 200, 200),  font=font_lbl)

    return img


def save_gif_3d(traj, output_path, start=0, end=9, stride=1,
                hide_H=False, fps=5, size=400, elev=20, azim=45,
                spin=False, n_workers=4, atom_scale=0.5, timestep_fs=None,
                label=""):
    """
    Fast 3D perspective GIF using PIL + perspective projection + multiprocessing.

    elev        : camera elevation in degrees
    azim        : starting azimuth in degrees
    spin        : if True, rotate azim 360° across frames
    n_workers   : parallel workers (set to 1 to disable multiprocessing)
    size        : output pixel size
    timestep_fs : MD timestep in fs used to stamp each frame with simulation time
                  in ps. If None, falls back to atoms.info['time'] (ASE convention,
                  also in fs). Set to 0 to disable the time stamp.
    """
    from PIL import Image
    from multiprocessing import Pool
    import os

    frame_indices = list(range(start, end + 1, stride))
    n_frames = len(frame_indices)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read all frame data in main process (traj has a file handle, can't pickle)
    args_list = []
    for k, i in enumerate(frame_indices):
        atoms = traj[i].copy()
        atoms.set_pbc([True, True, True])
        atoms.wrap()
        cell  = atoms.get_cell()
        cur_azim = (azim + k * 360 / n_frames) if spin else azim

        # Compute time label
        if timestep_fs == 0:
            time_label = ""
        elif timestep_fs is not None:
            time_ns = i * timestep_fs / 1e6
            time_label = f"{time_ns:.3f} ns"
        else:
            raw = atoms.info.get("time", None)   # ASE stores time in fs
            time_label = f"{raw / 1e6:.3f} ns" if raw is not None else f"frame {i}"

        args_list.append((
            atoms.get_positions(),
            atoms.get_chemical_symbols(),
            (float(cell[0,0]), float(cell[1,1]), float(cell[2,2])),
            size, hide_H, elev, cur_azim, atom_scale, time_label, label,
        ))

    if n_workers > 1:
        with Pool(n_workers) as pool:
            pil_frames = pool.map(_render_frame_3d, args_list)
    else:
        pil_frames = [_render_frame_3d(a) for a in args_list]

    pil_frames[0].save(
        output_path, save_all=True, append_images=pil_frames[1:],
        loop=0, duration=int(1000 / fps), optimize=True,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f'3D GIF saved → {output_path}  '
          f'({n_frames} frames, {fps} fps, {size_mb:.2f} MB)')


def browse_frames(traj, start=0, end=None, stride=1,
                  width=900, height=650, style='sphere', hide_H=False):
    """
    Interactive slider to browse trajectory frames one at a time.
    Drag the slider to load and render any frame on demand.
    """
    n_total = len(traj)
    if end is None:
        end = n_total - 1

    out = widgets.Output()

    def _render(frame_idx):
        with out:
            clear_output(wait=True)
            view_frame(traj, frame_idx=frame_idx, width=width, height=height,
                       style=style, hide_H=hide_H)

    slider = widgets.IntSlider(
        value=start, min=start, max=end, step=stride,
        description='Frame:', continuous_update=False,
        layout=widgets.Layout(width='80%'),
    )
    widgets.interact(_render, frame_idx=slider)
    display(out)
