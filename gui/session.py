"""Session save/resume — auto-save state to JSON, load on folder open."""

import json
from pathlib import Path

SESSION_VERSION = 1
_LAST_FOLDER_FILE = Path.home() / '.root_measure_last_folder'


def save_last_folder(folder):
    """Remember the last opened folder path."""
    try:
        _LAST_FOLDER_FILE.write_text(str(folder))
    except Exception:
        pass


def get_last_folder():
    """Return the last opened folder Path, or None."""
    try:
        if _LAST_FOLDER_FILE.exists():
            text = _LAST_FOLDER_FILE.read_text().strip()
            p = Path(text)
            if p.is_dir():
                return p
    except Exception:
        pass
    return None


def save_experiment_name(folder, name):
    """Persist experiment name in output folder."""
    try:
        p = folder / 'output' / 'experiment_name.txt'
        p.parent.mkdir(exist_ok=True)
        p.write_text(name)
    except Exception:
        pass


def get_experiment_name(folder):
    """Load saved experiment name, or empty string."""
    try:
        p = folder / 'output' / 'experiment_name.txt'
        if p.exists():
            return p.read_text().strip()
    except Exception:
        pass
    return ''


def save_session(session_path, app):
    """Collect app/sidebar/canvas state and write to JSON."""
    folder = app.folder
    if not folder:
        return

    data = {
        'version': SESSION_VERSION,
        'folder': str(folder),
        'images': [p.name for p in app.images],
        'processed_images': [str(p.name) for p in app._processed_images],
        'plate_offset': app._plate_offset,
        'root_offset': app._root_offset,
        'current_image': app.image_path.name if app.image_path else None,
        'settings': _collect_settings(app.sidebar),
        'canvas': _collect_canvas(app.canvas),
        'workflow_step': _get_workflow_step(app.sidebar),
    }

    session_path.parent.mkdir(exist_ok=True)
    tmp = session_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(session_path)


def load_session(session_path):
    """Read and validate session JSON. Returns dict or None."""
    try:
        if not session_path.exists():
            return None
        data = json.loads(session_path.read_text())
        if data.get('version') != SESSION_VERSION:
            return None
        return data
    except Exception:
        return None


def _collect_settings(sidebar):
    return {
        'dpi': sidebar.entry_dpi.get().strip(),
        'sensitivity': sidebar.var_sensitivity.get(),
        'multi_measurement': sidebar.var_multi.get(),
        'segments': sidebar.entry_segments.get().strip(),
        'split_plate': sidebar.var_split.get(),
        'experiment': sidebar.entry_experiment.get().strip(),
        'genotypes': sidebar.entry_genotypes.get().strip(),
        'conditions': sidebar.entry_condition.get().strip(),
        'csv_format': sidebar.var_csv_format.get(),
        'plot': sidebar.var_plot.get(),
    }


def _collect_canvas(canvas):
    all_marks = {}
    for k, v in canvas._all_marks.items():
        all_marks[str(k)] = [list(m) for m in v]
    traces = []
    for path, shades, mark_indices in canvas._traces:
        # use tolist() for numpy arrays to get native Python types
        p = path.tolist() if hasattr(path, 'tolist') else [list(r) for r in path]
        mi = (mark_indices.tolist() if hasattr(mark_indices, 'tolist')
              else [int(x) for x in mark_indices])
        traces.append({
            'path': p,
            'shades': list(shades),
            'mark_indices': mi,
        })
    return {
        'plates': [list(p) for p in canvas._plates],
        'root_points': [list(p) for p in canvas._root_points],
        'root_flags': list(canvas._root_flags),
        'root_groups': list(canvas._root_groups),
        'root_plates': list(canvas._root_plates),
        'all_marks': all_marks,
        'traces': traces,
    }


def _get_workflow_step(sidebar):
    """Determine current workflow step from button colors."""
    buttons = [
        sidebar.btn_select_plates, sidebar.btn_click_roots,
        sidebar.btn_measure, sidebar.btn_review, sidebar.btn_save,
    ]
    step = 0
    for i, btn in enumerate(buttons):
        color = btn.cget('fg_color')
        if color == sidebar._step_color_done:
            step = i + 2
        elif color == sidebar._step_color_active:
            step = i + 1
            break
    return step


def restore_settings(sidebar, settings):
    """Fill sidebar entries/vars from saved settings dict."""
    sidebar.entry_dpi.delete(0, 'end')
    sidebar.entry_dpi.insert(0, settings.get('dpi', ''))
    sidebar.var_sensitivity.set(settings.get('sensitivity', 'medium'))
    sidebar.var_multi.set(settings.get('multi_measurement', False))
    sidebar.entry_segments.delete(0, 'end')
    sidebar.entry_segments.insert(0, settings.get('segments', ''))
    if sidebar.var_multi.get():
        sidebar.frame_segments.pack(pady=(0, 8), fill='x')
    else:
        sidebar.frame_segments.pack_forget()
    sidebar.var_split.set(settings.get('split_plate', False))
    sidebar.entry_experiment.delete(0, 'end')
    sidebar.entry_experiment.insert(0, settings.get('experiment', ''))
    # genotypes and condition are NOT restored — they change per scan
    sidebar.var_csv_format.set(settings.get('csv_format', 'R'))
    sidebar.var_plot.set(settings.get('plot', True))
