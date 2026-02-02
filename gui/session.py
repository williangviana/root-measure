"""Session save/resume — auto-save state to JSON, load on folder open."""

import json
import re
from pathlib import Path

SESSION_VERSION = 1
_LAST_FOLDER_FILE = Path.home() / '.root_measure_last_folder'
_RECENT_FOLDERS_FILE = Path.home() / '.root_measure_recent_folders.json'

# --- Output folder layout ---
_ROOT = 'root_measure'


def _sanitize_name(name):
    """Convert experiment name to a safe directory name."""
    if not name:
        return ''
    name = name.strip()
    name = re.sub(r'[^\w\s\-.]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name


def data_dir(folder, experiment=''):
    """Return path to root_measure/[experiment]/data/ inside the scan folder."""
    safe = _sanitize_name(experiment)
    if safe:
        return folder / _ROOT / safe / 'data'
    return folder / _ROOT / 'data'


def traces_dir(folder, experiment=''):
    """Return path to root_measure/[experiment]/traces/ inside the scan folder."""
    safe = _sanitize_name(experiment)
    if safe:
        return folder / _ROOT / safe / 'traces'
    return folder / _ROOT / 'traces'


def session_dir(folder, experiment=''):
    """Return path to root_measure/[experiment]/.session/ inside the scan folder."""
    safe = _sanitize_name(experiment)
    if safe:
        return folder / _ROOT / safe / '.session'
    return folder / _ROOT / '.session'


def save_last_folder(folder):
    """Remember the last opened folder path."""
    try:
        _LAST_FOLDER_FILE.write_text(str(folder))
    except Exception:
        pass
    save_recent_folder(folder)


def save_recent_folder(folder):
    """Add folder to recent folders list (max 10, deduped)."""
    try:
        folders = []
        if _RECENT_FOLDERS_FILE.exists():
            folders = json.loads(_RECENT_FOLDERS_FILE.read_text())
        folder_str = str(folder)
        if folder_str in folders:
            folders.remove(folder_str)
        folders.insert(0, folder_str)
        folders = folders[:10]
        _RECENT_FOLDERS_FILE.write_text(json.dumps(folders))
    except Exception:
        pass


def get_recent_folders():
    """Return list of recent folder Paths that still exist on disk."""
    try:
        if not _RECENT_FOLDERS_FILE.exists():
            return []
        folders = json.loads(_RECENT_FOLDERS_FILE.read_text())
        result = []
        for f in folders:
            p = Path(f)
            if p.is_dir():
                result.append(p)
        return result
    except Exception:
        return []


def get_session_summaries(folder):
    """Scan for all experiment sessions under folder.

    Returns list of summary dicts, one per experiment session found.
    Each dict has: folder, folder_name, experiment, n_done, n_total,
    current_image.
    """
    results = []
    base = folder / _ROOT
    if not base.is_dir():
        return results
    # scan experiment subfolders: root_measure/<experiment>/.session/session.json
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        session_file = sub / '.session' / 'session.json'
        s = _load_session_summary(folder, session_file)
        if s:
            results.append(s)
    # also check for legacy session.json (no experiment subfolder)
    legacy = base / '.session' / 'session.json'
    if legacy.exists():
        s = _load_session_summary(folder, legacy)
        if s:
            results.append(s)
    return results


def get_session_summary(folder):
    """Load first valid session summary from folder. Returns dict or None."""
    summaries = get_session_summaries(folder)
    return summaries[0] if summaries else None


def _load_session_summary(folder, session_path):
    """Load a single session.json and return summary dict or None."""
    try:
        if not session_path.exists():
            return None
        data = json.loads(session_path.read_text())
        if data.get('version') != SESSION_VERSION:
            return None
        experiment = data.get('settings', {}).get('experiment', '')
        images = data.get('images', [])
        processed = data.get('processed_images', [])
        return {
            'folder': folder,
            'folder_name': folder.name,
            'experiment': experiment,
            'n_done': len(processed),
            'n_total': len(images),
            'current_image': data.get('current_image', ''),
        }
    except Exception:
        return None


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


def save_experiment_name(folder, name, experiment=''):
    """Persist experiment name in output folder."""
    try:
        p = session_dir(folder, experiment) / 'experiment_name.txt'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    except Exception:
        pass


def get_experiment_name(folder, experiment=''):
    """Load saved experiment name, or empty string."""
    try:
        p = session_dir(folder, experiment) / 'experiment_name.txt'
        if p.exists():
            return p.read_text().strip()
    except Exception:
        pass
    return ''


def save_persistent_settings(folder, settings, experiment=''):
    """Persist settings that carry across scans."""
    try:
        p = session_dir(folder, experiment) / 'persistent_settings.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(settings))
    except Exception:
        pass


def get_persistent_settings(folder, experiment=''):
    """Load persistent settings dict, or empty dict."""
    try:
        p = session_dir(folder, experiment) / 'persistent_settings.json'
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {}


def save_csv_format(folder, fmt, experiment=''):
    """Persist CSV format choice in output folder."""
    try:
        p = session_dir(folder, experiment) / 'csv_format.txt'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(fmt)
    except Exception:
        pass


def get_csv_format(folder, experiment=''):
    """Load saved CSV format, or empty string."""
    try:
        p = session_dir(folder, experiment) / 'csv_format.txt'
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

    # snapshot current image canvas into per-image dict
    if app.image_path:
        app._image_canvas_data[app.image_path.name] = _collect_canvas(app.canvas)

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
        'image_canvas_data': dict(app._image_canvas_data),
        'workflow_step': _get_workflow_step(app.sidebar),
        'click_state': {
            'plate_idx': getattr(app, '_current_plate_idx', 0),
            'split': getattr(app, '_split', False),
            'current_group': getattr(app, '_current_group', 0),
            'split_stage': getattr(app, '_split_stage', 0),
        },
    }

    session_path.parent.mkdir(parents=True, exist_ok=True)
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
        'mark_points': [list(p) for p in canvas._mark_points],
        'traces': traces,
    }


def _get_workflow_step(sidebar):
    """Determine current workflow step from button colors."""
    buttons = [
        sidebar.btn_select_plates, sidebar.btn_click_roots,
        sidebar.btn_measure, sidebar.btn_review,
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
    # restore genotypes and conditions so re-save after review works
    sidebar.entry_genotypes.delete(0, 'end')
    sidebar.entry_genotypes.insert(0, settings.get('genotypes', ''))
    sidebar.entry_condition.delete(0, 'end')
    sidebar.entry_condition.insert(0, settings.get('conditions', ''))
    sidebar.var_csv_format.set(settings.get('csv_format', 'R'))
    sidebar.var_plot.set(settings.get('plot', True))
