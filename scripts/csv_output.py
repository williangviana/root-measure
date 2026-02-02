import numpy as np
import pandas as pd


def _build_rows(results, plate_labels, plate_offset, root_offset,
                point_plates, num_marks, split_plate, image_name='',
                group_to_plate=None):
    """Build list of row dicts from measurement results."""
    labels = plate_labels.values() if isinstance(plate_labels, dict) else plate_labels
    is_factorial = plate_labels and any(
        cond is not None for (geno, cond) in labels)

    # Fallback: map sorted unique groups to sequential plate numbers
    if group_to_plate is None:
        unique_groups = sorted(set(point_plates)) if point_plates else []
        if split_plate:
            group_to_plate = {g: idx // 2 for idx, g in enumerate(unique_groups)}
        else:
            group_to_plate = {g: idx for idx, g in enumerate(unique_groups)}

    group_counters = {}
    rows = []
    for i, r in enumerate(results):
        root_num = root_offset + i + 1
        row = {
            'Root_ID': root_num,
            'Image': image_name,
            'Length_cm': round(r['length_cm'], 3) if r['length_cm'] is not None else '',
            'Length_px': round(r['length_px'], 1) if r['length_px'] is not None else '',
            'Warning': r.get('warning') or '',
        }

        if plate_labels and i < len(point_plates):
            group_idx = point_plates[i]
            local_plate = group_to_plate.get(group_idx, 0)
            row['Plate'] = plate_offset + local_plate + 1
            genotype, condition = plate_labels[group_idx]
            row['Genotype'] = genotype
            if is_factorial:
                row['Condition'] = condition or ''

            group_counters[group_idx] = group_counters.get(group_idx, 0) + 1
            row['Plant_ID'] = group_counters[group_idx]

        segments = r.get('segments', [])
        if num_marks > 0:
            for seg_i in range(num_marks + 1):
                col_name = f'Segment_{seg_i + 1}_cm'
                if seg_i < len(segments):
                    row[col_name] = round(segments[seg_i], 3)
                else:
                    row[col_name] = ''

        rows.append(row)

    return rows, is_factorial


# ---------------------------------------------------------------------------
# raw_data.csv — append-only log, one row per root
# ---------------------------------------------------------------------------

def _raw_col_order(is_factorial, num_marks):
    """Standard column order for raw_data.csv."""
    if is_factorial:
        cols = ['Root_ID', 'Image', 'Plate', 'Genotype', 'Plant_ID',
                'Condition', 'Length_cm']
    else:
        cols = ['Root_ID', 'Image', 'Plate', 'Genotype', 'Plant_ID',
                'Length_cm']
    if num_marks > 0:
        for seg_i in range(num_marks + 1):
            cols.append(f'Segment_{seg_i + 1}_cm')
    cols.extend(['Length_px', 'Warning'])
    return cols


def save_raw(df_new, csv_path, is_factorial, num_marks):
    """Append to raw_data.csv, dedup by Image, re-number Root_ID."""
    col_order = _raw_col_order(is_factorial, num_marks)

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        # remove old rows for this image (replace on re-measure)
        if 'Image' in df_existing.columns and 'Image' in df_new.columns:
            image_name = df_new['Image'].iloc[0] if len(df_new) > 0 else ''
            if image_name:
                df_existing = df_existing[df_existing['Image'] != image_name]
        for col in col_order:
            if col not in df_existing.columns:
                df_existing[col] = ''
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = ''
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    # re-number Root_ID sequentially
    if 'Root_ID' in df.columns:
        df['Root_ID'] = range(1, len(df) + 1)

    col_order = [c for c in col_order if c in df.columns]
    for col in df.columns:
        if col not in col_order:
            col_order.append(col)
    df = df[col_order]
    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# tidy_data.csv — generated once from raw_data at "Finish & Plot"
# ---------------------------------------------------------------------------

def generate_tidy(raw_path, tidy_path, csv_format='R'):
    """Read raw_data.csv and write a clean tidy_data.csv for analysis.

    - Drops Warning/Length_px/Image columns
    - Removes rows with warnings
    - Sorts by Genotype, then Condition (if present), then Plate
    - R format: tall (one row per root)
    - Prism format: wide (genotypes as columns)
    """
    if not raw_path.exists():
        return
    df = pd.read_csv(raw_path)
    if df.empty:
        return

    # drop warned rows
    if 'Warning' in df.columns:
        df = df[df['Warning'].fillna('').astype(str).str.strip() == ''].copy()

    if csv_format == 'Prism':
        _write_tidy_prism(df, tidy_path)
    else:
        _write_tidy_r(df, tidy_path)

    print(f"Tidy data written to: {tidy_path}")


def _write_tidy_r(df, tidy_path):
    """Tall R format: sorted by Genotype > Condition > Plate."""
    sort_cols = []
    if 'Genotype' in df.columns:
        sort_cols.append('Genotype')
    if 'Condition' in df.columns:
        sort_cols.append('Condition')
    if 'Plate' in df.columns:
        sort_cols.append('Plate')
    if sort_cols:
        df = df.sort_values(sort_cols, ignore_index=True)

    # drop internal columns
    drop = ['Warning', 'Length_px', 'Image']
    df = df.drop(columns=[c for c in drop if c in df.columns])

    # re-number Root_ID after sort
    if 'Root_ID' in df.columns:
        df['Root_ID'] = range(1, len(df) + 1)

    df.to_csv(tidy_path, index=False)


def _write_tidy_prism(df, tidy_path):
    """Wide Prism format from raw data."""
    has_condition = ('Condition' in df.columns and
                     df['Condition'].notna().any() and
                     (df['Condition'].astype(str).str.strip() != '').any())

    if has_condition:
        _write_tidy_prism_factorial(df, tidy_path)
    else:
        _write_tidy_prism_simple(df, tidy_path)


def _write_tidy_prism_simple(df, tidy_path):
    """Prism simple: each genotype is a column of Length_cm values."""
    if 'Genotype' not in df.columns:
        df.to_csv(tidy_path, index=False)
        return

    genotypes = df['Genotype'].unique().tolist()
    data = {}
    for geno in genotypes:
        data[geno] = df.loc[df['Genotype'] == geno, 'Length_cm'].tolist()

    max_len = max(len(v) for v in data.values()) if data else 0
    for geno in data:
        data[geno] = data[geno] + [np.nan] * (max_len - len(data[geno]))

    pd.DataFrame(data).to_csv(tidy_path, index=False)


def _write_tidy_prism_factorial(df, tidy_path):
    """Prism factorial: rows=conditions, blocks of columns per genotype."""
    genotypes = df['Genotype'].unique().tolist()
    conditions = df['Condition'].unique().tolist()

    max_reps = {}
    for geno in genotypes:
        max_n = 0
        for cond in conditions:
            n = len(df[(df['Genotype'] == geno) & (df['Condition'] == cond)])
            max_n = max(max_n, n)
        max_reps[geno] = max_n

    rows = []
    for cond in conditions:
        row = [cond]
        for geno in genotypes:
            vals = df.loc[(df['Genotype'] == geno) &
                          (df['Condition'] == cond),
                          'Length_cm'].tolist()
            n_cols = max_reps[geno]
            row.extend(vals + [''] * (n_cols - len(vals)))
        rows.append(row)

    header = ['']
    for geno in genotypes:
        header.append(geno)
        header.extend([''] * (max_reps[geno] - 1))

    pd.DataFrame(rows, columns=header).to_csv(tidy_path, index=False)


# ---------------------------------------------------------------------------
# Offset helpers
# ---------------------------------------------------------------------------

def get_offsets_from_csv(csv_path, exclude_image=''):
    """Read existing raw CSV and return (plate_offset, root_offset).

    If exclude_image is given, ignore rows for that image (so re-saving
    the same image produces correct offsets).
    """
    try:
        if not csv_path.exists():
            return 0, 0
        df = pd.read_csv(csv_path)
        if df.empty:
            return 0, 0
        if exclude_image and 'Image' in df.columns:
            df = df[df['Image'] != exclude_image]
        if df.empty:
            return 0, 0
        plate_off = 0
        if 'Plate' in df.columns:
            plate_off = int(df['Plate'].max())
        root_off = len(df)
        return plate_off, root_off
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def append_results_to_csv(results, csv_path, plates, plate_labels, plate_offset,
                          root_offset, point_plates, num_marks=0,
                          split_plate=False, image_name='',
                          group_to_plate=None):
    """Append measurements to raw_data.csv.

    Returns:
        (new_plate_offset, new_root_offset) for the next image
    """
    rows, is_factorial = _build_rows(
        results, plate_labels, plate_offset, root_offset,
        point_plates, num_marks, split_plate, image_name=image_name,
        group_to_plate=group_to_plate)

    df_new = pd.DataFrame(rows)
    save_raw(df_new, csv_path, is_factorial, num_marks)
    print(f"\nResults saved to: {csv_path}")

    new_plate_offset = plate_offset + len(plates)
    new_root_offset = root_offset + len(results)
    return new_plate_offset, new_root_offset


def save_metadata(meta_path, **kwargs):
    """Append one row of measurement metadata to a CSV log file."""
    col_order = ['timestamp', 'image_name', 'dpi', 'sensitivity',
                 'experiment', 'genotypes', 'conditions', 'csv_format',
                 'split_plate', 'num_marks', 'software_version']
    kwargs.setdefault('software_version', '1.0.0')
    row = {k: kwargs.get(k, '') for k in col_order}
    df_new = pd.DataFrame([row])

    if meta_path.exists():
        df_existing = pd.read_csv(meta_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    cols = [c for c in col_order if c in df.columns]
    for c in df.columns:
        if c not in cols:
            cols.append(c)
    df[cols].to_csv(meta_path, index=False)
