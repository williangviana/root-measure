import numpy as np
import pandas as pd


def _build_rows(results, plate_labels, plate_offset, root_offset,
                point_plates, num_marks, split_plate, image_name=''):
    """Build list of row dicts and detect if factorial design."""
    is_factorial = plate_labels and any(
        cond is not None for (geno, cond) in plate_labels)

    group_counters = {}
    rows = []
    for i, r in enumerate(results):
        root_num = root_offset + i + 1
        row = {
            'Root_ID': root_num,
            'Image': image_name,
            'Length_cm': round(r['length_cm'], 3) if r['length_cm'] is not None else '',
            'Length_px': round(r['length_px'], 1) if r['length_px'] is not None else '',
            'Warning': r['warning'] or '',
        }

        if plate_labels and i < len(point_plates):
            group_idx = point_plates[i]
            if split_plate:
                physical_plate = group_idx // 2
            else:
                physical_plate = group_idx
            row['Plate'] = plate_offset + physical_plate + 1
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


def _save_r_format(df_new, csv_path, is_factorial, num_marks):
    """Save in R (tidy/long) format — one row per root."""
    if is_factorial:
        col_order = ['Root_ID', 'Image', 'Plate', 'Genotype', 'Plant_ID',
                      'Condition', 'Length_cm']
    else:
        col_order = ['Root_ID', 'Image', 'Plate', 'Genotype', 'Plant_ID', 'Length_cm']

    if num_marks > 0:
        for seg_i in range(num_marks + 1):
            col_order.append(f'Segment_{seg_i + 1}_cm')
    col_order.extend(['Length_px', 'Warning'])

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        # remove old rows for this image (avoid duplicates on re-measure)
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

    # re-number Root_ID sequentially after any dedup
    if 'Root_ID' in df.columns:
        df['Root_ID'] = range(1, len(df) + 1)

    col_order = [c for c in col_order if c in df.columns]
    for col in df.columns:
        if col not in col_order:
            col_order.append(col)
    df = df[col_order]
    df.to_csv(csv_path, index=False)


def _save_prism_simple(df_new, csv_path):
    """Save in Prism simple format — each genotype is a column.

    Only uses Length_cm values and drops warned rows.
    Appends new data to existing columns if file exists.
    """
    # filter out warned rows
    valid = df_new[df_new['Warning'].fillna('').astype(str).str.strip() == '']
    if 'Genotype' not in valid.columns:
        valid.to_csv(csv_path, index=False)
        return

    genotypes = valid['Genotype'].unique().tolist()

    # build new data per genotype
    new_data = {}
    for geno in genotypes:
        vals = valid.loc[valid['Genotype'] == geno, 'Length_cm'].tolist()
        new_data[geno] = vals

    # if file exists, append to existing columns
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        for geno in genotypes:
            if geno in df_existing.columns:
                existing_vals = df_existing[geno].dropna().tolist()
                new_data[geno] = existing_vals + new_data[geno]
        # preserve genotypes from existing file that aren't in new data
        for col in df_existing.columns:
            if col not in new_data:
                new_data[col] = df_existing[col].dropna().tolist()

    # pad to equal length with NaN
    max_len = max(len(v) for v in new_data.values()) if new_data else 0
    for geno in new_data:
        vals = new_data[geno]
        new_data[geno] = vals + [np.nan] * (max_len - len(vals))

    df_out = pd.DataFrame(new_data)
    df_out.to_csv(csv_path, index=False)


def _save_prism_factorial(df_new, csv_path):
    """Save in Prism factorial format — rows are conditions, columns are replicates.

    Format: first column = condition label, then N columns per genotype
    (one per replicate), with genotype name as header of first column in block.
    """
    # filter out warned rows
    valid = df_new[df_new['Warning'].fillna('').astype(str).str.strip() == '']
    if 'Genotype' not in valid.columns or 'Condition' not in valid.columns:
        valid.to_csv(csv_path, index=False)
        return

    genotypes = valid['Genotype'].unique().tolist()
    conditions = valid['Condition'].unique().tolist()

    # if file exists, merge with existing data
    if csv_path.exists():
        df_existing = _read_prism_factorial(csv_path, genotypes)
        if df_existing is not None:
            valid = pd.concat([df_existing, valid], ignore_index=True)
            genotypes = valid['Genotype'].unique().tolist()
            conditions = valid['Condition'].unique().tolist()

    # find max replicates per genotype across all conditions
    max_reps = {}
    for geno in genotypes:
        max_n = 0
        for cond in conditions:
            n = len(valid[(valid['Genotype'] == geno) &
                          (valid['Condition'] == cond)])
            max_n = max(max_n, n)
        max_reps[geno] = max_n

    # build output: rows = conditions
    rows = []
    for cond in conditions:
        row = [cond]
        for geno in genotypes:
            vals = valid.loc[(valid['Genotype'] == geno) &
                             (valid['Condition'] == cond),
                             'Length_cm'].tolist()
            n_cols = max_reps[geno]
            vals_padded = vals + [''] * (n_cols - len(vals))
            row.extend(vals_padded)
        rows.append(row)

    # build header: first col empty, then genotype name + empty cols
    header = ['']
    for geno in genotypes:
        header.append(geno)
        header.extend([''] * (max_reps[geno] - 1))

    df_out = pd.DataFrame(rows, columns=header)
    df_out.to_csv(csv_path, index=False)


def _read_prism_factorial(csv_path, known_genotypes):
    """Try to read an existing Prism factorial CSV back into long format."""
    try:
        df = pd.read_csv(csv_path, header=0)
        if len(df) == 0 or len(df.columns) < 2:
            return None

        # detect genotype columns (non-empty header values after first)
        headers = list(df.columns)
        geno_blocks = []  # list of (geno_name, start_col_idx, end_col_idx)
        current_geno = None
        start = None
        for ci in range(1, len(headers)):
            h = str(headers[ci]).strip()
            if h and h != '' and h != 'Unnamed':
                if current_geno is not None:
                    geno_blocks.append((current_geno, start, ci))
                current_geno = h
                start = ci
        if current_geno is not None:
            geno_blocks.append((current_geno, start, len(headers)))

        if not geno_blocks:
            return None

        # reconstruct long format
        rows = []
        for _, row in df.iterrows():
            cond = row.iloc[0]
            for geno_name, col_start, col_end in geno_blocks:
                for ci in range(col_start, col_end):
                    val = row.iloc[ci]
                    if pd.notna(val) and str(val).strip() != '':
                        rows.append({
                            'Genotype': geno_name,
                            'Condition': cond,
                            'Length_cm': float(val),
                            'Warning': '',
                        })
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def get_offsets_from_csv(csv_path):
    """Read existing CSV and return (plate_offset, root_offset).

    plate_offset = max Plate value found (0 if no file).
    root_offset  = total number of rows already in the file.
    """
    try:
        if not csv_path.exists():
            return 0, 0
        df = pd.read_csv(csv_path)
        if df.empty:
            return 0, 0
        plate_off = 0
        if 'Plate' in df.columns:
            plate_off = int(df['Plate'].max())
        # use row count; Root_IDs are re-numbered sequentially on save
        root_off = len(df)
        return plate_off, root_off
    except Exception:
        return 0, 0


def append_results_to_csv(results, csv_path, plates, plate_labels, plate_offset,
                          root_offset, point_plates, num_marks=0,
                          split_plate=False, csv_format='R', image_name=''):
    """Append measurements to a shared CSV file.

    Args:
        results: list of measurement dicts from trace_root
        csv_path: Path to the data.csv file
        plates: list of (r1, r2, c1, c2) plate regions
        plate_labels: list of (genotype, condition) tuples
        plate_offset: starting plate number (0-based, so first image = 0)
        root_offset: starting root number (0-based)
        point_plates: list indicating which group each root belongs to
        num_marks: number of marks per root (0 = normal mode)
        split_plate: if True, point_plates stores group indices (2 per plate)
        csv_format: 'R' for tidy/long format, 'Prism' for wide format
        image_name: filename of the source image

    Returns:
        (new_plate_offset, new_root_offset) for the next image
    """
    rows, is_factorial = _build_rows(
        results, plate_labels, plate_offset, root_offset,
        point_plates, num_marks, split_plate, image_name=image_name)

    df_new = pd.DataFrame(rows)

    if csv_format == 'Prism':
        if is_factorial:
            _save_prism_factorial(df_new, csv_path)
        else:
            _save_prism_simple(df_new, csv_path)
    else:
        _save_r_format(df_new, csv_path, is_factorial, num_marks)

    print(f"\nResults appended to: {csv_path} ({csv_format} format)")

    new_plate_offset = plate_offset + len(plates)
    new_root_offset = root_offset + len(results)
    return new_plate_offset, new_root_offset


def save_metadata(meta_path, **kwargs):
    """Append one row of measurement metadata to a CSV log file.

    Each row records the settings used for one image, enabling
    full reproducibility of the measurement session.
    """
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
