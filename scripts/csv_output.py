import pandas as pd


def append_results_to_csv(results, csv_path, plates, plate_labels, plate_offset,
                          root_offset, point_plates, num_marks=0,
                          split_plate=False):
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

    Returns:
        (new_plate_offset, new_root_offset) for the next image
    """
    # check if this is a factorial design (any plate has a condition)
    is_factorial = plate_labels and any(cond is not None for (geno, cond) in plate_labels)

    # count plants per group to assign Plant_ID (restarts at 1 per group)
    group_counters = {}

    rows = []
    for i, r in enumerate(results):
        root_num = root_offset + i + 1
        row = {
            'Root_ID': root_num,
            'Length_cm': round(r['length_cm'], 3),
            'Length_px': round(r['length_px'], 1),
            'Warning': r['warning'] or '',
        }

        if plate_labels and i < len(point_plates):
            group_idx = point_plates[i]
            if split_plate:
                # group_idx 0,1 = plate 0 genotypes; 2,3 = plate 1 genotypes
                physical_plate = group_idx // 2
            else:
                physical_plate = group_idx
            row['Plate'] = plate_offset + physical_plate + 1
            genotype, condition = plate_labels[group_idx]
            row['Genotype'] = genotype
            if is_factorial:
                row['Condition'] = condition or ''

            # Plant_ID: per-group counter (restarts at 1 for each genotype group)
            group_counters[group_idx] = group_counters.get(group_idx, 0) + 1
            row['Plant_ID'] = group_counters[group_idx]

        # add segment columns if multi-measurement mode
        segments = r.get('segments', [])
        if num_marks > 0:
            # num_marks marks divide the root into (num_marks + 1) segments
            for seg_i in range(num_marks + 1):
                col_name = f'Segment_{seg_i + 1}_cm'
                if seg_i < len(segments):
                    row[col_name] = round(segments[seg_i], 3)
                else:
                    row[col_name] = ''

        rows.append(row)

    # set column order based on experiment type
    if is_factorial:
        col_order = ['Root_ID', 'Plate', 'Genotype', 'Plant_ID', 'Condition', 'Length_cm']
    else:
        col_order = ['Root_ID', 'Plate', 'Genotype', 'Plant_ID', 'Length_cm']

    # add segment columns
    if num_marks > 0:
        for seg_i in range(num_marks + 1):
            col_order.append(f'Segment_{seg_i + 1}_cm')

    col_order.extend(['Length_px', 'Warning'])

    df_new = pd.DataFrame(rows)

    # if file exists, append; otherwise create
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        # ensure columns match
        for col in col_order:
            if col not in df_existing.columns:
                df_existing[col] = ''
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = ''
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    # only include columns that exist
    col_order = [c for c in col_order if c in df.columns]
    # add any extra columns from existing file that aren't in col_order
    for col in df.columns:
        if col not in col_order:
            col_order.append(col)
    df = df[col_order]
    df.to_csv(csv_path, index=False)
    print(f"\nResults appended to: {csv_path}")

    # return updated offsets
    new_plate_offset = plate_offset + len(plates)
    new_root_offset = root_offset + len(results)
    return new_plate_offset, new_root_offset
