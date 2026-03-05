"""Publication-quality box plots with statistics for root measurement data."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
from config import sort_genotypes_wt_first


# -- bright colors for up to 30 genotypes (matches canvas trace colors) --
COLORS = [
    "#0072B2", "#E69F00", "#D55E00", "#56B4E9", "#009E73",
    "#F0E442", "#CC79A7", "#AA4400", "#882255", "#44AA99",
    "#AA4499", "#999933", "#6699CC", "#DD7788", "#117733",
    "#88CCEE", "#CC6677", "#DDCC77", "#332288", "#44BB99",
    "#EE8866", "#BBCC33", "#EEDD88", "#77AADD", "#EE6677",
    "#66CCEE", "#AA3377", "#BBBB33", "#4477AA", "#228833",
]


def _compact_letter_display(group_names, pairwise_pvalues, alpha=0.05):
    """Assign compact letter display (CLD) from pairwise p-values.

    Uses the insert-absorb algorithm from Piepho (2004).

    Args:
        group_names: list of group names
        pairwise_pvalues: dict of (name_i, name_j) -> p-value
        alpha: significance level

    Returns:
        dict of group_name -> letter string (e.g. 'a', 'ab', 'b')
    """
    n = len(group_names)
    if n == 0:
        return {}
    if n == 1:
        return {group_names[0]: 'a'}

    # build significance lookup
    def is_sig(a, b):
        p = pairwise_pvalues.get((a, b), pairwise_pvalues.get((b, a), 1.0))
        return p < alpha

    # letter_groups: list of frozensets, each is a group of non-different treatments
    # start with one group containing all treatments
    letter_groups = [set(group_names)]

    # for each significantly different pair, split groups that contain both
    for i in range(n):
        for j in range(i + 1, n):
            a, b = group_names[i], group_names[j]
            if not is_sig(a, b):
                continue
            # a and b must NOT share any letter group
            new_groups = []
            for lg in letter_groups:
                if a in lg and b in lg:
                    # split: create one group without a, one without b
                    ga = lg - {a}
                    gb = lg - {b}
                    if ga:
                        new_groups.append(ga)
                    if gb:
                        new_groups.append(gb)
                else:
                    new_groups.append(lg)
            # deduplicate after each split to prevent exponential growth
            seen = set()
            deduped = []
            for lg in new_groups:
                key = frozenset(lg)
                if key not in seen:
                    seen.add(key)
                    deduped.append(lg)
            letter_groups = deduped

    # absorb: remove groups that are subsets of other groups
    cleaned = []
    for lg in letter_groups:
        if not any(lg < other for other in letter_groups if other is not lg):
            cleaned.append(lg)
    # final deduplicate
    unique = []
    seen = set()
    for lg in cleaned:
        key = frozenset(lg)
        if key not in seen:
            seen.add(key)
            unique.append(lg)
    letter_groups = unique

    # sort groups by the mean rank of their first member (preserves input order)
    order = {name: i for i, name in enumerate(group_names)}
    letter_groups.sort(key=lambda lg: min(order[m] for m in lg))

    # assign letters
    letters = {name: [] for name in group_names}
    for i, lg in enumerate(letter_groups):
        letter = chr(ord('a') + i)
        for name in group_names:
            if name in lg:
                letters[name].append(letter)

    return {name: ''.join(sorted(ltrs)) for name, ltrs in letters.items()}


def _run_statistics_simple(df, value_col):
    """Run statistics for simple design (genotype only).

    Returns:
        (cld, stats_info) where cld is dict of genotype -> CLD letter string
        and stats_info is a dict with test details and pairwise results.
    """
    genotypes = sort_genotypes_wt_first(df['Genotype'].unique().tolist())
    groups = [df.loc[df['Genotype'] == g, value_col].dropna().values
              for g in genotypes]
    info = {'group_keys': genotypes, 'groups': groups, 'pairwise': {}}

    if len(genotypes) < 2:
        return {g: '' for g in genotypes}, info

    if len(genotypes) == 2:
        t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        info['test'] = "Welch's t-test"
        info['stat_name'] = 't'
        info['stat_val'] = t_stat
        info['p_val'] = p_val
        pairwise = {(genotypes[0], genotypes[1]): p_val}
    else:
        f_stat, p_anova = stats.f_oneway(*groups)
        info['test'] = 'One-way ANOVA'
        info['stat_name'] = 'F'
        info['stat_val'] = f_stat
        info['p_val'] = p_anova
        result = stats.tukey_hsd(*groups)
        pairwise = {}
        for i, j in combinations(range(len(genotypes)), 2):
            pairwise[(genotypes[i], genotypes[j])] = result.pvalue[i, j]

    info['pairwise'] = pairwise
    cld = _compact_letter_display(genotypes, pairwise)
    return cld, info


def _run_statistics_factorial(df, value_col):
    """Run statistics for factorial design (genotype x condition).

    Returns:
        (cld, stats_info) where cld is dict of (genotype, condition) -> CLD
        letter string and stats_info has test details and pairwise results.
    """
    genotypes = sort_genotypes_wt_first(df['Genotype'].unique().tolist())
    conditions = df['Condition'].unique().tolist()

    group_keys = []
    groups = []
    for cond in conditions:
        for geno in genotypes:
            subset = df.loc[(df['Genotype'] == geno) & (df['Condition'] == cond),
                            value_col].dropna().values
            if len(subset) > 0:
                group_keys.append((geno, cond))
                groups.append(subset)

    info = {'group_keys': group_keys, 'groups': groups, 'pairwise': {}}

    if len(group_keys) < 2:
        return {k: '' for k in group_keys}, info

    f_stat, p_anova = stats.f_oneway(*groups)
    info['test'] = 'One-way ANOVA'
    info['stat_name'] = 'F'
    info['stat_val'] = f_stat
    info['p_val'] = p_anova

    result = stats.tukey_hsd(*groups)
    pairwise = {}
    for i, j in combinations(range(len(group_keys)), 2):
        pairwise[(group_keys[i], group_keys[j])] = result.pvalue[i, j]

    info['pairwise'] = pairwise
    cld = _compact_letter_display(group_keys, pairwise)
    return cld, info


def format_statistics_summary(df, value_col, is_factorial, stats_info, cld):
    """Format a human-readable statistics summary."""
    line = '─' * 45
    lines = []

    # title
    title = value_col.replace('_', ' ').upper()
    if value_col == 'Length_cm':
        title = 'PRIMARY ROOT LENGTH'
    elif value_col.startswith('Segment_') and value_col.endswith('_cm'):
        seg = value_col.replace('Segment_', '').replace('_cm', '')
        title = f'SEGMENT {seg} LENGTH'
    lines.append('═' * 45)
    lines.append(title)
    lines.append('═' * 45)
    lines.append('')

    # descriptive statistics
    lines.append('Descriptive Statistics')
    lines.append(line)
    group_keys = stats_info['group_keys']
    groups = stats_info['groups']
    # determine label width
    if is_factorial:
        labels = [f'{k[0]} | {k[1]}' for k in group_keys]
    else:
        labels = list(group_keys)
    max_lbl = max((len(l) for l in labels), default=10)
    col_w = max(max_lbl, 10)
    header = f'{"Group":<{col_w}}  {"n":>3}  {"Mean":>7}  {"SD":>7}  {"SE":>7}'
    lines.append(header)
    lines.append(line)
    for lbl, g in zip(labels, groups):
        n = len(g)
        mean = np.mean(g)
        sd = np.std(g, ddof=1) if n > 1 else 0
        se = sd / np.sqrt(n) if n > 0 else 0
        lines.append(f'{lbl:<{col_w}}  {n:>3}  {mean:>7.2f}  {sd:>7.2f}  {se:>7.2f}')
    lines.append('')

    # overall test
    lines.append('Overall Test')
    lines.append(line)
    test = stats_info.get('test', '')
    if test:
        sn = stats_info['stat_name']
        sv = stats_info['stat_val']
        pv = stats_info['p_val']
        sig = ' *' if pv < 0.05 else ''
        lines.append(f'{test}: {sn} = {sv:.3f}, p = {pv:.4f}{sig}')
    else:
        lines.append('Not enough groups for statistical testing.')
    lines.append('')

    # pairwise comparisons — only for 3+ groups (Tukey HSD)
    pairwise = stats_info.get('pairwise', {})
    if len(group_keys) > 2 and pairwise:
        sig_pairs = {k: v for k, v in pairwise.items() if v < 0.05}
        lines.append('Significant Pairwise Differences (Tukey HSD, p < 0.05)')
        lines.append(line)
        if sig_pairs:
            for (a, b), p in sorted(sig_pairs.items(), key=lambda x: x[1]):
                if is_factorial:
                    la = f'{a[0]} | {a[1]}'
                    lb = f'{b[0]} | {b[1]}'
                else:
                    la, lb = a, b
                lines.append(f'  {la}  vs  {lb}: p = {p:.4f}')
        else:
            lines.append('  No significant differences found.')
        lines.append('')

    # CLD
    lines.append('Compact Letter Display')
    lines.append(line)
    lines.append('Groups sharing a letter are not significantly')
    lines.append('different (p > 0.05).')
    lines.append('')
    for lbl, key in zip(labels, group_keys):
        letter = cld.get(key, '')
        lines.append(f'  {lbl:<{col_w}}  {letter}')
    lines.append('')

    return '\n'.join(lines)


def _place_cld_letters(ax, is_factorial, df, value_col, genotypes, conditions,
                       cld, positions_map):
    """Place CLD letters after axis limits are finalized."""
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    offset = y_range * 0.03

    if not is_factorial:
        for i, geno in enumerate(genotypes):
            d = df.loc[df['Genotype'] == geno, value_col].dropna().values
            if len(d) == 0:
                continue
            letter = cld.get(geno, '')
            if not letter:
                continue
            q75 = np.percentile(d, 75)
            q25 = np.percentile(d, 25)
            iqr = q75 - q25
            upper_whisker = min(q75 + 1.5 * iqr, d.max())
            top = max(upper_whisker, d.max())
            pos = positions_map[geno]
            ax.text(pos, top + offset, letter, ha='center', va='bottom',
                    fontsize=12, fontweight='normal')
    else:
        for (geno, cond), pos in positions_map.items():
            d = df.loc[(df['Genotype'] == geno) & (df['Condition'] == cond),
                       value_col].dropna().values
            if len(d) == 0:
                continue
            letter = cld.get((geno, cond), '')
            if not letter:
                continue
            q75 = np.percentile(d, 75)
            q25 = np.percentile(d, 25)
            iqr = q75 - q25
            upper_whisker = min(q75 + 1.5 * iqr, d.max())
            top = max(upper_whisker, d.max())
            ax.text(pos, top + offset, letter, ha='center', va='bottom',
                    fontsize=10, fontweight='normal')


def _read_prism_simple(csv_path):
    """Read Prism simple format (genotypes as columns) into long format."""
    df = pd.read_csv(csv_path)
    rows = []
    for col in df.columns:
        for val in df[col].dropna():
            rows.append({'Genotype': col, 'Length_cm': float(val)})
    return pd.DataFrame(rows)


def _read_prism_factorial(csv_path):
    """Read Prism factorial format into long format."""
    df = pd.read_csv(csv_path, header=0)
    headers = list(df.columns)
    # detect genotype blocks: non-empty headers after first column
    geno_blocks = []
    current_geno = None
    start = None
    for ci in range(1, len(headers)):
        h = str(headers[ci]).strip()
        if h and h != '' and not h.startswith('Unnamed'):
            if current_geno is not None:
                geno_blocks.append((current_geno, start, ci))
            current_geno = h
            start = ci
    if current_geno is not None:
        geno_blocks.append((current_geno, start, len(headers)))

    rows = []
    for _, row in df.iterrows():
        cond = row.iloc[0]
        for geno_name, col_start, col_end in geno_blocks:
            for ci in range(col_start, col_end):
                val = row.iloc[ci]
                if pd.notna(val) and str(val).strip() != '':
                    rows.append({
                        'Genotype': geno_name,
                        'Condition': str(cond),
                        'Length_cm': float(val),
                    })
    return pd.DataFrame(rows)


def plot_results(csv_path, value_col=None, ylabel=None, csv_format='R',
                  genotype_colors=None):
    """Read CSV, run statistics, generate and save publication box plot.

    Args:
        csv_path: Path to the CSV file.
        value_col: Column to plot (default: prompt user or 'Length_cm').
        ylabel: Y-axis label (default: prompt user or auto-generate).
        csv_format: 'R' or 'Prism'.
        genotype_colors: Optional dict mapping genotype name -> color index.
    """
    if csv_format == 'Prism':
        # try factorial first (has condition column = first col non-numeric)
        df_test = pd.read_csv(csv_path, header=0, nrows=1)
        first_col = str(df_test.columns[0]).strip()
        first_val = str(df_test.iloc[0, 0]).strip() if len(df_test) > 0 else ''
        # factorial has a non-empty first column header (usually '') and
        # condition labels in first column
        if first_col == '' or first_col.startswith('Unnamed'):
            df = _read_prism_factorial(csv_path)
        else:
            df = _read_prism_simple(csv_path)
        value_col = value_col or 'Length_cm'
    else:
        df = pd.read_csv(csv_path)

    # drop rows with warnings
    if 'Warning' in df.columns:
        df = df[df['Warning'].fillna('').astype(str).str.strip() == ''].copy()

    if len(df) == 0:
        print("  No valid data to plot.")
        return

    # detect experiment type — factorial only with 2+ distinct conditions
    is_factorial = ('Condition' in df.columns
                    and df['Condition'].nunique(dropna=True) >= 2)

    # find plottable columns
    numeric_cols = ['Length_cm']
    for col in df.columns:
        if col.startswith('Segment_') and col.endswith('_cm'):
            numeric_cols.append(col)
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # resolve column to plot
    if value_col is None:
        if len(numeric_cols) == 1:
            value_col = numeric_cols[0]
        else:
            print("\n  Available columns to plot:")
            for i, col in enumerate(numeric_cols, 1):
                print(f"    {i}. {col}")
            while True:
                choice = input(f"\n  Column to plot (1-{len(numeric_cols)}, default: 1): ").strip()
                if choice == '':
                    value_col = numeric_cols[0]
                    break
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(numeric_cols):
                        value_col = numeric_cols[idx]
                        break
                except ValueError:
                    pass
                print("  Invalid choice.")

    # resolve y-axis label
    if ylabel is None:
        default_label = 'Primary root length (cm)' if value_col == 'Length_cm' else f'{value_col} (cm)'
        ylabel = input(f"\n  Y-axis label (default: {default_label}): ").strip()
        if not ylabel:
            ylabel = default_label

    print(f"\n  Plotting: {value_col}")
    print(f"  Design: {'factorial' if is_factorial else 'simple'}")

    # convert to numeric
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])

    # run statistics
    if is_factorial:
        cld, stats_info = _run_statistics_factorial(df, value_col)
    else:
        cld, stats_info = _run_statistics_simple(df, value_col)
    summary = format_statistics_summary(
        df, value_col, is_factorial, stats_info, cld)

    # generate plot - dynamic width based on number of groups
    # Fixed box width, figure scales horizontally
    n_genotypes = len(df['Genotype'].unique())
    box_width = 0.5  # consistent box width for all plots
    if is_factorial:
        n_conditions = len(df['Condition'].unique())
        group_width = max(n_genotypes * 0.4, 1.0)
        fig_width = 0.8 + n_conditions * group_width + 1.2  # margin + conditions + legend
    else:
        fig_width = 1.5 + n_genotypes * 0.55  # margin + genotypes
    fig_width = max(fig_width, 3.0)  # minimum width
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    genotypes = sort_genotypes_wt_first(df['Genotype'].unique().tolist())
    positions_map = {}

    def _geno_color(geno, fallback_idx):
        if genotype_colors and geno in genotype_colors:
            return COLORS[genotype_colors[geno] % len(COLORS)]
        return COLORS[fallback_idx % len(COLORS)]

    if is_factorial:
        conditions = df['Condition'].unique().tolist()
        n_geno = len(genotypes)
        n_cond = len(conditions)
        box_width = max(0.6 / n_geno, 0.15)

        box_gap = 0.05
        group_half = ((n_geno - 1) / 2) * (box_width + box_gap) + box_width / 2
        inter_gap = 0.25
        cond_spacing = group_half * 2 + inter_gap

        for ci, cond in enumerate(conditions):
            cx = ci * cond_spacing
            for gi, geno in enumerate(genotypes):
                d = df.loc[(df['Genotype'] == geno) & (df['Condition'] == cond),
                           value_col].dropna().values
                if len(d) == 0:
                    continue
                pos = cx + (gi - (n_geno - 1) / 2) * (box_width + box_gap)
                positions_map[(geno, cond)] = pos

                bp = ax.boxplot([d], positions=[pos], widths=box_width,
                                patch_artist=True, showfliers=False, zorder=2,
                                medianprops=dict(color='black', linewidth=1.5),
                                boxprops=dict(facecolor=_geno_color(geno, gi),
                                              edgecolor='black', linewidth=1),
                                whiskerprops=dict(color='black', linewidth=1),
                                capprops=dict(color='black', linewidth=1))
                jitter = np.random.default_rng(42 + ci * 10 + gi).uniform(
                    -box_width * 0.3, box_width * 0.3, size=len(d))
                ax.scatter(pos + jitter, d, color='black', s=12, alpha=0.6,
                           zorder=3, edgecolors='none')

        ax.set_xticks([ci * cond_spacing for ci in range(n_cond)])
        ax.set_xticklabels(conditions, fontsize=12)
        # tighten x-axis around actual box positions
        all_pos = list(positions_map.values())
        if all_pos:
            margin = box_width / 2 + inter_gap
            ax.set_xlim(min(all_pos) - margin, max(all_pos) + margin)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=_geno_color(genotypes[i], i),
                                 edgecolor='black') for i in range(n_geno)]
        ax.legend(handles, genotypes, loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=11, frameon=True, edgecolor='black',
                  handlelength=1, handleheight=1)
    else:
        conditions = []
        colors = [_geno_color(g, i) for i, g in enumerate(genotypes)]
        for i, geno in enumerate(genotypes):
            d = df.loc[df['Genotype'] == geno, value_col].dropna().values
            if len(d) == 0:
                continue
            positions_map[geno] = i
            bp = ax.boxplot([d], positions=[i], widths=0.35, patch_artist=True,
                            showfliers=False, zorder=2,
                            medianprops=dict(color='black', linewidth=1.5),
                            boxprops=dict(facecolor=colors[i], edgecolor='black',
                                          linewidth=1),
                            whiskerprops=dict(color='black', linewidth=1),
                            capprops=dict(color='black', linewidth=1))
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(d))
            ax.scatter(i + jitter, d, color='black', s=15, alpha=0.6,
                       zorder=3, edgecolors='none')

        ax.set_xticks(range(len(genotypes)))
        ax.set_xticklabels(genotypes, fontsize=12)

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(bottom=0)

    # styling: frame with all four spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2)

    # finalize y limits then place CLD letters
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.1)  # add 10% headroom for letters

    _place_cld_letters(ax, is_factorial, df, value_col, genotypes, conditions,
                       cld, positions_map)

    plt.tight_layout()

    # save PNG — descriptive filenames
    if value_col and value_col.startswith('Segment_'):
        seg_num = value_col.replace('Segment_', '').replace('_cm', '')
        png_path = csv_path.with_name(f'segment_{seg_num}_length.png')
    else:
        png_path = csv_path.with_name('primary_root_length.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  Plot saved to: {png_path}")

    plt.close(fig)
    return summary


def plot_segments_facet(csv_path, seg_cols, csv_format='R',
                        genotype_colors=None):
    """Generate a facet plot with one subplot per segment, shared y-axis."""
    if csv_format == 'Prism':
        df_test = pd.read_csv(csv_path, header=0, nrows=1)
        first_col = str(df_test.columns[0]).strip()
        if first_col == '' or first_col.startswith('Unnamed'):
            df = _read_prism_factorial(csv_path)
        else:
            df = _read_prism_simple(csv_path)
    else:
        df = pd.read_csv(csv_path)

    if 'Warning' in df.columns:
        df = df[df['Warning'].fillna('').astype(str).str.strip() == ''].copy()
    if len(df) == 0:
        return

    is_factorial = ('Condition' in df.columns
                    and df['Condition'].nunique(dropna=True) >= 2)
    genotypes = sort_genotypes_wt_first(df['Genotype'].unique().tolist())
    conditions = df['Condition'].unique().tolist() if is_factorial else []
    n_segs = len(seg_cols)

    def _geno_color(geno, fallback_idx):
        if genotype_colors and geno in genotype_colors:
            return COLORS[genotype_colors[geno] % len(COLORS)]
        return COLORS[fallback_idx % len(COLORS)]

    # figure sizing
    n_geno = len(genotypes)
    if is_factorial:
        n_cond = len(conditions)
        panel_w = max(0.8 + n_cond * max(n_geno * 0.4, 1.0), 2.5)
    else:
        panel_w = max(0.8 + n_geno * 0.55, 2.0)
    legend_w = 1.4 if is_factorial else 0
    fig_w = panel_w * n_segs + legend_w + 0.5
    fig, axes = plt.subplots(1, n_segs, sharey=True,
                             figsize=(fig_w, 5))
    if n_segs == 1:
        axes = [axes]

    for si, sc in enumerate(seg_cols):
        ax = axes[si]
        seg_num = sc.replace('Segment_', '').replace('_cm', '')
        df[sc] = pd.to_numeric(df[sc], errors='coerce')
        positions_map = {}

        if is_factorial:
            n_cond = len(conditions)
            box_width = max(0.6 / n_geno, 0.15)
            box_gap = 0.05
            group_half = ((n_geno - 1) / 2) * (box_width + box_gap) + box_width / 2
            inter_gap = 0.25
            cond_spacing = group_half * 2 + inter_gap

            for ci, cond in enumerate(conditions):
                cx = ci * cond_spacing
                for gi, geno in enumerate(genotypes):
                    d = df.loc[(df['Genotype'] == geno) & (df['Condition'] == cond),
                               sc].dropna().values
                    if len(d) == 0:
                        continue
                    pos = cx + (gi - (n_geno - 1) / 2) * (box_width + box_gap)
                    positions_map[(geno, cond)] = pos
                    ax.boxplot([d], positions=[pos], widths=box_width,
                               patch_artist=True, showfliers=False, zorder=2,
                               medianprops=dict(color='black', linewidth=1.5),
                               boxprops=dict(facecolor=_geno_color(geno, gi),
                                             edgecolor='black', linewidth=1),
                               whiskerprops=dict(color='black', linewidth=1),
                               capprops=dict(color='black', linewidth=1))
                    jitter = np.random.default_rng(42 + ci * 10 + gi).uniform(
                        -box_width * 0.3, box_width * 0.3, size=len(d))
                    ax.scatter(pos + jitter, d, color='black', s=10, alpha=0.6,
                               zorder=3, edgecolors='none')

            ax.set_xticks([ci * cond_spacing for ci in range(n_cond)])
            ax.set_xticklabels(conditions, fontsize=10)
            all_pos = list(positions_map.values())
            if all_pos:
                margin = box_width / 2 + inter_gap
                ax.set_xlim(min(all_pos) - margin, max(all_pos) + margin)
        else:
            for gi, geno in enumerate(genotypes):
                d = df.loc[df['Genotype'] == geno, sc].dropna().values
                if len(d) == 0:
                    continue
                positions_map[geno] = gi
                ax.boxplot([d], positions=[gi], widths=0.35,
                           patch_artist=True, showfliers=False, zorder=2,
                           medianprops=dict(color='black', linewidth=1.5),
                           boxprops=dict(facecolor=_geno_color(geno, gi),
                                         edgecolor='black', linewidth=1),
                           whiskerprops=dict(color='black', linewidth=1),
                           capprops=dict(color='black', linewidth=1))
                jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(d))
                ax.scatter(gi + jitter, d, color='black', s=12, alpha=0.6,
                           zorder=3, edgecolors='none')
            ax.set_xticks(range(n_geno))
            ax.set_xticklabels(genotypes, fontsize=10)

        ax.set_title(f'Segment {seg_num}', fontsize=12, fontweight='bold')
        ax.set_ylim(bottom=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

        # CLD letters
        if is_factorial:
            cld, _ = _run_statistics_factorial(df, sc)
        else:
            cld, _ = _run_statistics_simple(df, sc)
        ax.autoscale_view()
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax * 1.1)
        _place_cld_letters(ax, is_factorial, df, sc, genotypes, conditions,
                           cld, positions_map)

    axes[0].set_ylabel('Segment length (cm)', fontsize=13)

    # shared legend
    n_geno = len(genotypes)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=_geno_color(genotypes[i], i),
                              edgecolor='black') for i in range(n_geno)]
    axes[-1].legend(handles, genotypes, loc='upper left',
                    bbox_to_anchor=(1.02, 1), fontsize=10, frameon=True,
                    edgecolor='black', handlelength=1, handleheight=1)

    plt.tight_layout()
    png_path = csv_path.with_name('segments_facet.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  Facet plot saved to: {png_path}")
    plt.close(fig)
