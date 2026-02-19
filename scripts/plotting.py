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
        dict of genotype -> CLD letter string
    """
    genotypes = sort_genotypes_wt_first(df['Genotype'].unique().tolist())
    groups = [df.loc[df['Genotype'] == g, value_col].dropna().values
              for g in genotypes]

    if len(genotypes) < 2:
        return {g: '' for g in genotypes}

    if len(genotypes) == 2:
        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        print(f"  Welch's t-test: t={t_stat:.3f}, p={p_val:.4f}")
        pairwise = {(genotypes[0], genotypes[1]): p_val}
    else:
        # one-way ANOVA
        f_stat, p_anova = stats.f_oneway(*groups)
        print(f"  One-way ANOVA: F={f_stat:.3f}, p={p_anova:.4f}")
        # Tukey HSD
        result = stats.tukey_hsd(*groups)
        pairwise = {}
        for i, j in combinations(range(len(genotypes)), 2):
            p_val = result.pvalue[i, j]
            pairwise[(genotypes[i], genotypes[j])] = p_val
            print(f"    {genotypes[i]} vs {genotypes[j]}: p={p_val:.4f}")

    return _compact_letter_display(genotypes, pairwise)


def _run_statistics_factorial(df, value_col):
    """Run statistics for factorial design (genotype x condition).

    Returns:
        dict of (genotype, condition) -> CLD letter string
    """
    genotypes = sort_genotypes_wt_first(df['Genotype'].unique().tolist())
    conditions = df['Condition'].unique().tolist()

    # build all group combinations that exist in data
    group_keys = []
    groups = []
    for cond in conditions:
        for geno in genotypes:
            subset = df.loc[(df['Genotype'] == geno) & (df['Condition'] == cond),
                            value_col].dropna().values
            if len(subset) > 0:
                group_keys.append((geno, cond))
                groups.append(subset)

    if len(group_keys) < 2:
        return {k: '' for k in group_keys}

    # one-way ANOVA on all groups (treating each geno×cond as a separate group)
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"  ANOVA (all groups): F={f_stat:.3f}, p={p_anova:.4f}")

    # Tukey HSD on all groups
    result = stats.tukey_hsd(*groups)
    pairwise = {}
    for i, j in combinations(range(len(group_keys)), 2):
        p_val = result.pvalue[i, j]
        pairwise[(group_keys[i], group_keys[j])] = p_val
        gi, gj = group_keys[i], group_keys[j]
        print(f"    {gi[0]}|{gi[1]} vs {gj[0]}|{gj[1]}: p={p_val:.4f}")

    return _compact_letter_display(group_keys, pairwise)


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

    # detect experiment type
    is_factorial = 'Condition' in df.columns and df['Condition'].notna().any()

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
    print("\n  Statistics:")
    if is_factorial:
        cld = _run_statistics_factorial(df, value_col)
    else:
        cld = _run_statistics_simple(df, value_col)

    print("\n  Letters:", cld)

    # generate plot - dynamic width based on number of groups
    # Fixed box width, figure scales horizontally
    n_genotypes = len(df['Genotype'].unique())
    box_width = 0.5  # consistent box width for all plots
    if is_factorial:
        n_conditions = len(df['Condition'].unique())
        fig_width = 1.5 + n_conditions * 1.0 + 1.2  # margin + conditions + legend
    else:
        fig_width = 1.5 + n_genotypes * 0.7  # margin + genotypes
    fig_width = max(fig_width, 3.0)  # minimum width
    fig, ax = plt.subplots(figsize=(fig_width, 6))

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
        box_width = 0.6 / n_geno

        for ci, cond in enumerate(conditions):
            for gi, geno in enumerate(genotypes):
                d = df.loc[(df['Genotype'] == geno) & (df['Condition'] == cond),
                           value_col].dropna().values
                if len(d) == 0:
                    continue
                pos = ci + (gi - (n_geno - 1) / 2) * (box_width + 0.05)
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

        ax.set_xticks(range(n_cond))
        ax.set_xticklabels(conditions, fontsize=12)
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

    # save PNG
    png_path = csv_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  Plot saved to: {png_path}")

    plt.close(fig)
