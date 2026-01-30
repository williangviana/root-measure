"""Publication-quality box plots with statistics for root measurement data."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations


# -- colors matching user's publication style --
COLOR_DARK = '#555555'    # dark gray
COLOR_LIGHT = '#F5DEB3'   # wheat / light gold
COLORS = [COLOR_DARK, COLOR_LIGHT, '#A9A9A9', '#FAEBD7', '#808080', '#FFE4B5']


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
            letter_groups = new_groups

    # absorb: remove groups that are subsets of other groups
    cleaned = []
    for lg in letter_groups:
        if not any(lg < other for other in letter_groups if other is not lg):
            cleaned.append(lg)
    # deduplicate
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
    genotypes = df['Genotype'].unique().tolist()
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
    genotypes = df['Genotype'].unique().tolist()
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


def plot_results(csv_path):
    """Read CSV, run statistics, generate and save publication box plot."""
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

    # prompt user for column
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

    # prompt for y-axis label
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

    # generate plot
    fig, ax = plt.subplots(figsize=(3.5 + (1.5 if is_factorial else 1) *
                                    len(df['Genotype'].unique()), 5))

    genotypes = df['Genotype'].unique().tolist()
    positions_map = {}

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
                                boxprops=dict(facecolor=COLORS[gi % len(COLORS)],
                                              edgecolor='black', linewidth=1),
                                whiskerprops=dict(color='black', linewidth=1),
                                capprops=dict(color='black', linewidth=1))
                jitter = np.random.default_rng(42 + ci * 10 + gi).uniform(
                    -box_width * 0.3, box_width * 0.3, size=len(d))
                ax.scatter(pos + jitter, d, color='black', s=12, alpha=0.6,
                           zorder=3, edgecolors='none')

        ax.set_xticks(range(n_cond))
        ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=12)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i % len(COLORS)],
                                 edgecolor='black') for i in range(n_geno)]
        ax.legend(handles, genotypes, loc='upper right', fontsize=11,
                  frameon=True, edgecolor='black')
    else:
        conditions = []
        colors = [COLORS[i % len(COLORS)] for i in range(len(genotypes))]
        for i, geno in enumerate(genotypes):
            d = df.loc[df['Genotype'] == geno, value_col].dropna().values
            if len(d) == 0:
                continue
            positions_map[geno] = i
            bp = ax.boxplot([d], positions=[i], widths=0.5, patch_artist=True,
                            showfliers=False, zorder=2,
                            medianprops=dict(color='black', linewidth=1.5),
                            boxprops=dict(facecolor=colors[i], edgecolor='black',
                                          linewidth=1),
                            whiskerprops=dict(color='black', linewidth=1),
                            capprops=dict(color='black', linewidth=1))
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(d))
            ax.scatter(i + jitter, d, color='black', s=15, alpha=0.6,
                       zorder=3, edgecolors='none')

        ax.set_xticks(range(len(genotypes)))
        ax.set_xticklabels(genotypes, rotation=45, ha='right', fontsize=12)

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(bottom=0)

    # styling: remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
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

    plt.show()
