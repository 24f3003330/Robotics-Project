"""
python -m body_vitals.analyze [--features CSV]

Descriptive analysis of the WESAD feature table on the DEVELOPMENT subjects only
(the held-out test subjects in config.TEST_SUBJECTS are dropped before any
statistic is computed, so feature engineering is never informed by them).

Writes body_vitals/results/analysis/feature_analysis.json and prints:
  * class distribution (all subjects - counts only)
  * per feature and class: mean, median, std, IQR
  * separability for each class pair:
      pooled AUC        Mann-Whitney AUC over all windows (0.5 = no separation)
      within-subject d  median over subjects of (median_A - median_B) / subject SD
      sign consistency  number of subjects in which median_A > median_B
  * between-subject share of variance (variance of subject means / total variance)
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np

from . import config
from .dataset import load_features


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=config.FEATURES_CSV)
    p.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "analysis", "feature_analysis.json"))
    return p.parse_args(argv)


def auc(a, b):
    """P(a > b) + 0.5 P(a == b) - Mann-Whitney AUC."""
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(mannwhitneyu(a, b).statistic / (len(a) * len(b)))


def analyze(df, feature_names):
    out = {"features": {}}
    pairs = list(combinations(config.CLASSES, 2))
    for f in feature_names:
        info = {"by_class": {}, "pairs": {}}
        for c in config.CLASSES:
            x = df.loc[df["class"] == c, f]
            info["by_class"][c] = {"mean": float(x.mean()), "median": float(x.median()), "std": float(x.std()),
                                   "iqr": float(x.quantile(0.75) - x.quantile(0.25))}
        subj_means = df.groupby("subject")[f].mean()
        info["between_subject_variance_share"] = float(subj_means.var() / df[f].var()) if df[f].var() > 0 else 0.0
        for a, b in pairs:
            ds, signs = [], 0
            for _, g in df.groupby("subject"):
                xa, xb = g.loc[g["class"] == a, f], g.loc[g["class"] == b, f]
                sd = g[f].std()
                if len(xa) and len(xb) and sd > 0:
                    d = (xa.median() - xb.median()) / sd
                    ds.append(d)
                    signs += int(d > 0)
            info["pairs"][f"{a}_vs_{b}"] = {
                "pooled_auc": auc(df.loc[df["class"] == a, f], df.loc[df["class"] == b, f]),
                "within_subject_d_median": float(np.median(ds)) if ds else float("nan"),
                "subjects_a_greater": int(signs), "n_subjects": len(ds)}
        out["features"][f] = info
    return out


def run(args):
    df_all = load_features(args.features)
    out = {"class_distribution_all_subjects": df_all["class"].value_counts().reindex(config.CLASSES).to_dict()}
    df = df_all[~df_all["subject"].isin(config.TEST_SUBJECTS)]
    assert not set(df["subject"]) & set(config.TEST_SUBJECTS)
    feature_names = [f for f in config.ALL_FEATURE_NAMES if f in df.columns]
    out["development_subjects"] = sorted(df["subject"].unique(), key=lambda s: (len(s), s))
    out["class_distribution_development"] = df["class"].value_counts().reindex(config.CLASSES).to_dict()
    out.update(analyze(df, feature_names))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"Class distribution (all 15 subjects): {out['class_distribution_all_subjects']}")
    print(f"Statistics below use the {len(out['development_subjects'])} development subjects only "
          f"(test subjects {config.TEST_SUBJECTS} excluded).\n")
    print(f"{'feature':<20} {'between-subj':>12}   " + "   ".join(
        f"{a[:4]}-{b[:4]} AUC  d  n>" for a, b in combinations(config.CLASSES, 2)))
    for f, info in out["features"].items():
        cells = []
        for key, v in info["pairs"].items():
            cells.append(f"{v['pooled_auc']:.2f} {v['within_subject_d_median']:+.2f} "
                         f"{v['subjects_a_greater']:>2}/{v['n_subjects']}")
        print(f"{f:<20} {info['between_subject_variance_share']:>12.2f}   " + "   ".join(cells))
    print("\nAUC: pooled over windows. d: median within-subject standardized difference (A - B). "
          "n>: subjects with median A > median B.")
    print(f"Wrote {args.out}")
    return out


if __name__ == "__main__":
    run(parse_args())
