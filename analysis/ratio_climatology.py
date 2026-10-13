"""
ratio_climatology.py

How the vespagram catalogue's extra detections are distributed: the ratio of
vespagram to ROI occupancy (share of hours with an LSTID) as a function of
auroral activity (SME, Hp30; 3 h trailing means), solar cycle (year), UT hour
and month. A flat ratio means the vespagram finds proportionally more LSTIDs
everywhere; a ratio rising towards quiet conditions means its extra detections
are concentrated where LSTIDs are rare (where false alarms would show up first).

95% intervals: day-block bootstrap (whole days resampled, since hours within a
day are correlated).

Reads the hour pickles of catalogue_records.py.
Output: results/climatology_compare/ratio.png, ratio.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_climatology import (OUT, METHODS, common_grid, outcome, load,
                                 driver_on_grid, _style)

N_BOOT = 300
SEED   = 0
GROUPS = {
    "SME [nT], 3 h mean": ("SME", [0, 50, 100, 150, 200, 300, 400, 500, 700, 1000, np.inf]),
    "Hp30, 3 h mean":     ("Hp30", [0, 1, 2, 3, 4, 5, 6, np.inf]),
    "Year":               ("year", None),
    "UT hour":            ("hour", None),
    "Month":              ("month", None),
}
VESPA = ["vespagram_hm", "vespagram"]


def binned(grid):
    ts = pd.to_datetime(grid.to_numpy(), utc=True)
    out = {"day": ts.normalize().to_numpy()}
    for label, (key, edges) in GROUPS.items():
        if edges is None:
            x = getattr(ts, key).to_numpy()
            out[label] = (x, np.unique(x).astype(str))
        else:
            x = driver_on_grid(key, grid, z=False)
            b = np.digitize(x, edges[1:-1]).astype(float)
            b[~np.isfinite(x)] = np.nan
            names = [f"{edges[i]:g}–{edges[i + 1]:g}" if np.isfinite(edges[i + 1])
                     else f"{edges[i]:g}+" for i in range(len(edges) - 1)]
            out[label] = (b, np.array(names))
    return out


def ratio_table(ys, bins):
    """Per group and level: occupancy of each catalogue, ratio to ROI and its CI."""
    day = bins["day"]
    _, day_idx = np.unique(day, return_inverse=True)
    n_day = day_idx.max() + 1
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(n_day, np.full(n_day, 1 / n_day), size=N_BOOT)    # day weights

    rows = []
    for label in GROUPS:
        x, names = bins[label]
        levels = np.unique(x[np.isfinite(x)]) if x.dtype.kind == "f" else np.unique(x)
        for lv in levels:
            m = x == lv
            # per-day sums of active hours in this level
            hours = np.bincount(day_idx[m], minlength=n_day)
            act = {k: np.bincount(day_idx[m], weights=ys[k][m], minlength=n_day) for k in ys}
            name = names[int(lv)] if x.dtype.kind == "f" else str(lv)
            row = {"group": label, "level": name, "hours": int(m.sum())}
            for k in ys:
                row[f"occ_{k}"] = act[k].sum() / hours.sum() * 100
            for k in VESPA:
                boot = (W @ act[k]) / np.maximum(W @ act["roi"], 1e-9)
                row[f"ratio_{k}"] = act[k].sum() / max(act["roi"].sum(), 1e-9)
                row[f"lo_{k}"], row[f"hi_{k}"] = np.percentile(boot, [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)


def figure(tab, ys, path):
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(4.4 * len(GROUPS), 7.6),
                             gridspec_kw={"height_ratios": [1, 1.3]}, sharex="col")
    overall = {k: ys[k].sum() / ys["roi"].sum() for k in VESPA}
    for c, label in enumerate(GROUPS):
        d = tab[tab["group"] == label].reset_index(drop=True)
        x = np.arange(len(d))
        top, bot = axes[0, c], axes[1, c]
        for k in ["roi"] + VESPA:
            name, color = METHODS[k][:2]
            top.plot(x, d[f"occ_{k}"], marker="o", ms=3.5, lw=1.8, color=color, label=name)
        top.set_title(label, fontsize=11)
        _style(top)
        for k in VESPA:
            name, color = METHODS[k][:2]
            bot.fill_between(x, d[f"lo_{k}"], d[f"hi_{k}"], color=color, alpha=0.18, lw=0)
            bot.plot(x, d[f"ratio_{k}"], marker="o", ms=3.5, lw=1.8, color=color,
                     label=f"{name} / ROI")
            bot.axhline(overall[k], color=color, lw=1, ls="--")
        bot.set_yscale("log")
        bot.set_yticks([1, 1.5, 2, 3, 5, 10, 20], ["1", "1.5", "2", "3", "5", "10", "20"])
        bot.minorticks_off()
        _style(bot)
        step = 1 if len(d) <= 12 else 3
        bot.set_xticks(x[::step], d["level"][::step], rotation=45 if c < 2 else 0, fontsize=8)
    axes[0, 0].set_ylabel("Hours with an LSTID [%]")
    axes[1, 0].set_ylabel("Ratio vespagram / ROI  (log scale)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[1, 0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Vespagram vs ROI occupancy.  Bottom: ratio with 95% day-block bootstrap "
                 "interval; dashed = overall ratio "
                 f"(high+medium {overall['vespagram_hm']:.2f}, all {overall['vespagram']:.2f})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    grid = common_grid()
    ys = {k: outcome(grid, load(k, "hours")) for k in ["roi"] + VESPA}
    tab = ratio_table(ys, binned(grid))
    tab.to_csv(OUT / "ratio.csv", index=False)
    figure(tab, ys, OUT / "ratio.png")
    cols = ["level", "hours", "occ_roi", "occ_vespagram_hm", "occ_vespagram",
            "ratio_vespagram_hm", "lo_vespagram_hm", "hi_vespagram_hm",
            "ratio_vespagram", "lo_vespagram", "hi_vespagram"]
    for g in ("SME [nT], 3 h mean", "Hp30, 3 h mean", "Year"):
        print(f"\n{g}")
        print(tab[tab["group"] == g][cols].round(2).to_string(index=False))
    print(f"\nsaved -> {OUT / 'ratio.png'}")


if __name__ == "__main__":
    main()
