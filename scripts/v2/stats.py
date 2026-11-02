"""Policy-level paired statistics.

The unit is the frozen policy. Runs that share one are not independent, so
seeds are averaged within a policy first and the contrast is formed between
arm means inside that policy. Degrees of freedom follow from the number of
policies that contribute both arms, computed per contrast; no critical value
is fixed in advance.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class Contrast:
    name: str
    n: int
    df: int
    mean: float
    sd: float
    se: float
    ci_low: float
    ci_high: float
    t: float
    p_one_sided: float
    wins: int
    p_adjusted: float | None = None
    policies_dropped: int = 0
    runs_undefined: int = 0
    detail: dict = field(default_factory=dict)

    def line(self) -> str:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_one_sided
        return (f"{self.name:<34} n={self.n:3d} df={self.df:3d} "
                f"mean={self.mean:+.4f} sd={self.sd:.4f} se={self.se:.4f} "
                f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}] "
                f"p={p:.4f} wins {self.wins}/{self.n}")


def policy_means(rows, metric, policy_of, arm_of, require_defined=True):
    """Mean of ``metric`` per (policy, arm), and how many runs were undefined."""
    acc: dict = collections.defaultdict(list)
    undefined = 0
    for r in rows:
        v = r.get(metric)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            undefined += 1
            if require_defined:
                continue
        acc[(policy_of(r), arm_of(r))].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}, undefined


def paired(rows, metric, a, b, policy_of, arm_of, direction="greater",
           name=None) -> Contrast:
    """Paired contrast ``b - a`` over the policies carrying both arms.

    ``direction='greater'`` declares in advance that ``b`` is expected to score
    above ``a`` on ``metric``; the one-sided p value follows that declaration.
    """
    means, undefined = policy_means(rows, metric, policy_of, arm_of)
    pols = sorted({p for (p, arm) in means})
    d, dropped = [], 0
    for p in pols:
        if (p, a) in means and (p, b) in means:
            d.append(means[(p, b)] - means[(p, a)])
        else:
            dropped += 1
    d = np.asarray(d, float)
    n = d.size
    if n < 2:
        return Contrast(name or f"{b} - {a} on {metric}", n, max(n - 1, 0),
                        float(d.mean()) if n else float("nan"), float("nan"),
                        float("nan"), float("nan"), float("nan"), float("nan"),
                        float("nan"), int((d > 0).sum()),
                        policies_dropped=dropped, runs_undefined=undefined)
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    t = float(d.mean() / se) if se > 0 else float("inf")
    crit = float(stats.t.ppf(0.975, df))
    p2 = float(stats.t.sf(abs(t), df) * 2)
    p1 = p2 / 2 if (t > 0) == (direction == "greater") else 1 - p2 / 2
    return Contrast(name or f"{b} - {a} on {metric}", n, df, float(d.mean()),
                    sd, se, float(d.mean() - crit * se), float(d.mean() + crit * se),
                    t, p1, int((d > 0).sum()), policies_dropped=dropped,
                    runs_undefined=undefined, detail={"differences": d.tolist()})


def holm(contrasts) -> list:
    """Holm step-down adjustment over a pre-declared family, in place."""
    order = sorted(range(len(contrasts)), key=lambda i: contrasts[i].p_one_sided)
    m = len(contrasts)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * contrasts[i].p_one_sided)
        running = max(running, adj)
        contrasts[i].p_adjusted = running
    return contrasts


def reference_resample(rows, policy_of, arm_of, a, b, metric_bins="C_phi",
                       ref_counts="reference_counts", draws=400, seed=0):
    """How much of a contrast moves when the shared reference is resampled.

    The reference is one multinomial sample, so every arm in a policy carries
    its sampling error in the same direction. Redrawing the reference from its
    own counts and rescoring both arms separates that shared error from the
    difference between arms. The legacy runs cannot enter this: their reference
    counts were never stored.
    """
    rng = np.random.default_rng(seed)
    byp: dict = collections.defaultdict(list)
    for r in rows:
        if r.get(metric_bins) is None or r.get(ref_counts) is None:
            continue
        byp[policy_of(r)].append(r)
    out = []
    for _ in range(draws):
        diffs = []
        for p, members in byp.items():
            counts = np.asarray(members[0][ref_counts], float)
            tot = counts.sum()
            if tot <= 0:
                continue
            q = rng.multinomial(int(tot), counts / tot) / tot
            per_arm: dict = collections.defaultdict(list)
            for r in members:
                ph = np.asarray(r[metric_bins], float)
                per_arm[arm_of(r)].append(0.5 * np.abs(ph - q).sum())
            if a in per_arm and b in per_arm:
                diffs.append(np.mean(per_arm[a]) - np.mean(per_arm[b]))
        if len(diffs) > 1:
            out.append(float(np.mean(diffs)))
    arr = np.asarray(out)
    return dict(mean=float(arr.mean()), sd=float(arr.std(ddof=1)),
                lo=float(np.percentile(arr, 2.5)), hi=float(np.percentile(arr, 97.5)),
                draws=int(arr.size))
