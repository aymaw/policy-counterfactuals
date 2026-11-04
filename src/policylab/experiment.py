import argparse
import json
import time
from pathlib import Path

import numpy as np

from .core import crossfit, evaluate, generate, reward_means, target_policy
from .visual import save, table


def run(out, quick=False):
    start = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 600 if quick else 6000
    repetitions = 2 if quick else 30
    # One large independent context sample approximates population truth.
    oracle_x = np.random.default_rng(98765).normal(size=(100000, 3))
    values = (target_policy(oracle_x) * reward_means(oracle_x)).sum(1)
    truth = float(values.mean())
    truth_mc_se = float(values.std(ddof=1) / np.sqrt(len(values)))
    records = []
    for regime in ["good", "poor"]:
        for seed in range(repetitions):
            data = generate(n, 100 + seed, regime)
            q, fold = crossfit(data["x"], data["actions"], data["rewards"], seed=400 + seed)
            result = evaluate(data["actions"], data["rewards"], data["behavior"], data["target"], q)
            clipped = evaluate(
                data["actions"], data["rewards"], data["behavior"], data["target"], q, clip=10.0
            )
            record = {
                "regime": regime,
                "seed": seed,
                **result,
                "clipped_DR": clipped["estimates"]["DR"],
                "clipped_fraction": clipped["clipped_fraction"],
            }
            records.append(record)
    summary = []
    for regime in ["good", "poor"]:
        selected = [r for r in records if r["regime"] == regime]
        for method in ["DM", "IPS", "SNIPS", "DR", "clipped_DR"]:
            v = np.array(
                [r[method] if method == "clipped_DR" else r["estimates"][method] for r in selected]
            )
            coverage = None
            if method in ["DM", "IPS", "DR"]:
                coverage = float(
                    np.mean(
                        [
                            abs(r["estimates"][method] - truth) <= 1.96 * r["se"][method]
                            for r in selected
                        ]
                    )
                )
            summary.append(
                {
                    "overlap": regime,
                    "method": method,
                    "mean": round(float(v.mean()), 4),
                    "bias": round(float((v - truth).mean()), 4),
                    "RMSE": round(float(np.sqrt(np.mean((v - truth) ** 2))), 4),
                    "95% normal interval coverage": coverage,
                }
            )
    payload = {
        "quick": quick,
        "n": n,
        "repetitions": repetitions,
        "truth": truth,
        "truth_mc_se": truth_mc_se,
        "seconds": time.perf_counter() - start,
        "summary": summary,
        "records": records,
        "reward_model": "cross-fitted linear ridge, deliberately misspecified",
        "logging": "known synthetic randomized propensity; no hidden confounding",
    }
    body = (
        '<p class="tag">Independent Monte Carlo truth: ' + f"{truth:.4f} ± {truth_mc_se:.5f} MC SE"
        '</p><div class="card"><label>Overlap <select id="regime"><option>good</option>'
        '<option>poor</option></select></label><label>Logged dataset <input id="seed" '
        f'type="range" min="0" max="{repetitions - 1}" value="0"></label>'
        '<p id="stats"></p><canvas id="plot" width="900" height="400" role="img" '
        'aria-label="Policy value estimates compared to oracle truth"></canvas></div>'
        "<h2>Repeated independent logs</h2>"
        + table(
            summary, ["overlap", "method", "mean", "bias", "RMSE", "95% normal interval coverage"]
        )
        + "<p>Normal intervals are diagnostics, not finite-sample guarantees. "
        "DR uses out-of-fold reward predictions and the correct known propensity. "
        "DM is deliberately misspecified; clipping trades variance for bias and loses "
        "the exact unbiasedness property. ESS is computed from unmodified importance weights. "
        "All policies and data-generating processes are fixed before evaluating rewards.</p>"
    )
    script = """
    const c=document.getElementById('plot'),ctx=c.getContext('2d');
    const regime=document.getElementById('regime'),seed=document.getElementById('seed');
    function draw(){const r=DATA.records.find(r=>r.regime===regime.value&&r.seed===Number(seed.value));
      ctx.clearRect(0,0,900,400);const methods=['DM','IPS','SNIPS','DR','clipped_DR'];
      const vals=methods.map(m=>m==='clipped_DR'?r[m]:r.estimates[m]);
      const low=Math.min(.3,...vals)-.03,high=Math.max(.9,...vals)+.03;
      const x=v=>130+(v-low)/(high-low)*700;
      ctx.strokeStyle='#ffb86c';ctx.beginPath();ctx.moveTo(x(DATA.truth),20);ctx.lineTo(x(DATA.truth),360);ctx.stroke();
      methods.forEach((m,i)=>{const y=60+i*65;ctx.fillStyle='#e9f0ff';ctx.font='16px system-ui';ctx.fillText(m,10,y+4);
        const se=r.se[m];if(se!==undefined){ctx.strokeStyle='#66e4cc';ctx.beginPath();ctx.moveTo(x(vals[i]-1.96*se),y);ctx.lineTo(x(vals[i]+1.96*se),y);ctx.stroke()}
        ctx.fillStyle='#66e4cc';ctx.beginPath();ctx.arc(x(vals[i]),y,6,0,Math.PI*2);ctx.fill();
        ctx.fillText(vals[i].toFixed(3),x(vals[i])+10,y-8)});
      document.getElementById('stats').textContent=`ESS ${r.ess.toFixed(0)}/${DATA.n} · max weight ${r.max_weight.toFixed(1)} · weak overlap flag: ${r.weak_overlap} · orange line: population truth`;
    }regime.onchange=draw;seed.oninput=draw;draw();
    """
    save(
        out,
        payload,
        "Policy Counterfactuals",
        "Can you trust a policy estimate from decisions somebody else made? Compare "
        "direct modeling, importance weighting and cross-fitted doubly robust evaluation.",
        body,
        script,
    )
    # Static SVG of absolute error, independently readable in a GitHub README.
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 470">',
        '<rect width="800" height="470" fill="#101f36"/>',
        '<text x="24" y="35" fill="#e9f0ff" font-family="sans-serif" font-size="22">Policy value RMSE — lower is better</text>',
    ]
    for i, row in enumerate(summary):
        y = 68 + i * 35
        lines.append(
            f'<text x="24" y="{y + 15}" fill="#e9f0ff" font-family="sans-serif">{row["overlap"]} / {row["method"]}</text>'
        )
        lines.append(
            f'<rect x="220" y="{y}" width="{row["RMSE"] * 4000:.2f}" height="20" fill="#66e4cc"/>'
        )
        lines.append(
            f'<text x="680" y="{y + 15}" fill="#e9f0ff" font-family="sans-serif">{row["RMSE"]}</text>'
        )
    (out / "preview.svg").write_text("".join(lines) + "</svg>")
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/reference")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run(a.out, a.quick)
