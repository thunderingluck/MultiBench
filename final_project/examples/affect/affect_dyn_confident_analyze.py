"""Analyze confidence-gate results vs. existing DynMM baselines.

Compares four cells under text noise (sigma in {clean, 0.3, 0.5, 1.0}):
  1. DynMMNetV2          np=0.0  (clean-trained, content-only gate)         <- baseline
  2. DynMMNetV2Noisy     np=0.5  (noise-trained, content-only gate)
  3. DynMMNetV2Confident np=0.0  (clean-trained, confidence-aware gate)     <- key test
  4. DynMMNetV2Confident np=0.5  (noise-trained, confidence-aware gate)

The falsifiable prediction is that (3) routes toward E2 under text noise
(opposite of (1)) and matches or exceeds (2) on accuracy/correlation.
"""

import glob
import json
import os
import sys

REG = 0.01
RESULTS_DIR = "log/mosei/results"


def latest(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    return files[-1] if files else None


def load_baseline_clean():
    f = latest(f"dyn_enc_transformer_reg_{REG}_freezeFalse_cleaneval_*.json")
    return json.load(open(f)) if f else None


def load_baseline_noise05():
    files = sorted(glob.glob(os.path.join(
        RESULTS_DIR,
        f"dyn_enc_transformer_reg_{REG}_freezeFalse_np0.5_gaussian_2026*.json")))
    files = [f for f in files if "adapt" not in f and "_E1_" not in f and "_E2_" not in f]
    if not files:
        return None
    runs = []
    for f in files:
        d = json.load(open(f))
        rob = d.get("robustness", [])
        if rob and len(rob[0]) == 3:
            runs.append(d)
    return runs


def load_confident(np_train, normalize=False):
    suffix = "_norm" if normalize else ""
    f = latest(f"dyn_conf_enc_transformer_reg_{REG}_freezeFalse"
               + (f"_np{np_train}_gaussian" if np_train > 0 else "")
               + suffix + "_adapt_*.json")
    return json.load(open(f)) if f else None


def load_token_gate(np_train):
    f = latest(f"dyn_token_enc_transformer_reg_{REG}_freezeFalse"
               + (f"_np{np_train}_gaussian" if np_train > 0 else "")
               + "_adapt_*.json")
    return json.load(open(f)) if f else None


def fmt_row(label, clean, sigma_results):
    cells = [f"{clean.get('Accuracy', clean.get('accuracy', 0)):.4f}" if clean else "—"]
    for s in ["0.3", "0.5", "1.0"]:
        r = sigma_results.get(s) if sigma_results else None
        cells.append(f"{r['Accuracy']:.4f}" if r else "—")
    return f"  {label:<40}  " + "  ".join(f"{c:>8}" for c in cells)


def fmt_corr(label, clean, sigma_results):
    cells = [f"{clean.get('Corr', clean.get('corr', 0)):.4f}" if clean else "—"]
    for s in ["0.3", "0.5", "1.0"]:
        r = sigma_results.get(s) if sigma_results else None
        cells.append(f"{r['Corr']:.4f}" if r else "—")
    return f"  {label:<40}  " + "  ".join(f"{c:>8}" for c in cells)


def fmt_e2(label, clean_ratio, sigma_results):
    cells = [f"{clean_ratio:.3f}" if clean_ratio is not None else "—"]
    for s in ["0.3", "0.5", "1.0"]:
        r = sigma_results.get(s) if sigma_results else None
        cells.append(f"{r['E2_ratio']:.3f}" if r else "—")
    return f"  {label:<40}  " + "  ".join(f"{c:>8}" for c in cells)


def avg_runs(runs, sigma):
    if not runs:
        return None
    keys = ["Accuracy", "Loss", "Corr", "FLOP", "E2_ratio"]
    vals = {k: 0.0 for k in keys}
    for r in runs:
        for k in keys:
            vals[k] += r["robustness"][0][sigma][k]
    return {k: v / len(runs) for k, v in vals.items()}


def main():
    print("=" * 80)
    print(f"Confidence-aware DynMM analysis (reg={REG})")
    print("=" * 80)

    bl_clean = load_baseline_clean()
    bl_n05 = load_baseline_noise05()
    cf_n0 = load_confident(0.0)
    cf_n05 = load_confident(0.5)
    cfn_n0 = load_confident(0.0, normalize=True)
    cfn_n05 = load_confident(0.5, normalize=True)
    tk_n0 = load_token_gate(0.0)
    tk_n05 = load_token_gate(0.5)

    print()

    # === Accuracy ===
    print(f"  {'Configuration':<40}  " + "  ".join(f"{h:>8}" for h in ["clean", "σ=0.3", "σ=0.5", "σ=1.0"]))
    print(f"  {'-' * 40}  " + "  ".join("-" * 8 for _ in range(4)))
    print("  ACCURACY")
    if bl_clean:
        print(fmt_row("[1] DynMMNetV2 (np=0.0, content-gate)", bl_clean["clean"], bl_clean["robustness_text"]))
    if bl_n05:
        clean_avg = {"Accuracy": sum(r["mean"]["accuracy"] for r in bl_n05) / len(bl_n05)}
        sigma_avg = {s: avg_runs(bl_n05, s) for s in ["0.3", "0.5", "1.0"]}
        print(fmt_row("[2] DynMMNetV2Noisy (np=0.5, content-gate)", clean_avg, sigma_avg))
    if cf_n0:
        clean = {"Accuracy": cf_n0["mean"]["accuracy"]}
        sigma_results = cf_n0["robustness"][0]["text"] if cf_n0["robustness"] else {}
        print(fmt_row("[3] CONFIDENT (np=0.0, conf-gate)", clean, sigma_results))
    if cf_n05:
        clean = {"Accuracy": cf_n05["mean"]["accuracy"]}
        sigma_results = cf_n05["robustness"][0]["text"] if cf_n05["robustness"] else {}
        print(fmt_row("[4] CONFIDENT (np=0.5, conf-gate)", clean, sigma_results))
    if cfn_n0:
        clean = {"Accuracy": cfn_n0["mean"]["accuracy"]}
        sigma_results = cfn_n0["robustness"][0]["text"] if cfn_n0["robustness"] else {}
        print(fmt_row("[5] CONFIDENT-NORM (np=0.0, conf-norm)", clean, sigma_results))
    if cfn_n05:
        clean = {"Accuracy": cfn_n05["mean"]["accuracy"]}
        sigma_results = cfn_n05["robustness"][0]["text"] if cfn_n05["robustness"] else {}
        print(fmt_row("[6] CONFIDENT-NORM (np=0.5, conf-norm)", clean, sigma_results))
    if tk_n0:
        clean = {"Accuracy": tk_n0["mean"]["accuracy"]}
        sigma_results = tk_n0["robustness"][0]["text"] if tk_n0["robustness"] else {}
        print(fmt_row("[7] TOKEN-GATE (np=0.0)", clean, sigma_results))
    if tk_n05:
        clean = {"Accuracy": tk_n05["mean"]["accuracy"]}
        sigma_results = tk_n05["robustness"][0]["text"] if tk_n05["robustness"] else {}
        print(fmt_row("[8] TOKEN-GATE (np=0.5)", clean, sigma_results))

    print()
    print("  CORRELATION (phi coefficient — same metric as old tables)")
    if bl_clean:
        print(fmt_corr("[1] DynMMNetV2 (np=0.0, content-gate)", bl_clean["clean"], bl_clean["robustness_text"]))
    if bl_n05:
        clean_avg = {"Corr": sum(r["mean"]["corr"] for r in bl_n05) / len(bl_n05)}
        sigma_avg = {s: avg_runs(bl_n05, s) for s in ["0.3", "0.5", "1.0"]}
        print(fmt_corr("[2] DynMMNetV2Noisy (np=0.5, content-gate)", clean_avg, sigma_avg))
    if cf_n0:
        clean = {"Corr": cf_n0["mean"]["corr"]}
        sigma_results = cf_n0["robustness"][0]["text"] if cf_n0["robustness"] else {}
        print(fmt_corr("[3] CONFIDENT (np=0.0, conf-gate)", clean, sigma_results))
    if cf_n05:
        clean = {"Corr": cf_n05["mean"]["corr"]}
        sigma_results = cf_n05["robustness"][0]["text"] if cf_n05["robustness"] else {}
        print(fmt_corr("[4] CONFIDENT (np=0.5, conf-gate)", clean, sigma_results))
    if cfn_n0:
        clean = {"Corr": cfn_n0["mean"]["corr"]}
        sigma_results = cfn_n0["robustness"][0]["text"] if cfn_n0["robustness"] else {}
        print(fmt_corr("[5] CONFIDENT-NORM (np=0.0, conf-norm)", clean, sigma_results))
    if cfn_n05:
        clean = {"Corr": cfn_n05["mean"]["corr"]}
        sigma_results = cfn_n05["robustness"][0]["text"] if cfn_n05["robustness"] else {}
        print(fmt_corr("[6] CONFIDENT-NORM (np=0.5, conf-norm)", clean, sigma_results))
    if tk_n0:
        clean = {"Corr": tk_n0["mean"]["corr"]}
        sigma_results = tk_n0["robustness"][0]["text"] if tk_n0["robustness"] else {}
        print(fmt_corr("[7] TOKEN-GATE (np=0.0)", clean, sigma_results))
    if tk_n05:
        clean = {"Corr": tk_n05["mean"]["corr"]}
        sigma_results = tk_n05["robustness"][0]["text"] if tk_n05["robustness"] else {}
        print(fmt_corr("[8] TOKEN-GATE (np=0.5)", clean, sigma_results))

    print()
    print("  E2 ROUTING RATIO (key test: does (3) route toward E2 under text noise?)")
    if bl_clean:
        print(fmt_e2("[1] DynMMNetV2 (np=0.0, content-gate)", bl_clean["clean"]["E2_ratio"], bl_clean["robustness_text"]))
    if bl_n05:
        clean_avg = sum(r["mean"]["ratio"] for r in bl_n05) / len(bl_n05)
        sigma_avg = {s: avg_runs(bl_n05, s) for s in ["0.3", "0.5", "1.0"]}
        print(fmt_e2("[2] DynMMNetV2Noisy (np=0.5, content-gate)", clean_avg, sigma_avg))
    if cf_n0:
        sigma_results = cf_n0["robustness"][0]["text"] if cf_n0["robustness"] else {}
        print(fmt_e2("[3] CONFIDENT (np=0.0, conf-gate)", cf_n0["mean"]["ratio"], sigma_results))
    if cf_n05:
        sigma_results = cf_n05["robustness"][0]["text"] if cf_n05["robustness"] else {}
        print(fmt_e2("[4] CONFIDENT (np=0.5, conf-gate)", cf_n05["mean"]["ratio"], sigma_results))
    if cfn_n0:
        sigma_results = cfn_n0["robustness"][0]["text"] if cfn_n0["robustness"] else {}
        print(fmt_e2("[5] CONFIDENT-NORM (np=0.0, conf-norm)", cfn_n0["mean"]["ratio"], sigma_results))
    if cfn_n05:
        sigma_results = cfn_n05["robustness"][0]["text"] if cfn_n05["robustness"] else {}
        print(fmt_e2("[6] CONFIDENT-NORM (np=0.5, conf-norm)", cfn_n05["mean"]["ratio"], sigma_results))
    if tk_n0:
        sigma_results = tk_n0["robustness"][0]["text"] if tk_n0["robustness"] else {}
        print(fmt_e2("[7] TOKEN-GATE (np=0.0)", tk_n0["mean"]["ratio"], sigma_results))
    if tk_n05:
        sigma_results = tk_n05["robustness"][0]["text"] if tk_n05["robustness"] else {}
        print(fmt_e2("[8] TOKEN-GATE (np=0.5)", tk_n05["mean"]["ratio"], sigma_results))


if __name__ == "__main__":
    main()
