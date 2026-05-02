"""Compute MOSEI standard multi-metric eval for DynMM checkpoints.

Motivation: Finding 6 showed that noise-augmented training improves regression
correlation under text noise but not binary accuracy. Both metrics are
derived from the same continuous prediction — binary accuracy only checks
the sign. This script adds the finer-grained MOSEI metrics (5-class and
7-class accuracy, MAE) to check whether the correlation improvement also
manifests in a finer classification bucket.

Outputs all standard MOSEI metrics (Acc-2, Acc-5, Acc-7, MAE, Corr, F1-2)
for clean-trained and noise-trained (np=0.5) DynMM checkpoints at text-noise
sigma in {clean, 0.3, 0.5, 1.0}.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.insert(0, os.path.join(os.getcwd(), "examples/affect"))

from affect_dyn import DynMMNetV2, DynMMNetV2Noisy  # noqa: E402
from datasets.affect.get_data import get_dataloader  # noqa: E402


REGS = [0.001, 0.01, 0.1]
NPS = [0.0, 0.5]  # clean-trained vs noise-trained
SIGMAS = [0.0, 0.3, 0.5, 1.0]  # 0.0 = clean eval
DATA = "mosei"
RESULTS_DIR = os.path.join("./log", DATA, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def promote_to_noisy(model):
    """Lossless class-swap so eval-time text-noise injection works."""
    if isinstance(model, DynMMNetV2Noisy):
        return model
    model.__class__ = DynMMNetV2Noisy
    model.text_noise_prob = 0.0
    model.text_noise_std = 0.0
    model.text_noise_mode = "gaussian"
    model.eval_noise_std = 0.0
    model.eval_noise_prob = 1.0
    model.eval_noise_modality = 2  # text
    return model


def checkpoint_name(reg, np_):
    if np_ > 0:
        return f"./log/{DATA}/dyn_enc_transformer_reg_{reg}freezeFalse_np{np_}_gaussian.pt"
    return f"./log/{DATA}/dyn_enc_transformer_reg_{reg}freezeFalse.pt"


def collect_predictions(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inp = [[x.float().cuda() for x in batch[0]], [x.cuda() for x in batch[1]]]
            out = model(inp)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.detach().cpu())
            labels.append(batch[-1])
    return torch.cat(preds), torch.cat(labels)


def mosei_metrics(preds, labels):
    """Standard MOSEI metric set."""
    p = preds.flatten().numpy().astype(np.float64)
    y = labels.flatten().numpy().astype(np.float64)

    mae = float(np.mean(np.abs(p - y)))
    corr = float(np.corrcoef(p, y)[0, 1])

    # Binary (pos/neg by sign, matching existing posneg-classification)
    p2 = (p >= 0).astype(int)
    y2 = (y >= 0).astype(int)
    acc2 = float(np.mean(p2 == y2))
    f1_2 = float(f1_score(y2, p2, average="weighted"))

    # 7-class (round, clip to [-3, 3])
    p7 = np.clip(np.round(p), -3, 3).astype(int)
    y7 = np.clip(np.round(y), -3, 3).astype(int)
    acc7 = float(np.mean(p7 == y7))

    # 5-class (round, clip to [-2, 2])
    p5 = np.clip(np.round(p), -2, 2).astype(int)
    y5 = np.clip(np.round(y), -2, 2).astype(int)
    acc5 = float(np.mean(p5 == y5))

    return {
        "Acc2": acc2, "Acc5": acc5, "Acc7": acc7,
        "MAE": mae, "Corr": corr, "F1_2": f1_2,
    }


def main():
    _, _, testdata = get_dataloader(
        "./data/mosei_senti_data.pkl",
        robust_test=False, data_type=DATA, num_workers=0,
    )

    results = {}
    for reg in REGS:
        for np_ in NPS:
            ckpt = checkpoint_name(reg, np_)
            if not os.path.exists(ckpt):
                print(f"skip (not found): {ckpt}")
                continue
            print("=" * 60)
            print(f"reg={reg}  np_train={np_}  ckpt={ckpt}")
            print("=" * 60)

            model = torch.load(ckpt).cuda()
            model = promote_to_noisy(model)
            model.infer_mode = 0  # adaptive
            model.eval()

            for sigma in SIGMAS:
                model.eval_noise_std = sigma
                model.reset_weight()
                preds, labels = collect_predictions(model, testdata)
                m = mosei_metrics(preds, labels)
                ratio = model.weight_stat()
                flop = model.cal_flop()
                m.update({"E2_ratio": float(ratio), "FLOP": float(flop)})
                key = f"reg={reg}|np={np_}|sigma={sigma}"
                results[key] = m
                print(f"  sigma={sigma}: Acc2={m['Acc2']:.4f} Acc5={m['Acc5']:.4f} "
                      f"Acc7={m['Acc7']:.4f} MAE={m['MAE']:.4f} Corr={m['Corr']:.4f}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"mosei_multimetric_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all metrics to {out_path}")


if __name__ == "__main__":
    main()
