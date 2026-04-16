"""Audio+vision late-fusion transformer on CMU-MOSEI (text-free baseline).

Drops the text modality from the standard `affect_mm.py --fusion 3` (lf_tran)
setup. The aim is a viability check: if A+V alone clears the threshold set by
E2-under-text-noise in the DynMM robust eval (~0.71 accuracy at sigma=1.0),
then a three-branch model with a text-free expert is worth building — the gate
would have a real rescue branch to route to. If A+V alone is far below that,
MOSEI is too text-dominated for any routing policy to rescue text corruption.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from datasets.affect.get_data import get_dataloader  # noqa: E402
from fusions.common_fusions import Concat  # noqa: E402
from training_structures.Supervised_Learning import MMDL, test, train  # noqa: E402
from unimodals.common_models import MLP, Transformer  # noqa: E402


class AVOnlyMMDL(MMDL):
    """MMDL that uses only the first two modality slots (vision, audio).

    MOSEI batches carry three modalities [vision, audio, text]; slicing the
    input before delegating to MMDL.forward keeps the encoder/fusion/head
    wiring identical to the 3-modality late-fusion baseline.
    """

    def forward(self, inputs):
        if self.has_padding:
            sliced = [inputs[0][:2], inputs[1][:2]]
        else:
            sliced = inputs[:2]
        return super().forward(sliced)


def parse_args():
    p = argparse.ArgumentParser("audio+vision LF transformer on MOSEI")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data", type=str, default="mosei", choices=["mosei", "mosi"])
    p.add_argument("--n-runs", type=int, default=1)
    p.add_argument("--n-epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    datafile = {"mosi": "mosi_raw", "mosei": "mosei_senti_data"}
    traindata, validdata, testdata = get_dataloader(
        "./data/" + datafile[args.data] + ".pkl",
        robust_test=False, data_type=args.data, num_workers=0,
    )

    filename = os.path.join("./log", args.data, "lf_tran_av.pt")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    log = np.zeros((args.n_runs, 3))
    for n in range(args.n_runs):
        encoders = [
            Transformer(35, 60).cuda(),    # vision
            Transformer(74, 120).cuda(),   # audio
        ]
        head = MLP(180, 128, 1).cuda()     # 60 + 120
        fusion = Concat().cuda()
        model = AVOnlyMMDL(encoders, fusion, head, has_padding=True).cuda()

        if not args.eval_only:
            train(
                None, None, None, traindata, validdata, args.n_epochs,
                task="regression", optimtype=torch.optim.AdamW,
                is_packed=True, early_stop=True, lr=args.lr,
                save=filename, weight_decay=args.wd,
                objective=torch.nn.L1Loss(), moe_model=model,
            )

        print(f"Testing model {filename}:")
        model = torch.load(filename).cuda()

        print("-" * 30 + " Val data " + "-" * 30)
        test(model=model, test_dataloaders_all=validdata, dataset=args.data,
             is_packed=True, criterion=torch.nn.L1Loss(),
             task="posneg-classification", no_robust=True)

        print("-" * 30 + " Test data " + "-" * 30)
        tmp = test(model=model, test_dataloaders_all=testdata, dataset=args.data,
                   is_packed=True, criterion=torch.nn.L1Loss(),
                   task="posneg-classification", no_robust=True)
        log[n] = tmp["Accuracy"], tmp["Loss"], tmp["Corr"]

    print(log)
    print(f"Finish {args.n_runs} runs")
    print(f"Test Accuracy {np.mean(log[:, 0]) * 100:.2f} ± {np.std(log[:, 0]) * 100:.2f}")
    print(f"Loss {np.mean(log[:, 1]):.4f} ± {np.std(log[:, 1]):.4f}")
    print(f"Corr {np.mean(log[:, 2]):.4f} ± {np.std(log[:, 2]):.4f}")

    results_dir = os.path.join("./log", args.data, "results")
    os.makedirs(results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"lf_tran_av_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "checkpoint": filename,
            "per_run": {
                "accuracy": log[:, 0].tolist(),
                "loss":     log[:, 1].tolist(),
                "corr":     log[:, 2].tolist(),
            },
            "mean": {
                "accuracy": float(np.mean(log[:, 0])),
                "loss":     float(np.mean(log[:, 1])),
                "corr":     float(np.mean(log[:, 2])),
            },
            "std": {
                "accuracy": float(np.std(log[:, 0])),
                "loss":     float(np.std(log[:, 1])),
                "corr":     float(np.std(log[:, 2])),
            },
        }, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
