"""Eval-only noise robustness sweep for clean-trained DynMM checkpoints.

Fills the gap in the original sweep: the np=0.0 checkpoints were never evaluated
under test-time Gaussian text noise. To answer "does noise-augmented DynMM beat
normal DynMM on noisy data?" we need robustness numbers for the clean-trained
checkpoints too.

The clean checkpoints are DynMMNetV2 instances (no noise-injection forward).
We class-swap each loaded checkpoint to DynMMNetV2Noisy — the subclass adds a
corrupt-input step before the shared forward but introduces no new learnable
parameters, so the swap is lossless — and then run the standard robust-eval
loop at text sigma in {0.3, 0.5, 1.0}.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.insert(0, os.path.join(os.getcwd(), "examples/affect"))

from affect_dyn import DynMMNetV2, DynMMNetV2Noisy  # noqa: E402
from datasets.affect.get_data import get_dataloader  # noqa: E402
from training_structures.Supervised_Learning import test  # noqa: E402


REGS = [0.001, 0.01, 0.1]
SIGMAS = [0.3, 0.5, 1.0]
DATA = "mosei"
RESULTS_DIR = os.path.join("./log", DATA, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def promote_to_noisy(model):
    """Class-swap a clean DynMMNetV2 into DynMMNetV2Noisy and set noise attrs."""
    assert isinstance(model, DynMMNetV2) and not isinstance(model, DynMMNetV2Noisy)
    model.__class__ = DynMMNetV2Noisy
    model.text_noise_prob = 0.0
    model.text_noise_std = 0.0
    model.text_noise_mode = "gaussian"
    model.eval_noise_std = 0.0
    model.eval_noise_prob = 1.0
    model.eval_noise_modality = 2  # text
    return model


def main():
    _, validdata, testdata = get_dataloader(
        "./data/mosei_senti_data.pkl",
        robust_test=False, data_type=DATA, num_workers=0,
    )

    for reg in REGS:
        ckpt = f"./log/{DATA}/dyn_enc_transformer_reg_{reg}freezeFalse.pt"
        print("=" * 60)
        print(f"Eval clean-trained DynMM: reg={reg}  ckpt={ckpt}")
        print("=" * 60)

        model = torch.load(ckpt).cuda()
        model = promote_to_noisy(model)
        model.infer_mode = 0  # adaptive gate
        model.eval()

        # Clean re-test (sanity — should match original clean numbers)
        model.eval_noise_std = 0.0
        model.reset_weight()
        clean = test(model=model, test_dataloaders_all=testdata, dataset=DATA,
                     is_packed=True, criterion=torch.nn.L1Loss(reduction="sum"),
                     task="posneg-classification", no_robust=True, additional_loss=True)
        clean_flop = model.cal_flop()
        clean_ratio = model.weight_stat()
        print(f"  clean: acc={clean['Accuracy']:.4f} corr={clean['Corr']:.4f} "
              f"flop={clean_flop:.1f} ratio={clean_ratio:.3f}")

        robust_log = {}
        for sigma in SIGMAS:
            model.eval_noise_std = sigma
            model.reset_weight()
            print("-" * 20 + f" reg={reg} text sigma={sigma} " + "-" * 20)
            tmp = test(model=model, test_dataloaders_all=testdata, dataset=DATA,
                       is_packed=True, criterion=torch.nn.L1Loss(reduction="sum"),
                       task="posneg-classification", no_robust=True, additional_loss=True)
            flop = model.cal_flop()
            ratio = model.weight_stat()
            robust_log[str(sigma)] = {
                "Accuracy": tmp["Accuracy"], "Loss": tmp["Loss"],
                "Corr": tmp["Corr"], "FLOP": flop, "E2_ratio": ratio,
            }
            print(f"  sigma={sigma}: acc={tmp['Accuracy']:.4f} corr={tmp['Corr']:.4f} "
                  f"flop={flop:.1f} ratio={ratio:.3f}")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            RESULTS_DIR,
            f"dyn_enc_transformer_reg_{reg}_freezeFalse_cleaneval_{stamp}.json",
        )
        with open(out_path, "w") as f:
            json.dump({
                "reg": reg,
                "checkpoint": ckpt,
                "text_noise_prob_train": 0.0,
                "clean": {
                    "Accuracy": clean["Accuracy"], "Loss": clean["Loss"],
                    "Corr": clean["Corr"], "FLOP": clean_flop, "E2_ratio": clean_ratio,
                },
                "robustness_text": robust_log,
            }, f, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
