"""Confidence-aware gate for DynMMNetV2.

Architectural extension to DynMMNetV2Noisy: the gate sees per-modality scalar
confidence signals (mean L2 norm of each modality's features) in addition to
the standard content features. The hypothesis is that this lets the gate
recognize an unreliable modality at inference time without needing to be
trained on corrupted data.

Falsifiable prediction: clean-trained confidence-aware gate should route
toward E2 under text corruption (current clean-trained baseline routes
*away* from E2 — see Table 5 of results.md), and should match or exceed
the noise-augmented baseline (np=0.5) on accuracy and correlation.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.insert(0, os.path.join(os.getcwd(), "examples/affect"))

from affect_dyn import DiffSoftmax, DynMMNetV2Noisy  # noqa: E402
from datasets.affect.get_data import get_dataloader  # noqa: E402
from training_structures.Supervised_Learning import test, train  # noqa: E402


class DynMMNetV2Confident(DynMMNetV2Noisy):
    """DynMMNetV2Noisy with a confidence-aware gate.

    The gate's classifier head receives per-modality L2-norm features in
    addition to the existing 10-dim content summary. Norms are computed
    after any noise injection, so the gate sees the same view as the
    branches do.
    """

    def __init__(self, temp, hard_gate, freeze, model_name_list,
                 text_noise_prob=0.0, text_noise_std=0.0, text_noise_mode="gaussian",
                 confidence_dim=3, normalize_confidence=False):
        super().__init__(temp, hard_gate, freeze, model_name_list,
                         text_noise_prob, text_noise_std, text_noise_mode)
        self.confidence_dim = confidence_dim
        self.normalize_confidence = normalize_confidence
        self.gate_transformer = self.gate[0]
        self.gate_classifier = nn.Linear(10 + confidence_dim, self.branch_num)
        if normalize_confidence:
            # BatchNorm1d standardizes each confidence dim across the batch
            # so the linear layer sees comparable scales for vision/audio/text norms.
            self.confidence_norm = nn.BatchNorm1d(confidence_dim)
        self.gate = None  # break the old Sequential reference

    def compute_confidence(self, inputs):
        # Mean L2 norm of each modality across its sequence dimension.
        # Returns shape [batch, n_modalities] = [batch, 3] for MOSEI.
        feats = []
        for x in inputs[0]:
            norm = x.detach().norm(dim=-1).mean(dim=-1)
            feats.append(norm)
        return torch.stack(feats, dim=-1)

    def forward(self, inputs):
        inputs = self._corrupt_text(inputs)
        confidence = self.compute_confidence(inputs)
        if self.normalize_confidence:
            confidence = self.confidence_norm(confidence)

        x = torch.cat(inputs[0], dim=2)
        h = self.gate_transformer([x, inputs[1][0]])
        gate_in = torch.cat([h, confidence], dim=-1)
        weight = DiffSoftmax(self.gate_classifier(gate_in), tau=self.temp, hard=self.hard_gate)

        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))

        pred_list = [
            self.text_head(self.text_encoder([inputs[0][2], inputs[1][2]])),
            self.branch2(inputs),
        ]
        if self.infer_mode > 0:
            return pred_list[self.infer_mode - 1], 0
        if self.infer_mode == -1:
            weight = torch.ones_like(weight) / self.branch_num

        output = weight[:, 0:1] * pred_list[0] + weight[:, 1:2] * pred_list[1]
        return output, weight[:, 1].mean()


def parse_args():
    p = argparse.ArgumentParser("Confidence-aware DynMM on MOSEI")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data", type=str, default="mosei", choices=["mosei", "mosi"])
    p.add_argument("--n-runs", type=int, default=1)
    p.add_argument("--enc", type=str, default="transformer")
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--hard-gate", action="store_true")
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--infer-mode", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--text-noise-prob", type=float, default=0.0)
    p.add_argument("--text-noise-std", type=float, default=0.0)
    p.add_argument("--text-noise-mode", type=str, default="gaussian")
    p.add_argument("--eval-noise-sigmas", type=str, default="0.3,0.5,1.0")
    p.add_argument("--eval-noise-modalities", type=str, default="text")
    p.add_argument("--normalize-confidence", action="store_true",
                   help="Apply BatchNorm1d to per-modality confidence inputs before gate fusion.")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    datafile = {"mosi": "mosi_raw", "mosei": "mosei_senti_data"}
    traindata, validdata, testdata = get_dataloader(
        "./data/" + datafile[args.data] + ".pkl",
        robust_test=False, data_type=args.data, num_workers=16,
    )

    model_name_list = ["./log/mosei/b1_reg_transformer_encoder_text.pt",
                       "./log/mosei/b2_lf_tran.pt"]
    noise_tag = (f"_np{args.text_noise_prob}_{args.text_noise_mode}"
                 if args.text_noise_prob > 0 else "")
    norm_tag = "_norm" if args.normalize_confidence else ""
    filename = os.path.join("./log", args.data,
                            "dyn_conf_enc_" + args.enc + "_reg_" + str(args.reg) +
                            "freeze" + str(args.freeze) + noise_tag + norm_tag + ".pt")

    log = np.zeros((args.n_runs, 5))
    robust_log_all = []

    for n in range(args.n_runs):
        model = DynMMNetV2Confident(
            args.temp, args.hard_gate, args.freeze, model_name_list,
            text_noise_prob=args.text_noise_prob,
            text_noise_std=args.text_noise_std,
            text_noise_mode=args.text_noise_mode,
            normalize_confidence=args.normalize_confidence,
        ).cuda()

        if not args.eval_only:
            train(None, None, None, traindata, validdata, args.n_epochs,
                  task="regression", optimtype=torch.optim.AdamW,
                  is_packed=True, early_stop=True, lr=args.lr, save=filename,
                  weight_decay=args.wd, objective=torch.nn.L1Loss(),
                  moe_model=model, additional_loss=True, lossw=args.reg)

        print(f"Testing model {filename}:")
        model = torch.load(filename).cuda()
        model.infer_mode = args.infer_mode

        # Clean test
        print("-" * 30 + " Val data " + "-" * 30)
        model.reset_weight()
        test(model=model, test_dataloaders_all=validdata, dataset=args.data,
             is_packed=True, criterion=torch.nn.L1Loss(reduction="sum"),
             task="posneg-classification", no_robust=True, additional_loss=True)

        model.reset_weight()
        print("-" * 30 + " Test data " + "-" * 30)
        tmp = test(model=model, test_dataloaders_all=testdata, dataset=args.data,
                   is_packed=True, criterion=torch.nn.L1Loss(reduction="sum"),
                   task="posneg-classification", no_robust=True, additional_loss=True)
        log[n] = tmp["Accuracy"], tmp["Loss"], tmp["Corr"], model.cal_flop(), model.weight_stat()

        if args.eval_noise_sigmas:
            sigmas = [float(s) for s in args.eval_noise_sigmas.split(",") if s.strip()]
            mod_name_to_idx = {"vision": 0, "audio": 1, "text": 2}
            mods = [m.strip() for m in args.eval_noise_modalities.split(",") if m.strip()]
            robust_log = {}
            for mod_name in mods:
                if mod_name not in mod_name_to_idx:
                    continue
                mod_idx = mod_name_to_idx[mod_name]
                robust_log[mod_name] = {}
                for sigma in sigmas:
                    model.eval_noise_std = sigma
                    model.eval_noise_prob = 1.0
                    model.eval_noise_modality = mod_idx
                    model.reset_weight()
                    print("-" * 20 + f" Test ({mod_name} sigma={sigma}) " + "-" * 20)
                    tmp = test(model=model, test_dataloaders_all=testdata, dataset=args.data,
                               is_packed=True, criterion=torch.nn.L1Loss(reduction="sum"),
                               task="posneg-classification", no_robust=True, additional_loss=True)
                    flop = model.cal_flop()
                    ratio = model.weight_stat()
                    robust_log[mod_name][str(sigma)] = {
                        "Accuracy": tmp["Accuracy"], "Loss": tmp["Loss"],
                        "Corr": tmp["Corr"], "FLOP": flop, "E2_ratio": ratio,
                    }
            model.eval_noise_std = 0.0
            model.eval_noise_modality = 2
            print("-" * 30 + "Robustness Summary" + "-" * 30)
            print(json.dumps(robust_log, indent=2))
            robust_log_all.append(robust_log)

    print(f"Finish {args.n_runs} runs")
    print(f"Test Accuracy {np.mean(log[:, 0]) * 100:.2f} ± {np.std(log[:, 0]) * 100:.2f}")
    print(f"Loss {np.mean(log[:, 1]):.4f} ± {np.std(log[:, 1]):.4f}")
    print(f"Corr {np.mean(log[:, 2]):.4f} ± {np.std(log[:, 2]):.4f}")
    print(f"FLOP {np.mean(log[:, 3]):.2f} ± {np.std(log[:, 3]):.2f}")
    print(f"Ratio {np.mean(log[:, 4]):.3f} ± {np.std(log[:, 4]):.3f}")

    results_dir = os.path.join("./log", args.data, "results")
    os.makedirs(results_dir, exist_ok=True)
    mode_tag = {0: "adapt", 1: "E1", 2: "E2", -1: "uniform"}.get(args.infer_mode, f"mode{args.infer_mode}")
    run_name = (f"dyn_conf_enc_{args.enc}_reg_{args.reg}_freeze{args.freeze}{noise_tag}{norm_tag}"
                f"_{mode_tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    results_path = os.path.join(results_dir, run_name + ".json")
    with open(results_path, "w") as f:
        json.dump({
            "args": vars(args),
            "checkpoint": filename,
            "per_run": {
                "accuracy": log[:, 0].tolist(),
                "loss":     log[:, 1].tolist(),
                "corr":     log[:, 2].tolist(),
                "flop":     log[:, 3].tolist(),
                "ratio":    log[:, 4].tolist(),
            },
            "mean": {
                "accuracy": float(np.mean(log[:, 0])),
                "loss":     float(np.mean(log[:, 1])),
                "corr":     float(np.mean(log[:, 2])),
                "flop":     float(np.mean(log[:, 3])),
                "ratio":    float(np.mean(log[:, 4])),
            },
            "robustness": robust_log_all,
        }, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
