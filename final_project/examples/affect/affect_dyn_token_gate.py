"""Per-timestep (token-level) gating conditioned on per-modality reliability.

Architectural extension of DynMMNetV2Noisy that fires the gate once per
timestep rather than once per sequence. At each timestep t the gate sees:
  * concatenated per-modality content features (vision, audio, text encoder
    outputs at timestep t)
  * per-modality confidence (mean L2 norm of each modality's input at t,
    normalized via BatchNorm1d)

The final prediction is the length-masked mean of per-timestep predictions:
  pred_t = w_t[E1] * e1_head(text_feat_t) + w_t[E2] * e2_head(concat_feat_t)
  pred   = sum(pred_t * mask) / sum(mask)

Motivation: MOSEI utterances have moments where text carries everything (a
clear sentiment word) and moments where prosody/face matters more (sarcasm,
hesitation). A sequence-level gate cannot exploit local variation; a
token-level gate conditioned on local reliability can.
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


# Per-modality encoded feature dims (mirror MOSEI E2 architecture):
# vision Transformer(35, 60)  -> 60
# audio  Transformer(74, 120) -> 120
# text   Transformer(300, 120) -> 120
DIM_VISION = 60
DIM_AUDIO = 120
DIM_TEXT = 120
DIM_CONTENT = DIM_VISION + DIM_AUDIO + DIM_TEXT  # 300
DIM_CONFIDENCE = 3


def encode_per_timestep(encoder, x):
    """Run a Transformer encoder and return per-timestep features [batch, seq, embed].

    The base Transformer.forward does `transformer(x)[-1]` to pool to the last
    timestep. We replicate the pre-pool steps and skip the selection so we get
    every timestep's hidden state.
    """
    if isinstance(x, list):
        x = x[0]
    h = encoder.conv(x.permute([0, 2, 1]))   # [batch, embed, seq]
    h = h.permute([2, 0, 1])                  # [seq, batch, embed]
    h = encoder.transformer(h)                # [seq, batch, embed]
    return h.permute([1, 0, 2])               # [batch, seq, embed]


class DynMMNetTokenGate(DynMMNetV2Noisy):
    """DynMMNetV2Noisy with a per-timestep, confidence-aware gate."""

    def __init__(self, temp, hard_gate, freeze, model_name_list,
                 text_noise_prob=0.0, text_noise_std=0.0, text_noise_mode="gaussian"):
        super().__init__(temp, hard_gate, freeze, model_name_list,
                         text_noise_prob, text_noise_std, text_noise_mode)
        # The sequence-level gate from the parent is unused here.
        self.gate = None
        # New per-timestep heads and gate (trained from scratch).
        self.token_e1_head = nn.Linear(DIM_TEXT, 1)
        self.token_e2_head = nn.Linear(DIM_CONTENT, 1)
        self.token_gate = nn.Linear(DIM_CONTENT + DIM_CONFIDENCE, self.branch_num)
        # Per-modality confidence normalization across batch+time.
        self.confidence_norm = nn.BatchNorm1d(DIM_CONFIDENCE)

    def compute_token_confidence(self, inputs):
        """Per-modality L2 norm at each timestep -> [batch, seq, 3]."""
        feats = []
        for x in inputs[0]:
            norm = x.detach().norm(dim=-1)  # [batch, seq]
            feats.append(norm)
        return torch.stack(feats, dim=-1)

    def forward(self, inputs):
        inputs = self._corrupt_text(inputs)

        # Per-timestep features from each encoder.
        # E1 path uses the text-only branch's encoder; E2 path uses branch2's three encoders.
        text_e1 = encode_per_timestep(self.text_encoder, [inputs[0][2], inputs[1][2]])
        vis_e2 = encode_per_timestep(self.branch2.encoders[0], inputs[0][0])
        aud_e2 = encode_per_timestep(self.branch2.encoders[1], inputs[0][1])
        txt_e2 = encode_per_timestep(self.branch2.encoders[2], inputs[0][2])

        content_t = torch.cat([vis_e2, aud_e2, txt_e2], dim=-1)  # [B, T, 300]

        # Per-timestep confidence, normalized across batch+time per modality.
        conf = self.compute_token_confidence(inputs)             # [B, T, 3]
        B, T, _ = conf.shape
        conf_normed = self.confidence_norm(conf.reshape(B * T, DIM_CONFIDENCE)).reshape(B, T, DIM_CONFIDENCE)

        # Per-timestep gate.
        gate_in = torch.cat([content_t, conf_normed], dim=-1)    # [B, T, 303]
        gate_logits = self.token_gate(gate_in)                   # [B, T, 2]
        weights = DiffSoftmax(gate_logits, tau=self.temp, hard=self.hard_gate)  # [B, T, 2]

        # Per-timestep predictions.
        e1_pred_t = self.token_e1_head(text_e1)                  # [B, T, 1]
        e2_pred_t = self.token_e2_head(content_t)                # [B, T, 1]

        if self.infer_mode == 1:
            pred_t = e1_pred_t
        elif self.infer_mode == 2:
            pred_t = e2_pred_t
        elif self.infer_mode == -1:
            uniform = torch.ones_like(weights) / self.branch_num
            pred_t = uniform[..., 0:1] * e1_pred_t + uniform[..., 1:2] * e2_pred_t
        else:
            pred_t = weights[..., 0:1] * e1_pred_t + weights[..., 1:2] * e2_pred_t

        # Length-masked mean over timesteps.
        lengths = inputs[1][0].to(pred_t.device).long()          # [B]
        max_len = pred_t.shape[1]
        mask = (torch.arange(max_len, device=pred_t.device).unsqueeze(0) < lengths.unsqueeze(1))
        mask = mask.float().unsqueeze(-1)                        # [B, T, 1]
        pred = (pred_t * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B, 1]

        # For weight tracking and FLOP regularization, average per-token E2 weight
        # across valid timesteps.
        avg_weight = (weights * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B, 2]
        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, avg_weight.cpu()))

        return pred, avg_weight[:, 1].mean()


def parse_args():
    p = argparse.ArgumentParser("Token-gate DynMM on MOSEI")
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
    filename = os.path.join("./log", args.data,
                            "dyn_token_enc_" + args.enc + "_reg_" + str(args.reg) +
                            "freeze" + str(args.freeze) + noise_tag + ".pt")

    log = np.zeros((args.n_runs, 5))
    robust_log_all = []

    for n in range(args.n_runs):
        model = DynMMNetTokenGate(
            args.temp, args.hard_gate, args.freeze, model_name_list,
            text_noise_prob=args.text_noise_prob,
            text_noise_std=args.text_noise_std,
            text_noise_mode=args.text_noise_mode,
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
    run_name = (f"dyn_token_enc_{args.enc}_reg_{args.reg}_freeze{args.freeze}{noise_tag}"
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
