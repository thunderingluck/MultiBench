import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP, Transformer, Sequential, Identity
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test, MMDL


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class DynMMNet(nn.Module):
    def __init__(self, temp, hard_gate, freeze=True, model_name_list=None):
        super(DynMMNet, self).__init__()
        self.branch_num = 3
        self.encoders, self.heads = self.load_model(model_name_list)
        if freeze:
            self.freeze_model()

        # gating network
        # self.gate = GRU(409, 3, dropout=True, has_padding=True)
        self.gate = nn.Sequential(Transformer(409, 10), nn.Linear(10, self.branch_num))
        self.temp = temp
        self.hard_gate = hard_gate
        self.weight_list = torch.Tensor()
        self.store_weight = False
        self.infer_mode = 0

    def load_model(self, model_name_list):
        encoder_list, head_list = [], []
        if model_name_list is not None:
            for model_name in model_name_list:
                encoder_list.append(torch.load(model_name))
                head_list.append(torch.load(model_name.replace('encoder', 'head')))
                print(f'Loading model {model_name}')
            return nn.ModuleList(encoder_list), nn.ModuleList(head_list)

    def freeze_model(self):
        for param in self.encoders.parameters():
            param.requires_grad = False
        for param in self.heads.parameters():
            param.requires_grad = False

    def reset_weight(self):
        self.weight_list = torch.Tensor()
        self.store_weight = True

    def weight_stat(self):
        print(self.weight_list)
        tmp = torch.mean(self.weight_list, dim=0)
        print(f'mean branch weight {tmp[0].item():.4f}, {tmp[1].item():.4f}, {tmp[2].item():.4f}')
        self.store_weight = False

    def cal_flop(self):
        tmp = torch.mean(self.weight_list, dim=0)
        total_flop = (self.flop * tmp).sum()
        print(f'Total Flops {total_flop.item():.2f}M')
        return total_flop.item()

    def forward2(self, inputs):
        x = torch.cat(inputs[0], dim=2)
        weight = DiffSoftmax(self.gate([x, inputs[1][0]]), tau=self.temp, hard=self.hard_gate)
        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))

        pred_list = []
        for i in range(len(inputs[0])):
            mid = self.encoders[i]([inputs[0][i], inputs[1][i]])
            pred_list.append(self.heads[i](mid))

        if self.infer_mode > 0:
            return pred_list[self.infer_mode - 1]
        if self.infer_mode == -1:
            weight = torch.ones_like(weight) / self.branch_num

        output = weight[:, 0:1] * pred_list[0] + weight[:, 1:2] * pred_list[1] + weight[:, 2:3] * pred_list[2]
        return output, weight[:, 2].mean()

    def forward(self, inputs, path, weight_enable):
        if weight_enable:
            x = torch.cat(inputs[0], dim=2)
            weight = DiffSoftmax(self.gate([x, inputs[1][0]]), tau=self.temp, hard=self.hard_gate)
        mid = self.encoders[path]([inputs[0][path], inputs[1][path]])
        output = self.heads[path](mid)
        return output


class DynMMNetV2(nn.Module):
    def __init__(self, temp, hard_gate, freeze, model_name_list):
        super(DynMMNetV2, self).__init__()
        self.branch_num = 2
        self.text_encoder = torch.load(model_name_list[0])
        self.text_head = torch.load(model_name_list[0].replace('encoder', 'head'))
        self.branch2 = torch.load(model_name_list[1])

        if freeze:
            self.freeze_branch(self.text_encoder)
            self.freeze_branch(self.text_head)
            self.freeze_branch(self.branch2)

        self.gate = nn.Sequential(Transformer(409, 10), nn.Linear(10, self.branch_num))
        self.temp = temp
        self.hard_gate = hard_gate
        self.weight_list = torch.Tensor()
        self.store_weight = False
        self.infer_mode = 0
        self.flop = torch.Tensor([135.13226, 320.03205])
        # self.flop = torch.Tensor([156.02, 340.92])

    def freeze_branch(self, m):
        for param in m.parameters():
            param.requires_grad = False

    def reset_weight(self):
        self.weight_list = torch.Tensor()
        self.store_weight = True

    def weight_stat(self):
        print(self.weight_list)
        tmp = torch.mean(self.weight_list, dim=0)
        print(f'mean branch weight {tmp[0].item():.4f}, {tmp[1].item():.4f}')
        self.store_weight = False
        # print('path 0', torch.where(self.weight_list[:, 0] == 1))
        # print('path 1', torch.where(self.weight_list[:, 1] == 1))
        return tmp[1].item()

    def cal_flop(self):
        tmp = torch.mean(self.weight_list, dim=0)
        total_flop = (self.flop * tmp).sum()
        print(f'Total Flops {total_flop.item():.2f}M')
        return total_flop.item()

    def forward(self, inputs):
        x = torch.cat(inputs[0], dim=2)
        weight = DiffSoftmax(self.gate([x, inputs[1][0]]), tau=self.temp, hard=self.hard_gate)
        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))

        pred_list = [self.text_head(self.text_encoder([inputs[0][2], inputs[1][2]])), self.branch2(inputs)]
        if self.infer_mode > 0:
            return pred_list[self.infer_mode - 1], 0
        if self.infer_mode == -1:
            weight = torch.ones_like(weight) / self.branch_num

        output = weight[:, 0:1] * pred_list[0] + weight[:, 1:2] * pred_list[1]
        return output, weight[:, 1].mean()

    def forward_separate_branch(self, inputs, path, weight_enable):  # see separate branch performance
        if weight_enable:
            x = torch.cat(inputs[0], dim=2)
            weight = DiffSoftmax(self.gate([x, inputs[1][0]]), tau=self.temp, hard=self.hard_gate)
        if path == 1:
            output = self.text_head(self.text_encoder([inputs[0][2], inputs[1][2]]))
        else:
            output = self.branch2(inputs)
        return output


class DynMMNetV2Noisy(DynMMNetV2):
    """DynMMNetV2 with per-sample text-modality noise augmentation during training.

    Motivation: if E1 (text-only) dominates E2 (multimodal) on clean data, the gate
    collapses to always-E1. Corrupting text on a fraction of training samples creates
    a "hard" slice where E2 must win, giving the gate something non-trivial to learn
    and training robustness to text-channel noise directly.
    """

    def __init__(self, temp, hard_gate, freeze, model_name_list,
                 text_noise_prob=0.0, text_noise_std=0.0, text_noise_mode='gaussian'):
        super().__init__(temp, hard_gate, freeze, model_name_list)
        self.text_noise_prob = text_noise_prob
        self.text_noise_std = text_noise_std
        self.text_noise_mode = text_noise_mode
        # eval-time override: when >0, applies corruption at inference regardless of training mode
        self.eval_noise_std = 0.0
        self.eval_noise_prob = 1.0
        self.eval_noise_modality = 2  # 0=vision, 1=audio, 2=text (MOSEI convention)

    # modality indices: 0=vision, 1=audio, 2=text
    def _corrupt_modality(self, inputs, mod_idx, prob, std_override, mode):
        x = inputs[0][mod_idx]
        B = x.shape[0]
        mask = (torch.rand(B, device=x.device) < prob)
        if not mask.any():
            return inputs
        view = (-1,) + (1,) * (x.dim() - 1)
        m = mask.view(*view).to(x.dtype)
        if mode == 'zero':
            new_x = x * (1.0 - m)
        elif mode == 'shuffle':
            perm = torch.randperm(B, device=x.device)
            new_x = x * (1.0 - m) + x[perm] * m
        else:  # gaussian
            std = std_override if std_override > 0 else x.detach().std()
            noise = torch.randn_like(x) * std
            new_x = x + noise * m
        new_mods = list(inputs[0])
        new_mods[mod_idx] = new_x
        return (new_mods, inputs[1])

    def _corrupt_text(self, inputs):
        # training-time augmentation: always hits text (training objective is text-noise robustness)
        if self.training and self.text_noise_prob > 0.0:
            return self._corrupt_modality(inputs, 2, self.text_noise_prob,
                                          getattr(self, 'text_noise_std', 0.0),
                                          self.text_noise_mode)
        # eval-time injection: configurable modality
        if (not self.training) and getattr(self, 'eval_noise_std', 0.0) > 0.0:
            mod_idx = getattr(self, 'eval_noise_modality', 2)
            return self._corrupt_modality(inputs, mod_idx, self.eval_noise_prob,
                                          self.eval_noise_std, self.text_noise_mode)
        return inputs

    def forward(self, inputs):
        inputs = self._corrupt_text(inputs)
        return super().forward(inputs)

    def forward_separate_branch(self, inputs, path, weight_enable):
        inputs = self._corrupt_text(inputs)
        return super().forward_separate_branch(inputs, path, weight_enable)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("unimodal network on mosi",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--data", type=str, default='mosei', help="dataset: mosi / mosei")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--enc", type=str, default='transformer', help="gru / transformer")
    argparser.add_argument("--n-epochs", type=int, default=50, help="number of epochs")
    argparser.add_argument("--temp", type=float, default=1, help="temperature")
    argparser.add_argument("--hard-gate", action='store_true', help='hard gates')
    argparser.add_argument("--reg", type=float, default=0.0, help="reg loss weight")
    argparser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
    argparser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    argparser.add_argument("--infer-mode", type=int, default=0, help="inference mode")
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--freeze", action='store_true', help='freeze other parts of the model')
    argparser.add_argument("--text-noise-prob", type=float, default=0.0,
                           help="per-sample prob of corrupting text modality during training")
    argparser.add_argument("--text-noise-std", type=float, default=0.0,
                           help="stdev for gaussian text noise; if 0, uses batch stdev")
    argparser.add_argument("--text-noise-mode", type=str, default='gaussian',
                           choices=['gaussian', 'zero', 'shuffle'],
                           help="how to corrupt text: additive gaussian, zero-out, or shuffle across batch")
    argparser.add_argument("--eval-noise-sigmas", type=str, default='',
                           help="comma-separated sigmas for eval-time corruption, e.g. '0.3,0.5,1.0'")
    argparser.add_argument("--eval-noise-modalities", type=str, default='text',
                           help="comma-separated list of modalities to corrupt at eval: 'vision,audio,text'")
    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load data
    datafile = {'mosi': 'mosi_raw', 'mosei': 'mosei_senti_data'}
    traindata, validdata, testdata = get_dataloader('./data/' + datafile[args.data] + '.pkl', robust_test=False,
                                                    data_type=args.data, num_workers=16)

    log = np.zeros((args.n_runs, 5))
    robust_log_all = []
    for n in range(args.n_runs):
        # Init Model
        # modality = ['visual', 'audio', 'text']
        # model_name_list = []
        # for m in modality:
        #     model_name_list.append(os.path.join('./log', args.data, 'reg_' + args.enc + '_encoder_' + m + '.pt'))
        # model = DynMMNet(3, args.temp, args.hard_gate, args.freeze, model_name_list).cuda()
        model_name_list = ['./log/mosei/b1_reg_transformer_encoder_text.pt', './log/mosei/b2_lf_tran.pt']
        if args.text_noise_prob > 0.0:
            model = DynMMNetV2Noisy(args.temp, args.hard_gate, args.freeze, model_name_list,
                                    text_noise_prob=args.text_noise_prob,
                                    text_noise_std=args.text_noise_std,
                                    text_noise_mode=args.text_noise_mode).cuda()
        else:
            model = DynMMNetV2(args.temp, args.hard_gate, args.freeze, model_name_list).cuda()
        noise_tag = f'_np{args.text_noise_prob}_{args.text_noise_mode}' if args.text_noise_prob > 0 else ''
        filename = os.path.join('./log', args.data, 'dyn_enc_' + args.enc + '_reg_' + str(args.reg) +
                                'freeze' + str(args.freeze) + noise_tag + '.pt')

        # Train
        if not args.eval_only:
            train(None, None, None, traindata, validdata, args.n_epochs, task="regression", optimtype=torch.optim.AdamW,
                  is_packed=True, early_stop=True, lr=args.lr, save=filename, weight_decay=args.wd,
                  objective=torch.nn.L1Loss(), moe_model=model, additional_loss=True, lossw=args.reg)

        # Test
        print(f"Testing model {filename}:")
        model = torch.load(filename).cuda()
        model.infer_mode = args.infer_mode
        if isinstance(model, DynMMNetV2Noisy):
            if not hasattr(model, 'eval_noise_std'):
                model.eval_noise_std = 0.0
            if not hasattr(model, 'eval_noise_prob'):
                model.eval_noise_prob = 1.0
            if not hasattr(model, 'text_noise_mode'):
                model.text_noise_mode = args.text_noise_mode
        print('-' * 30 + 'Val data' + '-' * 30)
        tmp = test(model=model, test_dataloaders_all=validdata, dataset=args.data, is_packed=True,
                   criterion=torch.nn.L1Loss(reduction='sum'), task='posneg-classification', no_robust=True, additional_loss=True)

        model.reset_weight()
        print('-' * 30 + 'Test data' + '-' * 30)
        tmp = test(model=model, test_dataloaders_all=testdata, dataset=args.data, is_packed=True,
                   criterion=torch.nn.L1Loss(reduction='sum'), task='posneg-classification', no_robust=True, additional_loss=True)
        log[n] = tmp['Accuracy'], tmp['Loss'], tmp['Corr'], model.cal_flop(), model.weight_stat()

        # Robustness sweep: inject per-modality noise at eval time
        if args.eval_noise_sigmas and isinstance(model, DynMMNetV2Noisy):
            sigmas = [float(s) for s in args.eval_noise_sigmas.split(',') if s.strip()]
            mod_name_to_idx = {'vision': 0, 'audio': 1, 'text': 2}
            mods = [m.strip() for m in args.eval_noise_modalities.split(',') if m.strip()]
            robust_log = {}
            for mod_name in mods:
                if mod_name not in mod_name_to_idx:
                    print(f'Skipping unknown modality {mod_name}')
                    continue
                mod_idx = mod_name_to_idx[mod_name]
                robust_log[mod_name] = {}
                for sigma in sigmas:
                    model.eval_noise_std = sigma
                    model.eval_noise_prob = 1.0
                    model.eval_noise_modality = mod_idx
                    model.reset_weight()
                    print('-' * 20 + f' Test ({mod_name} sigma={sigma}) ' + '-' * 20)
                    tmp = test(model=model, test_dataloaders_all=testdata, dataset=args.data, is_packed=True,
                               criterion=torch.nn.L1Loss(reduction='sum'), task='posneg-classification',
                               no_robust=True, additional_loss=True)
                    flop = model.cal_flop()
                    ratio = model.weight_stat()
                    robust_log[mod_name][sigma] = {'Accuracy': tmp['Accuracy'], 'Loss': tmp['Loss'],
                                                    'Corr': tmp['Corr'], 'FLOP': flop, 'E2_ratio': ratio}
            model.eval_noise_std = 0.0  # restore
            model.eval_noise_modality = 2
            print('-' * 30 + 'Robustness Summary' + '-' * 30)
            print(json.dumps(robust_log, indent=2))
            robust_log_all.append(robust_log)

    print(log[:, 0])
    print(log[:, 1])
    print(log[:, 2])
    print(log[:, 3])
    print(log[:, 4])
    print('-' * 60)
    print(f'Finish {args.n_runs} runs')
    print(f'Test Accuracy {np.mean(log[:, 0]) * 100:.2f} ± {np.std(log[:, 0]) * 100:.2f}')
    print(f'Loss {np.mean(log[:, 1]):.4f} ± {np.std(log[:, 1]):.4f}')
    print(f'Corr {np.mean(log[:, 2]):.4f} ± {np.std(log[:, 2]):.4f}')
    print(f'FLOP {np.mean(log[:, 3]):.2f} ± {np.std(log[:, 3]):.2f}')
    print(f'Ratio {np.mean(log[:, 4]):.3f} ± {np.std(log[:, 4]):.2f}')

    idx = np.argmax(log[:, 1])
    print('Best result', log[idx, :])

    results_dir = os.path.join('./log', args.data, 'results')
    os.makedirs(results_dir, exist_ok=True)
    noise_tag = f'_np{args.text_noise_prob}_{args.text_noise_mode}' if args.text_noise_prob > 0 else ''
    mode_tag = {0: 'adapt', 1: 'E1', 2: 'E2', -1: 'uniform'}.get(args.infer_mode, f'mode{args.infer_mode}')
    run_name = (f'dyn_enc_{args.enc}_reg_{args.reg}_freeze{args.freeze}{noise_tag}'
                f'_{mode_tag}_{time.strftime("%Y%m%d_%H%M%S")}')
    results_path = os.path.join(results_dir, run_name + '.json')
    summary = {
        'args': vars(args),
        'checkpoint': filename,
        'per_run': {
            'accuracy': log[:, 0].tolist(),
            'loss': log[:, 1].tolist(),
            'corr': log[:, 2].tolist(),
            'flop': log[:, 3].tolist(),
            'ratio': log[:, 4].tolist(),
        },
        'mean': {
            'accuracy': float(np.mean(log[:, 0])),
            'loss': float(np.mean(log[:, 1])),
            'corr': float(np.mean(log[:, 2])),
            'flop': float(np.mean(log[:, 3])),
            'ratio': float(np.mean(log[:, 4])),
        },
        'std': {
            'accuracy': float(np.std(log[:, 0])),
            'loss': float(np.std(log[:, 1])),
            'corr': float(np.std(log[:, 2])),
            'flop': float(np.std(log[:, 3])),
            'ratio': float(np.std(log[:, 4])),
        },
        'best_run': log[idx, :].tolist(),
        'robustness': robust_log_all,
    }
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved results to {results_path}')