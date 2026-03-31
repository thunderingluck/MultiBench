"""Training script: Per-Layer Gated Multimodal Fusion on CMU-MOSEI.

This is an end-to-end alternative to DynMM that learns the optimal fusion
depth for each modality independently via per-layer sigmoid gates.

No pre-training of unimodal branches is required.

Usage (from MultiBench root):
    python examples/affect/affect_plg.py \\
        --data_path ./data/mosei_senti_data.pkl \\
        --d_model 64 --n_layers 4 \\
        --reg 0.01 --epochs 40 --lr 1e-3

See fusions/per_layer_gated.py for the model definition.
"""

import os
import sys
import argparse

import torch

sys.path.append(os.getcwd())

from datasets.affect.get_data import get_dataloader
from fusions.per_layer_gated import PLGModel
from training_structures.Supervised_Learning import train, test


def build_parser():
    p = argparse.ArgumentParser(
        description='Per-Layer Gated Fusion on CMU-MOSEI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument('--data_path',  type=str,   default='./data/mosei_senti_data.pkl',
                   help='Path to mosei_senti_data.pkl')
    p.add_argument('--data_type',  type=str,   default='mosei',
                   help='Dataset identifier passed to get_dataloader')
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--num_workers',type=int,   default=4)
    p.add_argument('--max_seq_len',type=int,   default=50,
                   help='Sequence length for max-padding')
    # Model
    p.add_argument('--d_model',    type=int,   default=64,
                   help='Hidden dimension shared by all modalities')
    p.add_argument('--n_heads',    type=int,   default=4,
                   help='Transformer attention heads (must divide d_model)')
    p.add_argument('--n_layers',   type=int,   default=4,
                   help='Transformer depth = number of gate levels per modality')
    p.add_argument('--dropout',    type=float, default=0.1)
    # Training
    p.add_argument('--reg',        type=float, default=0.01,
                   help='Weight for gate sparsity regularisation loss')
    p.add_argument('--epochs',     type=int,   default=40)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--wd',         type=float, default=1e-4,
                   help='AdamW weight decay')
    p.add_argument('--early_stop', action='store_true', default=True,
                   help='Stop if validation loss does not improve for 7 epochs')
    # Output
    p.add_argument('--save',       type=str,   default='./log/mosei/plg_best.pt',
                   help='Where to save the best checkpoint')
    p.add_argument('--gpu',        type=int,   default=0)
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # max_pad=True  →  _process_2 collate  →  (vision, audio, text, labels)
    # is_packed=False in train() so model receives [vision, audio, text]
    # ------------------------------------------------------------------
    print(f'Loading data from {args.data_path} ...')
    traindata, validdata, testdata = get_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        max_pad=True,
        max_seq_len=args.max_seq_len,
        robust_test=False,
        data_type=args.data_type,
        num_workers=args.num_workers,
    )
    print('Data loaded.')

    # ------------------------------------------------------------------
    # Model
    # CMU-MOSEI modality dims: visual=35, audio=74, text=300 (GloVe)
    # ------------------------------------------------------------------
    model = PLGModel(
        in_dims=[35, 74, 300],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'PLGModel | d_model={args.d_model} n_layers={args.n_layers} '
          f'| {n_params:,} trainable parameters')

    # ------------------------------------------------------------------
    # Train
    # moe_model= bypasses MMDL wrapping; model.forward returns (pred, reg)
    # The training loop computes:  loss = L1(pred, target) + lossw * reg
    # ------------------------------------------------------------------
    print(f'\nTraining for up to {args.epochs} epochs '
          f'(reg={args.reg}, lr={args.lr}) ...')
    train(
        encoders=None, fusion=None, head=None,
        train_dataloader=traindata,
        valid_dataloader=validdata,
        total_epochs=args.epochs,
        moe_model=model,
        additional_loss=True,
        lossw=args.reg,
        task='regression',
        optimtype=torch.optim.Adam,
        lr=args.lr,
        weight_decay=args.wd,
        objective=torch.nn.L1Loss(),
        save=args.save,
        early_stop=args.early_stop,
        is_packed=False,
        track_complexity=False,
    )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print(f'Loading best model from {args.save} ...')
    best_model = torch.load(args.save).cuda()
    best_model.reset_weight()

    print('--- Validation set ---')
    test(
        model=best_model,
        test_dataloaders_all=validdata,
        dataset=args.data_type,
        is_packed=False,
        criterion=torch.nn.L1Loss(reduction='sum'),
        task='posneg-classification',
        no_robust=True,
        additional_loss=True,
    )
    best_model.weight_stat()

    best_model.reset_weight()
    print('\n--- Test set ---')
    results = test(
        model=best_model,
        test_dataloaders_all=testdata,
        dataset=args.data_type,
        is_packed=False,
        criterion=torch.nn.L1Loss(reduction='sum'),
        task='posneg-classification',
        no_robust=True,
        additional_loss=True,
    )
    best_model.weight_stat()
    print('\nFinal results:', results)
