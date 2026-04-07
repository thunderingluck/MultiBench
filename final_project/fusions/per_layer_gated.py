"""Per-Layer Sigmoid Gates for Multimodal Fusion.

Extension of DynMM (Xue & Marculescu, CVPR 2023).

Instead of DynMM's single input-level gate that selects between pre-trained
branches, we place an independent sigmoid gate at each transformer layer for
each modality.  This lets the network learn patterns like "use text from layer 1,
add audio at layer 3, skip video entirely" — all on a per-sample basis.

Gate definition:
    g_{m,l} = sigmoid(Linear(mean_pool(h_{m,l})))  ∈ (0, 1)

Applied as a gated residual so gradients always flow:
    h_{m,l+1} = g_{m,l} * TransformerLayer_m_l(h_{m,l})
              + (1 - g_{m,l}) * h_{m,l}

  g ≈ 1  →  modality m fully updates at layer l  (participates in fusion)
  g ≈ 0  →  modality m passes through unchanged  (skipped at this depth)

Key classes:
  PerLayerGatedFusion  —  encoder that returns (fused_rep, gate_tensor)
  PLGModel             —  full model (encoder + head) with the moe_model
                          interface expected by Supervised_Learning.train()
"""

import torch
import torch.nn as nn


class PerLayerGatedFusion(nn.Module):
    """Multimodal encoder with per-layer, per-modality sigmoid gates.

    Args:
        in_dims  : list of input feature dimensions, one per modality.
                   E.g. [35, 74, 300] for CMU-MOSEI (visual, audio, text).
        d_model  : hidden dimension shared across all modalities.
        n_heads  : number of attention heads (must evenly divide d_model).
        n_layers : number of transformer layers (= gate depth per modality).
        dropout  : dropout probability inside each TransformerEncoderLayer.
    """

    def __init__(self, in_dims, d_model=64, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.n_modalities = len(in_dims)
        self.n_layers = n_layers
        self.d_model = d_model

        # Per-modality 1-D convolution to project raw features to d_model
        self.projections = nn.ModuleList([
            nn.Conv1d(dim, d_model, kernel_size=1)
            for dim in in_dims
        ])

        # Per-modality × per-layer transformer encoder layers
        # self.tf_layers[m][l]  processes modality m at depth l
        self.tf_layers = nn.ModuleList([
            nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    batch_first=False,   # expects [T, B, d_model]
                )
                for _ in range(n_layers)
            ])
            for _ in range(self.n_modalities)
        ])

        # Per-modality × per-layer gate networks
        # gate_nets[m][l] : Linear(d_model → 1) followed by sigmoid in forward
        self.gate_nets = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(d_model, 1)
                for _ in range(n_layers)
            ])
            for _ in range(self.n_modalities)
        ])

    def forward(self, inputs):
        """Run gated multimodal encoding.

        Args:
            inputs : list of tensors, each [B, T, F_m], one per modality.

        Returns:
            fused  : [B, n_modalities * d_model]  concatenated final reps.
            gates  : [B, n_modalities, n_layers]   sigmoid gate values,
                     useful for regularisation and visualisation.
        """
        # Project each modality to d_model; reshape to [T, B, d_model]
        hs = []
        for x, proj in zip(inputs, self.projections):
            # x: [B, T, F_m]
            x_proj = proj(x.permute(0, 2, 1))    # [B, d_model, T]
            hs.append(x_proj.permute(2, 0, 1))   # [T, B, d_model]

        # all_gates[l][m] = [B, 1]
        all_gates = []

        for l in range(self.n_layers):
            layer_gates = []
            new_hs = []
            for m in range(self.n_modalities):
                h = hs[m]                                      # [T, B, d_model]

                # Compute gate from the mean-pooled hidden state
                ctx = h.mean(dim=0)                            # [B, d_model]
                g = torch.sigmoid(self.gate_nets[m][l](ctx))  # [B, 1]
                layer_gates.append(g)

                # Apply transformer layer then gated residual
                h_new = self.tf_layers[m][l](h)                # [T, B, d_model]
                g_exp = g.unsqueeze(0)                         # [1, B, 1]
                h_out = g_exp * h_new + (1.0 - g_exp) * h     # [T, B, d_model]
                new_hs.append(h_out)

            hs = new_hs
            all_gates.append(layer_gates)

        # Take last-timestep representation for each modality
        reps = [h[-1] for h in hs]          # list of [B, d_model]
        fused = torch.cat(reps, dim=-1)      # [B, n_modalities * d_model]

        # Assemble gate tensor [B, n_modalities, n_layers]
        # all_gates[l][m] = [B, 1]  →  stack over modalities then layers
        gates = torch.stack(
            [
                torch.cat([all_gates[l][m] for l in range(self.n_layers)], dim=-1)
                for m in range(self.n_modalities)
            ],
            dim=1,
        )  # [B, n_modalities, n_layers]

        return fused, gates


class PLGModel(nn.Module):
    """Full prediction model: PerLayerGatedFusion encoder + linear head.

    Implements the ``moe_model`` interface expected by
    ``training_structures.Supervised_Learning.train()``:

        forward(inputs)  →  (prediction [B, out_dim], reg_loss scalar)

    and the validation-diagnostic hooks:

        reset_weight()   —  called before each validation pass
        weight_stat()    —  called after each validation pass (prints gate means)
        get_mean_gates() —  returns [n_modalities, n_layers] for plotting

    Args:
        in_dims   : list of input feature dims per modality.
        d_model   : hidden dimension.
        n_heads   : transformer attention heads.
        n_layers  : transformer depth / number of gates per modality.
        dropout   : dropout rate.
        out_dim   : output dimension (1 for regression).
    """

    MODALITY_NAMES = ['visual', 'audio', 'text']

    def __init__(self, in_dims, d_model=64, n_heads=4, n_layers=4,
                 dropout=0.1, out_dim=1):
        super().__init__()
        self.encoder = PerLayerGatedFusion(in_dims, d_model, n_heads, n_layers, dropout)
        self.head = nn.Linear(len(in_dims) * d_model, out_dim)
        self.n_layers = n_layers
        self.n_modalities = len(in_dims)

        # Diagnostics: accumulated gate tensors from the current validation pass
        self._gate_accum = []
        self._tracking = False

    # ------------------------------------------------------------------
    # moe_model interface (called by Supervised_Learning.train / test)
    # ------------------------------------------------------------------

    def forward(self, inputs):
        """Run the model.

        Args:
            inputs : list of [B, T, F_m] float tensors (non-packed format).

        Returns:
            pred     : [B, out_dim] prediction.
            reg_loss : scalar — mean gate activation (sparsity regularizer).
                       Minimising this pushes unused gates toward 0.
        """
        fused, gates = self.encoder(inputs)   # gates: [B, M, L]
        pred = self.head(fused)               # [B, out_dim]
        reg_loss = gates.mean()

        if self._tracking:
            self._gate_accum.append(gates.detach().cpu())

        return pred, reg_loss

    def reset_weight(self):
        """Reset accumulated gate statistics (called before validation pass)."""
        self._gate_accum = []
        self._tracking = True

    def weight_stat(self):
        """Print mean gate values per modality per layer (called after validation)."""
        self._tracking = False
        if not self._gate_accum:
            return
        mean_gates = self.get_mean_gates()   # [M, L]
        names = self.MODALITY_NAMES[:self.n_modalities]
        print('  Mean gate values per modality (rows) × layer (cols):')
        for m, name in enumerate(names):
            vals = '  '.join(f'{mean_gates[m, l].item():.3f}'
                             for l in range(self.n_layers))
            print(f'    {name:6s}: [{vals}]')

    def get_mean_gates(self):
        """Return mean gate tensor [n_modalities, n_layers] from last validation pass."""
        if not self._gate_accum:
            return None
        return torch.cat(self._gate_accum, dim=0).mean(dim=0)
