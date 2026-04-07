"""Per-Layer Sigmoid Gates for Multimodal Fusion (v2 — notebook design).

Extension of DynMM (Xue & Marculescu, CVPR 2023), implementing the design
from the per-layer adaptive fusion notebook (affect_dyn_v2.py).

Design differences vs. per_layer_gated.py (v1)
------------------------------------------------
1. Centralized gate network
   v1: one small Linear(d_model→1) per (modality, layer), computed from that
       modality's own hidden state as encoding proceeds.
   v2: a single PerLayerGate module receives ALL concatenated raw inputs and
       outputs the complete [B, M, L] gate matrix upfront, before any encoding.
       This allows cross-modal context to inform gating decisions.

2. Gate mechanism
   v1: gated residual — h_out = g*h_new + (1-g)*h_prev
       Gradient always flows; g=0 still propagates the previous hidden state.
   v2: multiplicative zeroing — h = h * g
       g=0 fully suppresses a layer's output.  Combined with hard gates this
       produces exact sparse routing.

3. Hard gating
   v1: not supported; gates are always soft sigmoids.
   v2: straight-through hard sigmoid (hard_sigmoid): discrete {0,1} in the
       forward pass, soft sigmoid gradient in the backward pass.

4. Resource loss
   v1: gates.mean()  — uniform penalty across all gates.
   v2: sum(gates_mean * flop_per_layer)  — each gate is penalized in
       proportion to the actual FLOPs of the layer it controls.

5. Infer modes
   v1: none — always uses learned gates.
   v2: infer_mode=0 (learned), 1 (text-only forced on), -1 (all-on).

6. Pretrained encoder initialization
   v1: not supported.
   v2: init_from_pretrained() copies conv + transformer weights from an
       existing late-fusion model checkpoint.

Key classes
-----------
  PerLayerGate              — centralized gate network (all modalities → all gates)
  GatedTransformerEncoder   — single-modality encoder with external gates
  PerLayerGatedFusionV2     — combines gate net + per-modality encoders
  PLGModelV2                — full model (fusion + head) with moe_model interface
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hard_sigmoid(x, tau=1.0):
    """Straight-through estimator for a hard sigmoid gate.

    Forward:  returns {0, 1} by thresholding sigmoid(x/tau) at 0.5.
    Backward: gradient flows as if the output were soft sigmoid(x/tau).
    """
    y_soft = torch.sigmoid(x / tau)
    y_hard = (y_soft > 0.5).float()
    return y_hard - y_soft.detach() + y_soft


# ---------------------------------------------------------------------------
# Centralized gate network
# ---------------------------------------------------------------------------

class PerLayerGate(nn.Module):
    """Centralized gate network: all modalities in → all gate values out.

    Receives the time-series of all concatenated raw modality features
    [B, T, sum(in_dims)] and outputs a gate tensor [B, M, L] in (0, 1).

    By seeing all modalities at once the gate can model cross-modal context,
    e.g. "text confidence is already high — suppress visual at all layers".
    This is the key structural change relative to v1, where each gate was
    computed independently from that modality's own hidden state.

    Architecture:
        Conv1d(input_dim → hidden_dim)   — cheap channel projection
        TransformerEncoderLayer           — sequence summarisation
        mean-pool over T
        Linear(hidden_dim → M * L)        — output all gate logits at once

    Args:
        input_dim      : sum of all modality raw feature dimensions.
        num_modalities : M — number of modalities.
        num_layers     : L — number of transformer layers / gates per modality.
        hidden_dim     : internal projection size (default 10).
        temp           : sigmoid temperature; lower → sharper / more binary gates.
    """

    def __init__(self, input_dim, num_modalities, num_layers,
                 hidden_dim=10, temp=1.0):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_layers = num_layers
        self.temp = temp

        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False)
        self.tf_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=1,
            dim_feedforward=4 * hidden_dim,
            batch_first=False,   # expects [T, B, hidden_dim]
        )
        self.fc = nn.Linear(hidden_dim, num_modalities * num_layers)

    def forward(self, x, hard=False):
        """Compute gate tensor from concatenated raw inputs.

        Args:
            x    : [B, T, input_dim]  concatenated raw modality features.
            hard : use straight-through hard thresholding if True.

        Returns:
            gates : [B, num_modalities, num_layers] in (0, 1).
        """
        # Project and reshape: [B, T, D] → [T, B, hidden_dim]
        h = self.proj(x.permute(0, 2, 1))   # [B, hidden_dim, T]
        h = h.permute(2, 0, 1)              # [T, B, hidden_dim]
        h = self.tf_layer(h)                 # [T, B, hidden_dim]
        h = h.mean(dim=0)                    # [B, hidden_dim]  mean pooling

        logits = self.fc(h).view(-1, self.num_modalities, self.num_layers)
        if hard:
            return hard_sigmoid(logits, tau=self.temp)
        return torch.sigmoid(logits / self.temp)


# ---------------------------------------------------------------------------
# Per-modality encoder with external gates
# ---------------------------------------------------------------------------

class GatedTransformerEncoder(nn.Module):
    """Single-modality transformer encoder driven by externally supplied gates.

    Gate application (v2 design):
        h_l = TransformerLayer_l(h_{l-1})
        h_l = h_l * g_l              (multiplicative zeroing)

    gate=1: layer output fully passes through (modality active at this depth).
    gate=0: layer output is zero (modality skipped at this depth).

    This differs from v1's gated residual (g*h_new + (1-g)*h_prev), which
    blends the previous state back in and never fully suppresses a layer.

    Gates are passed in from PerLayerGateFusionV2 — they are NOT computed
    inside this module.  This separation means the gate network can see all
    modalities before any encoding begins.

    Args:
        in_features : raw feature dimension for this modality.
        d_model     : hidden dimension.
        num_layers  : transformer depth.
        nhead       : attention heads (must divide d_model).
    """

    def __init__(self, in_features, d_model, num_layers=5, nhead=5):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.conv = nn.Conv1d(in_features, d_model, kernel_size=1, bias=False)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       batch_first=False)
            for _ in range(num_layers)
        ])

    def forward(self, x, gates=None):
        """Encode one modality.

        Args:
            x     : [B, T, in_features] or list whose first element is that tensor.
            gates : [B, num_layers] gate values.  None → all layers fully active.

        Returns:
            [B, d_model]  last-timestep representation.
        """
        if isinstance(x, (list, tuple)):
            x = x[0]

        # [B, T, F] → [T, B, d_model]
        h = self.conv(x.permute(0, 2, 1))   # [B, d_model, T]
        h = h.permute(2, 0, 1)              # [T, B, d_model]

        for l, layer in enumerate(self.layers):
            h = layer(h)                     # [T, B, d_model]
            if gates is not None:
                # [B] → [1, B, 1]  broadcast over T and d_model
                g = gates[:, l].unsqueeze(0).unsqueeze(-1)
                h = h * g

        return h[-1]   # [B, d_model]


# ---------------------------------------------------------------------------
# Combined fusion module
# ---------------------------------------------------------------------------

class PerLayerGatedFusionV2(nn.Module):
    """Multimodal encoder with a centralized per-layer gate network.

    Flow:
        1. Concatenate all raw modality inputs → x_cat [B, T, sum(F_m)]
        2. PerLayerGate(x_cat) → gates [B, M, L]
        3. For each modality m: GatedTransformerEncoder(x_m, gates[:,m,:]) → rep_m
        4. Concatenate reps → fused [B, sum(d_models)]

    Args:
        in_dims    : list of raw feature dimensions, one per modality.
                     E.g. [35, 74, 300] for (visual, audio, text) on CMU-MOSEI.
        d_models   : int or list of ints — encoder hidden dim(s).
        num_layers : transformer depth (same for all modalities).
        nhead      : attention heads per encoder layer.
        hidden_dim : gate network internal projection size.
        temp       : gate temperature.
    """

    def __init__(self, in_dims, d_models=120, num_layers=5, nhead=5,
                 hidden_dim=10, temp=1.0):
        super().__init__()
        self.num_modalities = len(in_dims)
        self.num_layers = num_layers

        if isinstance(d_models, int):
            d_models = [d_models] * self.num_modalities
        self.d_models = d_models

        self.encoders = nn.ModuleList([
            GatedTransformerEncoder(in_dims[m], d_models[m],
                                    num_layers=num_layers, nhead=nhead)
            for m in range(self.num_modalities)
        ])

        self.gate_net = PerLayerGate(
            input_dim=sum(in_dims),
            num_modalities=self.num_modalities,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            temp=temp,
        )

        # FLOPs cost per (modality, layer) for resource-weighted loss
        flop_costs = torch.zeros(self.num_modalities, num_layers)
        for m, dim in enumerate(d_models):
            flop_costs[m, :] = dim * dim * 4 / 1e6
        self.register_buffer('flop_per_layer', flop_costs)

    def forward(self, modality_inputs, hard=False, infer_mode=0):
        """Encode all modalities with learned per-layer gates.

        Args:
            modality_inputs : list of [B, T, F_m] tensors.
            hard            : use hard (straight-through) gates.
            infer_mode      : 0=learned, 1=text-only (last modality), -1=all-on.

        Returns:
            fused : [B, sum(d_models)]
            gates : [B, M, L]  gate values used (for logging / resource loss).
        """
        B, device = modality_inputs[0].shape[0], modality_inputs[0].device

        if infer_mode == 1:
            gates = torch.zeros(B, self.num_modalities, self.num_layers,
                                device=device)
            gates[:, -1, :] = 1.0          # text-only (last modality)
        elif infer_mode == -1:
            gates = torch.ones(B, self.num_modalities, self.num_layers,
                               device=device)
        else:
            x_cat = torch.cat(modality_inputs, dim=-1)   # [B, T, sum(F_m)]
            gates = self.gate_net(x_cat, hard=hard)       # [B, M, L]

        reps = []
        for m in range(self.num_modalities):
            rep = self.encoders[m](modality_inputs[m], gates=gates[:, m, :])
            reps.append(rep)

        return torch.cat(reps, dim=-1), gates

    def init_from_pretrained(self, pretrained_encoders):
        """Copy conv + transformer weights from a list of pretrained encoders.

        Args:
            pretrained_encoders : list of GatedTransformerEncoder (or compatible).
                                  None entries are skipped.
        """
        for m, src in enumerate(pretrained_encoders):
            if src is None:
                continue
            if hasattr(src, 'conv'):
                self.encoders[m].conv.load_state_dict(src.conv.state_dict())
            if hasattr(src, 'layers'):
                n = min(len(src.layers), self.num_layers)
                for l in range(n):
                    self.encoders[m].layers[l].load_state_dict(
                        src.layers[l].state_dict())
                print(f'  Encoder {m}: copied {n} layer(s) from pretrained')


# ---------------------------------------------------------------------------
# Full prediction model
# ---------------------------------------------------------------------------

class PLGModelV2(nn.Module):
    """Full prediction model with centralized per-layer gates.

    Implements the ``moe_model`` interface used by
    ``training_structures.Supervised_Learning.train()``:

        forward(inputs)  →  (prediction [B, out_dim], resource_loss scalar)
        reset_weight()   —  called before each validation pass
        weight_stat()    —  called after each validation pass

    Resource loss:
        lambda * sum( mean_gates_over_batch * flop_per_layer )

    Infer modes (set model.infer_mode before evaluation):
        0  : learned gates  (default, used during training)
        1  : text-only — last modality fully on, all others off
       -1  : all modalities on at every layer

    Args:
        in_dims    : list of raw feature dims per modality.
        d_models   : int or list — encoder hidden dim(s).
        num_layers : transformer depth.
        nhead      : attention heads.
        hidden_dim : gate network internal size.
        temp       : gate temperature.
        out_dim    : prediction output size (1 for regression).
        hard_gate  : use straight-through hard gates during training.
    """

    MODALITY_NAMES = ['visual', 'audio', 'text']

    def __init__(self, in_dims, d_models=120, num_layers=5, nhead=5,
                 hidden_dim=10, temp=1.0, out_dim=1, hard_gate=False):
        super().__init__()
        self.hard_gate = hard_gate
        self.infer_mode = 0
        self.n_modalities = len(in_dims)
        self.n_layers = num_layers

        self.fusion = PerLayerGatedFusionV2(
            in_dims=in_dims, d_models=d_models, num_layers=num_layers,
            nhead=nhead, hidden_dim=hidden_dim, temp=temp,
        )

        if isinstance(d_models, int):
            d_models = [d_models] * len(in_dims)
        self.head = nn.Linear(sum(d_models), out_dim)

        self._gate_accum = []
        self._tracking = False

    # ------------------------------------------------------------------
    # moe_model interface
    # ------------------------------------------------------------------

    def forward(self, inputs):
        """Run the model.

        Args:
            inputs : either
                     - list of [B, T, F_m] tensors (non-packed), or
                     - [modality_list, padding_list] (is_packed=True format,
                       padding is accepted but not used by the gate network).

        Returns:
            pred          : [B, out_dim]
            resource_loss : scalar, FLOPs-weighted mean gate activation.
        """
        if (isinstance(inputs, (list, tuple)) and len(inputs) == 2
                and isinstance(inputs[0], (list, tuple))):
            modality_inputs = [t.float() for t in inputs[0]]
        else:
            modality_inputs = [t.float() for t in inputs]

        fused, gates = self.fusion(
            modality_inputs,
            hard=self.hard_gate,
            infer_mode=self.infer_mode,
        )
        pred = self.head(fused)

        # FLOPs-weighted resource penalty
        gates_mean = gates.mean(dim=0)   # [M, L]
        resource_loss = (gates_mean * self.fusion.flop_per_layer).sum()

        if self._tracking:
            self._gate_accum.append(gates.detach().cpu())

        return pred, resource_loss

    def reset_weight(self):
        """Reset accumulated gate statistics (call before each eval pass)."""
        self._gate_accum = []
        self._tracking = True

    def weight_stat(self):
        """Print mean gate values per modality per layer; return non-text mean."""
        self._tracking = False
        if not self._gate_accum:
            return 0.0
        mean_gates = self.get_mean_gates()   # [M, L]
        names = self.MODALITY_NAMES[:self.n_modalities]
        print('\n--- Gate Statistics (mean activation per layer) ---')
        for m, name in enumerate(names):
            vals = ' '.join(
                f'L{l}:{mean_gates[m, l].item():.3f}'
                for l in range(self.n_layers)
            )
            print(f'  {name}: {vals}')
        non_text_mean = mean_gates[:-1].mean().item()
        print(f'  non-text mean gate: {non_text_mean:.4f}')
        return non_text_mean

    def get_mean_gates(self):
        """Return [n_modalities, n_layers] mean gates from the last eval pass."""
        if not self._gate_accum:
            return None
        return torch.cat(self._gate_accum, dim=0).mean(dim=0)

    def cal_flop(self):
        """Estimate effective MAdds from the last eval pass."""
        if not self._gate_accum:
            return 0.0
        mean_gates = self.get_mean_gates()
        effective = (mean_gates * self.fusion.flop_per_layer.cpu()).sum().item()
        base_flop = 135.0
        total = base_flop + effective
        print(f'Effective FLOPs: {total:.2f}M (base={base_flop:.0f}, '
              f'gated={effective:.2f})')
        return total

    def init_from_pretrained(self, pretrained_encoders, pretrained_head=None):
        """Warm-start encoder and/or head from a pretrained late-fusion model.

        Args:
            pretrained_encoders : list of encoder modules, one per modality.
            pretrained_head     : optional linear/MLP head module.
        """
        self.fusion.init_from_pretrained(pretrained_encoders)
        if pretrained_head is not None:
            try:
                self.head.load_state_dict(pretrained_head.state_dict())
                print('  Head: loaded from pretrained')
            except Exception as e:
                print(f'  Head: could not load pretrained weights ({e})')
