# Copyright (c) ModelScope Contributors. All rights reserved.
#
# GKD with TIP (Token Importance in on-Policy distillation) Soft-OR scoring.
# Reproduces the Soft-OR algorithm from arxiv 2604.14084v2.
#
# Per-token score (Eq. 5):
#     s_t = h_hat_t + delta_hat_t - h_hat_t * delta_hat_t
#         = 1 - (1 - h_hat_t) * (1 - delta_hat_t)
# where, for each on-policy token t in the batch,
#     h_t      = H(P_S(. | c_t)) / log|V|        (normalized student entropy)
#     delta_t  = D_KL(P_S(. | c_t) || P_T(. | c_t))
#                  (== "reverse KL" by GKD paper convention, β=1 in parent GKDTrainer;
#                   the TIP/Soft-OR paper itself calls it "forward KL" — same math,
#                   different naming. The expectation is taken under the *student*,
#                   so it is the mode-seeking direction.)
# and h_hat / delta_hat are per-batch min-max normalizations of h / delta into [0, 1].
#
# Token selection (Eq. 6):
#     T = TopK({s_t}, floor(rho * m))
#
# Loss (Eq. 7):
#     L_TIP = (1 / |T|) * sum_{t in T} D_KL(P_S || P_T)
#
# IMPORTANT — `beta` is hard-coded:
#     This trainer ALWAYS computes  D_KL(P_S || P_T)  (== GKDTrainer's β=1, reverse KL).
#     The parent's `beta` argument is accepted for signature compatibility but is
#     IGNORED inside `generalized_jsd_loss` below. Whatever value you pass on the CLI
#     (e.g. --beta 1, --beta 0.5, --beta 0) has NO effect on the loss. Set
#     `--beta 1` in your launch script purely so the recorded GKD config is
#     self-consistent with what is actually being computed.

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .arguments import GKDConfig
from .gkd_trainer import GKDTrainer


@dataclass
class GKDSoftORConfig(GKDConfig):
    """GKD config extended with TIP / Soft-OR token-importance selection.

    Args:
        softor_retention_ratio (float): rho in (0, 1]. Fraction of tokens kept by
            top-K Soft-OR score and contributing to the distillation loss. The
            paper reports 50% / 20% / 10%; no universal default. Defaults to 0.5.
        softor_norm_eps (float): Numerical floor added to (max - min) when
            min-max normalizing entropy and divergence. Defaults to 1e-8.
    """
    softor_retention_ratio: float = field(
        default=0.5,
        metadata={
            'help': (
                'TIP retention ratio rho in (0, 1]. Top-rho fraction of tokens by '
                'Soft-OR score contribute to the distillation loss. The paper '
                'reports 0.5, 0.2, 0.1.'
            )
        },
    )
    softor_norm_eps: float = field(
        default=1e-8,
        metadata={'help': 'Numerical floor for the (max - min) denominator in min-max normalization.'},
    )

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 < self.softor_retention_ratio <= 1.0):
            raise ValueError(
                f'softor_retention_ratio must be in (0, 1], got {self.softor_retention_ratio}'
            )


class GKDSoftORTrainer(GKDTrainer):
    """GKD trainer that selects top-rho tokens by the Soft-OR score (TIP).

    Inherits everything from `GKDTrainer` (on-policy rollout, vLLM support, vocab
    alignment, top-k teacher logits, ...) and only overrides `generalized_jsd_loss`
    so that, after the standard label/topk/temperature preprocessing:

      1. We compute per-token reverse KL  delta_t = KL(P_S || P_T)  (with grad).
         (Reverse KL in the GKD-paper sense — the TIP paper calls the same thing
         "forward KL". Mathematically identical: the expectation is taken under
         the *student*, so this is the mode-seeking direction.)
      2. We compute per-token normalized student entropy  h_t = H(P_S) / log|V|.
      3. We min-max normalize h and delta over the valid tokens in the batch,
         then take the Soft-OR score   s = h_hat + d_hat - h_hat * d_hat.
      4. We pick the top  k = floor(rho * num_valid)  tokens by score.
      5. The reported loss is the mean of  delta_t  over selected tokens.

    Note: this trainer **hard-codes the KL direction to reverse KL** (KL(P_S || P_T),
    equivalent to GKDTrainer with β=1). The `beta` argument on the CLI / config is
    IGNORED here — pass `--beta 1` only to keep the recorded GKD config consistent
    with what is actually being computed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softor_retention_ratio: float = getattr(self.args, 'softor_retention_ratio', 0.5)
        self.softor_norm_eps: float = getattr(self.args, 'softor_norm_eps', 1e-8)
        # Buffer of per-step Soft-OR diagnostics. Drained on every `log()` call
        # and merged into the HF Trainer logs dict, so they show up in wandb.
        self._softor_metrics_buf: Dict[str, list] = defaultdict(list)

    def _push_softor_metric(self, name: str, value: float) -> None:
        # NaN / inf guard — degenerate batches (e.g. constant entropy) can yield
        # them in the min-max normalization branch; skip rather than corrupt the
        # running mean.
        if value != value or value in (float('inf'), float('-inf')):
            return
        self._softor_metrics_buf[name].append(float(value))

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Drain the Soft-OR buffer into `logs` so HF Trainer ships them to wandb."""
        if self._softor_metrics_buf:
            for k, vals in self._softor_metrics_buf.items():
                if vals:
                    logs[k] = sum(vals) / len(vals)
            self._softor_metrics_buf.clear()
        super().log(logs, start_time)

    # ------------------------------------------------------------------
    # Core loss override
    # ------------------------------------------------------------------

    def generalized_jsd_loss(
        self,
        student_logits,
        teacher_logits=None,
        labels=None,
        beta=0.5,                # NOTE: ignored — see class / module docstring.
        temperature=1.0,
        chunk_size=512,
        topk=None,
        teacher_topk_logprobs=None,
        teacher_topk_indices=None,
    ):
        # The `beta` kwarg is accepted only to match the parent signature; the
        # KL direction is hard-coded below to reverse KL (KL(P_S || P_T), == GKD
        # β=1). Whatever value the caller passes here has no effect on the loss.
        del beta  # silence linters and make the intent unmistakable

        # ---------- vocab alignment (same as parent) ----------
        if teacher_logits is not None:
            student_logits, teacher_logits = self._align_vocab_size(student_logits, teacher_logits)

        # ---------- top-k mode: gather / slice (same as parent) ----------
        # When the teacher is served with top-k logits only, both entropy and KL
        # are computed on the truncated vocab. This is consistent with the paper:
        # entropy/divergence are defined w.r.t. the distribution actually used
        # for distillation. log|V| in the entropy normalizer therefore uses the
        # tensor's effective vocab size.
        if teacher_topk_logprobs is not None and teacher_topk_indices is not None:
            student_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)
            student_logits.div_(temperature)
            teacher_logits = teacher_topk_logprobs / temperature
            temperature = 1.0
        elif topk is not None and teacher_logits is not None:
            teacher_logits, topk_idx = torch.topk(teacher_logits, k=topk, dim=-1)
            teacher_logits.div_(temperature)
            student_logits = torch.gather(student_logits, dim=-1, index=topk_idx)
            student_logits.div_(temperature)
            temperature = 1.0

        # ---------- label masking (same as parent) ----------
        if labels is not None:
            mask = labels != -100
            # Per-sample on-policy completion length (supervised tokens per row).
            # `labels` is shape [batch, seq] with -100 covering the prompt, so
            # `mask.sum(dim=-1)` is exactly the student-rollout response length.
            with torch.no_grad():
                seq_lengths = mask.sum(dim=-1).float()
                if seq_lengths.numel() > 0:
                    self._push_softor_metric(
                        'softor/completion_length_mean', seq_lengths.mean().item())
                    self._push_softor_metric(
                        'softor/completion_length_min', seq_lengths.min().item())
                    self._push_softor_metric(
                        'softor/completion_length_max', seq_lengths.max().item())
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
            num_valid = mask.sum()
        else:
            student_logits = student_logits.view(-1, student_logits.size(-1))
            teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
            num_valid = student_logits.size(0)
        student_logits.div_(temperature)
        teacher_logits.div_(temperature)

        if num_valid == 0:
            return student_logits.new_zeros(())

        num_valid_int = num_valid if isinstance(num_valid, int) else num_valid.item()

        effective_vocab = student_logits.size(-1)
        log_vocab = math.log(max(effective_vocab, 2))  # guard against |V| == 1

        # ---------- chunked pass: per-token reverse KL (with grad) and entropy ----------
        # KL(P_S || P_T): expectation under the student → mode-seeking direction.
        # (TIP paper calls this "forward KL"; GKD paper calls it "reverse KL", β=1.)
        per_token_kl_chunks = []   # tensors with grad, each shape [chunk]
        per_token_h_chunks = []    # tensors WITHOUT grad, each shape [chunk]

        for start_idx in range(0, num_valid_int, chunk_size):
            end_idx = min(start_idx + chunk_size, num_valid_int)
            s_chunk = student_logits[start_idx:end_idx]
            t_chunk = teacher_logits[start_idx:end_idx]

            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            del s_chunk, t_chunk

            # Reverse KL per token (GKD β=1): KL(P_S || P_T)
            #   = sum_v P_S * (log P_S - log P_T)
            # F.kl_div(input, target, log_target=True) with input=t_log_probs,
            # target=s_log_probs computes sum_v exp(s) * (s - t) = KL(s || t).
            # (TIP paper notation calls this "forward KL" — same math.)
            kl_per_token = F.kl_div(
                t_log_probs, s_log_probs, reduction='none', log_target=True
            ).sum(dim=-1)  # [chunk], with grad
            per_token_kl_chunks.append(kl_per_token)

            # Free the teacher log-probs *before* the entropy computation. They
            # are not needed for entropy and dropping them now makes ~`chunk *
            # vocab * dtype_size` bytes available for the `slp.exp()` temp.
            del t_log_probs

            # Normalized student entropy (no grad — only used for ranking, so
            # fp32 precision is overkill). Two memory-saving tweaks vs. the
            # naive `-(slp.exp() * slp).sum()`:
            #   1. Cast the detached log-probs down to the model's compute
            #      dtype (typically bf16). Under autocast, `F.log_softmax`
            #      returns fp32 for numerical stability, which would make
            #      `slp.exp()` a `[chunk, vocab]` *fp32* tensor — twice the
            #      memory of the underlying logits.
            #   2. Use in-place `mul_` so `exp(slp)` and `exp(slp) * slp` do
            #      not co-exist as two separate `[chunk, vocab]` tensors.
            # Together these cut the entropy peak transient memory ~4×, which
            # is what was triggering the OOM at line 230 on long completions.
            with torch.no_grad():
                slp = s_log_probs.detach()
                if slp.dtype == torch.float32 and student_logits.dtype != torch.float32:
                    slp = slp.to(student_logits.dtype)
                probs = slp.exp()           # [chunk, vocab]
                probs.mul_(slp)             # in-place: probs = exp(slp) * slp
                h_per_token = probs.sum(dim=-1).neg_().div_(log_vocab)  # [chunk]
                del probs, slp
            per_token_h_chunks.append(h_per_token)

            del s_log_probs

        per_token_kl = torch.cat(per_token_kl_chunks, dim=0)  # [num_valid] with grad
        per_token_h = torch.cat(per_token_h_chunks, dim=0)    # [num_valid] no grad
        del per_token_kl_chunks, per_token_h_chunks

        # ---------- Soft-OR scoring (no grad) ----------
        with torch.no_grad():
            delta = per_token_kl.detach()
            eps = self.softor_norm_eps

            h_min, h_max = per_token_h.min(), per_token_h.max()
            d_min, d_max = delta.min(), delta.max()

            h_hat = (per_token_h - h_min) / (h_max - h_min).clamp_min(eps)
            d_hat = (delta - d_min) / (d_max - d_min).clamp_min(eps)
            # numerical safety
            h_hat = h_hat.clamp_(0.0, 1.0)
            d_hat = d_hat.clamp_(0.0, 1.0)

            # Eq. 5
            score = h_hat + d_hat - h_hat * d_hat

            # Eq. 6: top-rho selection. Always keep at least one token to avoid
            # a zero-division on tiny micro-batches.
            k = max(1, min(num_valid_int, int(self.softor_retention_ratio * num_valid_int)))
            _, topk_idx = torch.topk(score, k=k)
            selection_mask = torch.zeros_like(score)
            selection_mask[topk_idx] = 1.0

            # ---- Diagnostics — drained by `log()` into HF Trainer logs ----
            sel_bool = selection_mask.bool()
            n_sel = int(sel_bool.sum().item())
            n_drp = num_valid_int - n_sel

            self._push_softor_metric('softor/num_valid', float(num_valid_int))
            self._push_softor_metric('softor/num_selected', float(n_sel))
            self._push_softor_metric('softor/selected_ratio', n_sel / max(num_valid_int, 1))
            self._push_softor_metric('softor/score_min', score.min().item())
            self._push_softor_metric('softor/score_max', score.max().item())
            self._push_softor_metric('softor/score_mean', score.mean().item())
            self._push_softor_metric('softor/h_mean', per_token_h.mean().item())
            self._push_softor_metric('softor/delta_mean', delta.mean().item())

            # Selected vs dropped — these are the diagnostics that tell you
            # whether Soft-OR is actually picking "informative" tokens.
            if n_sel > 0:
                self._push_softor_metric('softor/h_mean_selected', per_token_h[sel_bool].mean().item())
                self._push_softor_metric('softor/delta_mean_selected', delta[sel_bool].mean().item())
            if n_drp > 0:
                drp_bool = ~sel_bool
                self._push_softor_metric('softor/h_mean_dropped', per_token_h[drp_bool].mean().item())
                self._push_softor_metric('softor/delta_mean_dropped', delta[drp_bool].mean().item())

        # ---------- Eq. 7: reverse KL averaged over selected tokens ----------
        denom = selection_mask.sum().clamp_min(1.0)
        loss = (per_token_kl * selection_mask).sum() / denom
        return loss
