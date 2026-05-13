# Copyright (c) ModelScope Contributors. All rights reserved.
#
# GKD with confidence-aware entropy-RATIO filtering (quantile mode).
#
# Motivation:
#   Standard GKD applies the JSD/KL loss equally at every token position.
#   When the teacher distribution is much MORE uncertain than the student
#   (i.e. the student is already sharper than the teacher about the next
#   token), forcing the student to match the teacher injects noise and can
#   actively hurt the student.  This trainer SKIPS those positions and only
#   back-propagates through positions where the teacher is relatively more
#   confident than the student.
#
# Filter:
#   Let H_t and H_s be the per-token entropies of the teacher and student
#   distributions (both in nats, computed over the same vocab / top-k slice).
#   Define the per-token ratio
#
#       r_i = H_t,i / max(H_s,i, eps)
#
#   For each (per-rank, per-microbatch) loss call we keep the
#   ``entropy_ratio_keep_fraction`` fraction of tokens with the SMALLEST
#   r_i and drop the rest.  Intuitively low r_i means "the teacher is at
#   least as confident as the student here" — the supervisory signal is
#   trustworthy.
#
#   When the student is fully deterministic (H_s = 0) we clamp the
#   denominator with a tiny epsilon, so such tokens get a very large
#   ratio and are naturally dropped — which matches the intent: if the
#   student is already 100% committed, we only let an equally committed
#   teacher pull it.
#
# Implementation notes:
#   The loss is computed in TWO PASSES to avoid wasted compute on tokens
#   that will be dropped:
#       Pass 1 (no_grad) chunk-wise computes the entropy ratio for every
#               valid token (forward log_softmax only, no autograd graph
#               is retained) and selects the k indices with the smallest
#               ratios via torch.topk.
#       Pass 2 (with grad) gathers the kept logits and chunk-wise computes
#               log_softmax + JSD/KL only on the kept tokens, then divides
#               the summed loss by k.
#   With keep_fraction = 0.5 this saves roughly 25% of the per-step flops
#   compared with computing the loss on all tokens and masking afterwards,
#   and it also halves the autograd graph memory footprint in the backward
#   pass.
#
# Background reference:
#   EOPD (Entropy-Aware On-Policy Distillation, ICML 2025, arxiv 2603.07079).
#   The original confidence-aware variant of this trainer used an absolute
#   teacher-entropy threshold (H_t < tau).  This file replaces that with a
#   per-batch quantile filter on the teacher/student entropy ratio.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .arguments import GKDConfig
from .gkd_trainer import GKDTrainer

# Numerical floor for the student-entropy denominator.  Student entropy is in
# nats, so 1e-8 is well below any realistic value but large enough to keep the
# division safe in bf16/fp16.
_STUDENT_ENTROPY_EPS = 1e-8


@dataclass
class GKDConfidenceConfig(GKDConfig):
    """GKD config extended with a teacher/student entropy-ratio quantile filter.

    Args:
        entropy_ratio_keep_fraction (float): Fraction in (0, 1] of the
            valid tokens (per microbatch, per rank) to keep, ranked by the
            entropy ratio H_teacher / H_student in ASCENDING order. The
            tokens with the smallest ratios — i.e. those where the teacher
            is most confident relative to the student — are kept, and the
            rest are dropped from the GKD/JSD loss.
            Default: 0.5 (keep the lowest-ratio 50%).
    """
    entropy_ratio_keep_fraction: float = field(
        default=0.5,
        metadata={
            'help': (
                'Per-microbatch fraction of tokens to keep, ranked by the '
                'entropy ratio H_teacher / H_student ascending. Tokens with '
                'the smallest ratios (teacher most confident relative to '
                'student) are kept. Range: (0, 1]. Default: 0.5.'
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()


class GKDConfidenceTrainer(GKDTrainer):
    """GKD trainer that keeps only the lowest-ratio fraction of tokens.

    Inherits everything from GKDTrainer (on-policy rollout, vLLM support,
    etc.) and only changes ``generalized_jsd_loss``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Read the keep fraction from the HF training args
        # (GKDConfidenceConfig). Fall back to 0.5 if the config is the plain
        # GKDConfig (shouldn't happen in normal usage, but safe to guard).
        self.entropy_ratio_keep_fraction: float = float(
            getattr(self.args, 'entropy_ratio_keep_fraction', 0.5))
        if not (0.0 < self.entropy_ratio_keep_fraction <= 1.0):
            raise ValueError(
                'entropy_ratio_keep_fraction must lie in (0, 1], got '
                f'{self.entropy_ratio_keep_fraction!r}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _per_token_jsd(s_log_probs, t_log_probs, beta, beta_t, log_beta, log_1_minus_beta):
        """Per-token JSD/KL value (sum over vocab) for a chunk.

        Returns a 1-D tensor of shape ``[chunk]``.
        """
        if beta == 0:
            # KL(teacher || student) — reverse KL in on-policy GKD
            jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
        elif beta == 1:
            # KL(student || teacher) — forward KL
            jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
        else:
            # Generalised JSD with mixture distribution
            mixture_log_probs = torch.logsumexp(
                torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
            kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
            del mixture_log_probs
            jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
            del kl_teacher, kl_student
        return jsd_chunk.sum(dim=-1)

    # ------------------------------------------------------------------
    # Core loss override
    # ------------------------------------------------------------------

    def generalized_jsd_loss(
        self,
        student_logits,
        teacher_logits=None,
        labels=None,
        beta=0.5,
        temperature=1.0,
        chunk_size=512,
        topk=None,
        teacher_topk_logprobs=None,
        teacher_topk_indices=None,
    ):
        """Two-pass JSD loss with entropy-ratio quantile filtering.

        See module docstring for the full description.  Behaviour summary:

          * Vocab alignment, top-k handling, label masking and temperature
            scaling are identical to the parent's ``generalized_jsd_loss``.
          * After flattening to ``[num_valid, V]`` we run
              Pass 1: no_grad chunked log_softmax + entropy + ratio
                      → torch.topk(largest=False) selects keep_indices
              Pass 2: index_select gathers the kept logits, then chunked
                      log_softmax + JSD with grad on those rows only.
          * The final loss is ``sum(per_token_loss[kept]) / num_kept``.

        Note on top-k mode: when ``topk`` / ``teacher_topk_*`` is used,
        student and teacher log-probs are restricted to the SAME top-k
        vocabulary slice (the teacher's top-k indices). Both entropies are
        therefore computed over the same support and the ratio remains
        well-defined.

        Note on cross-vocab: the parent's ``_align_vocab_size`` only handles
        the case where teacher and student share the same tokenizer and
        differ only in head size (the extra rows are filled by copying the
        larger model's logits, which makes their KL contribution zero).
        Truly different tokenizers are NOT supported by this code path.
        """
        # ---------- vocab alignment (same as parent) ----------
        if teacher_logits is not None:
            student_logits, teacher_logits = self._align_vocab_size(student_logits, teacher_logits)

        # ---------- top-k mode: gather / slice (same as parent) ----------
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

        # Pre-compute the (log_)beta scalars once.
        if beta != 0 and beta != 1:
            beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

        # ============================================================
        # Pass 1 (no_grad): compute the per-token entropy ratio and pick
        # the k smallest-ratio token indices.  Skipped when keep_frac=1.0.
        # ============================================================
        keep_frac = self.entropy_ratio_keep_fraction
        if keep_frac >= 1.0:
            keep_indices = None
            num_kept = num_valid_int
        else:
            with torch.no_grad():
                ratio_chunks = []
                for start_idx in range(0, num_valid_int, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_valid_int)
                    s_log_probs = F.log_softmax(student_logits[start_idx:end_idx], dim=-1)
                    t_log_probs = F.log_softmax(teacher_logits[start_idx:end_idx], dim=-1)
                    # Cast entropies to fp32 to keep topk ranking stable in
                    # bf16/fp16 training.
                    t_entropy = -(t_log_probs.exp() * t_log_probs).sum(dim=-1).float()
                    s_entropy = -(s_log_probs.exp() * s_log_probs).sum(dim=-1).float()
                    ratio_chunks.append(t_entropy / s_entropy.clamp_min(_STUDENT_ENTROPY_EPS))
                    del s_log_probs, t_log_probs, t_entropy, s_entropy
                ratios = torch.cat(ratio_chunks)  # [num_valid], fp32
                del ratio_chunks

            num_kept = int(num_valid_int * keep_frac + 0.5)
            num_kept = max(min(num_kept, num_valid_int), 1)
            _, keep_indices = torch.topk(ratios, num_kept, largest=False, sorted=False)
            del ratios
            # Sorting keep_indices does not change correctness but tends to
            # improve memory locality of the subsequent index_select.
            keep_indices, _ = torch.sort(keep_indices)

        # ============================================================
        # Pass 2 (with grad): chunked log_softmax + JSD/KL on kept tokens.
        # ============================================================
        if keep_indices is not None:
            student_logits = student_logits.index_select(0, keep_indices)
            teacher_logits = teacher_logits.index_select(0, keep_indices)

        total_loss = student_logits.new_zeros(())
        for start_idx in range(0, num_kept, chunk_size):
            end_idx = min(start_idx + chunk_size, num_kept)
            s_chunk = student_logits[start_idx:end_idx]
            t_chunk = teacher_logits[start_idx:end_idx]

            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            del s_chunk, t_chunk

            per_token_loss = self._per_token_jsd(
                s_log_probs, t_log_probs, beta, beta_t, log_beta, log_1_minus_beta)
            del s_log_probs, t_log_probs

            total_loss = total_loss + per_token_loss.sum()
            del per_token_loss

        return total_loss / num_kept
