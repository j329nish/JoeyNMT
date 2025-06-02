# coding: utf-8
"""
Module to implement training loss
"""
import torch
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0, lm_prior: dict = None):
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        self.kl_lambda = lm_prior.get("kl_lambda", 0.0)
        self.kl_tau = lm_prior.get("kl_tau", 0.0)
        self.token = lm_prior.get("access_token_name", None)
        self.model_name = lm_prior.get("model_file", None)
        if all([self.kl_lambda, self.kl_tau, self.model_name, self.token]):
            self.lm_model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.token)
            self.lm_tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.token)
            special_tokens_dict = {"pad_token": "<|finetune_right_pad_id|>", "unk_token": "<|reserved_special_token_0|>"}
            self.lm_tokenizer.add_special_tokens(special_tokens_dict)
        else:
            self.lm_model = None
        self.criterion: _Loss  # (type annotation)
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets: Tensor, vocab_size: int) -> Variable:
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(
            targets.data == self.pad_index, as_tuple=False
        )
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    def _reshape(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        vocab_size = log_probs.size(-1)

        # reshape log_probs to (batch*seq_len x vocab_size)
        log_probs_flat = log_probs.contiguous().view(-1, vocab_size)

        if self.smoothing > 0:
            targets_flat = self._smooth_targets(
                targets=targets.contiguous().view(-1), vocab_size=vocab_size
            )
            # targets: distributions with batch*seq_len x vocab_size
            assert log_probs_flat.size() == targets_flat.size(), (
                log_probs.size(),
                targets_flat.size(),
            )
        else:
            # targets: indices with batch*seq_len
            targets_flat = targets.contiguous().view(-1)
            assert log_probs_flat.size(0) == targets_flat.size(0), (
                log_probs.size(0),
                targets_flat.size(0),
            )

        return log_probs_flat, targets_flat

    def forward(self, log_probs: Tensor, kl_log_probs: Tensor, **kwargs) -> Tensor:
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :return: logits
        """
        assert "trg" in kwargs
        log_probs, targets = self._reshape(log_probs, kwargs["trg"])

        # compute loss
        logits = self.criterion(log_probs, targets)
        if all([self.kl_lambda, self.kl_tau, self.lm_model]):
            kl_log_probs, _ = self._reshape(kl_log_probs, kwargs["trg"])
            with torch.no_grad():
                lm_input_ids = insert_eos_before_padding(
                    kwargs["trg_input"], 
                    eos_token_id=self.lm_tokenizer.eos_token_id, 
                    pad_token_id=self.lm_tokenizer.pad_token_id
                    )
                lm_inputs = {
                    "input_ids": lm_input_ids.to(log_probs.device),
                    "attention_mask": (lm_input_ids != self.lm_tokenizer.pad_token_id).to(log_probs.device)
                    }
                lm_logits = self.lm_model(**lm_inputs).logits[:, 1:, :]
                lm_probs = F.softmax(lm_logits / self.kl_tau, dim=-1)
                lm_probs_flat = lm_probs.reshape(-1, lm_probs.size(-1))
                non_pad_mask = (kwargs["trg"].contiguous().view(-1) != self.pad_index)
            return logits + self.kl_lambda * self.kl_tau * self.kl_tau * F.kl_div(kl_log_probs[non_pad_mask], lm_probs_flat[non_pad_mask], reduction='batchmean')
        else:
            return logits

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(criterion={self.criterion}, "
            f"smoothing={self.smoothing})"
        )
    
def insert_eos_before_padding(input_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> torch.Tensor:
    B, L = input_ids.size()
    output_ids = torch.full((B, L + 1), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)

    for i in range(B):
        seq = input_ids[i]
        pad_pos = (seq == pad_token_id).nonzero(as_tuple=True)[0]
        if len(pad_pos) > 0:
            insert_pos = pad_pos[0].item()
        else:
            insert_pos = L

        output_ids[i, :insert_pos] = seq[:insert_pos]
        output_ids[i, insert_pos] = eos_token_id
        output_ids[i, insert_pos + 1:L + 1] = seq[insert_pos:]

    return output_ids
