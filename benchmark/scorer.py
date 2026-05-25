import os
from typing import List

import torch
import numpy as np
import torch.nn.functional as F


def get_js_divs(
    final_logits: torch.FloatTensor,
    premature_layers_logits: torch.FloatTensor,
) -> torch.FloatTensor:

    # 2. Calculate the softmax values for mature_layer and all premature_layers
    # shape: (batch_size, vocab_size)
    softmax_mature_layer = F.softmax(final_logits, dim=-1)
    # print("softmax_mature_layer", softmax_mature_layer.shape)

    # shape: (num_premature_layers, batch_size, vocab_size)
    softmax_premature_layers = F.softmax(premature_layers_logits, dim=-1)
    # print("softmax_premature_layers", softmax_premature_layers.shape)

    # 3. Calculate the average distribution
    # shape: (num_premature_layers, batch_size, vocab_size)
    avg_dist = 0.5 * (softmax_mature_layer + softmax_premature_layers)

    # 4. Calculate log-softmax for the KL divergence
    # shape: (batch_size, vocab_size)
    log_softmax_mature_layer = F.log_softmax(final_logits, dim=-1)

    # shape: (num_premature_layers, batch_size, vocab_size)
    log_softmax_premature_layers = F.log_softmax(premature_layers_logits, dim=-1)

    # 5. Calculate the KL divergences and then the JS divergences
    # shape: (num_premature_layers, batch_size)
    kl1 = F.kl_div(log_softmax_mature_layer, avg_dist, reduction="none").mean(-1)
    # shape: (num_premature_layers, batch_size)
    kl2 = F.kl_div(log_softmax_premature_layers, avg_dist, reduction="none").mean(-1)

    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)
    js_divs = js_divs.mean(-1)

    return js_divs  #  js_divs.detach().cpu().float().numpy()


class CDScorer:
    def __init__(self, model1, model2, tokenizer, device="cuda"):
        self.model = model1
        self.model_cd = model2
        self.tokenizer = tokenizer
        self.device = device

    def get_relative_top_filter(
        self,
        scores: torch.FloatTensor,
        relative_top: float = 0.1,
        min_tokens_to_keep: int = 1,
    ):
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores < probs_thresh

    def get_filter_new(
        self,
        scores: torch.FloatTensor,
        relative_top: float = 0.1,
        min_tokens_to_keep: int = 1,
    ):
        # scores_normalized = scores.log_softmax(dim=-1)
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def lm_score(
        self,
        input_text1,
        input_text2,
        mode="baseline",
        cd_temperature=1.0,
        relative_top=0.1,
        relative_top_value=-1000.0,
        post_softmax=True,
        mature_layer=None,
        premature_layer=None,
        candidate_premature_layers=None,
        layer_indices: List[List[int]] = None,
        head_indices: List[List[int]] = None,
        alpha=0.0,
        verbose=False,
    ):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
                self.device
            )
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(
                self.device
            )
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]

            if mode == "greedy":
                outputs = self.model(input_ids)[0].squeeze(0)  # (seq_len, vocab_size)
                outputs = outputs.log_softmax(dim=-1)
                relevant_outputs = outputs[prefix_ids.shape[-1] - 1 : -1, :]
                relevant_outputs = relevant_outputs.float()
                log_probs = (
                    relevant_outputs[range(relevant_outputs.shape[0]), continue_ids]
                    .sum()
                    .item()
                )

            elif mode == "prunecd":
                outputs_model = self.model(input_ids)[0].squeeze(
                    0
                )  # (seq_len, vocab_size)
                outputs_model_cd = self.model_cd(input_ids)[0].squeeze(
                    0
                )  # (seq_len, vocab_size)

                original_log_probs = F.log_softmax(outputs_model, dim=-1)
                pruned_log_probs = F.log_softmax(outputs_model_cd, dim=-1)

                original_log_probs = original_log_probs[
                    prefix_ids.shape[-1] - 1 : -1, :
                ]
                pruned_log_probs = pruned_log_probs[prefix_ids.shape[-1] - 1 : -1, :]

                top_tokens_final = torch.argmax(original_log_probs, dim=-1)
                top_tokens_cd = torch.argmax(pruned_log_probs, dim=-1)

                cd_log_probs = original_log_probs - cd_temperature * pruned_log_probs

                cd_log_probs = cd_log_probs.float()

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(
                        original_log_probs, relative_top
                    )
                    cd_log_probs = torch.where(
                        relative_top_mask,
                        torch.tensor(relative_top_value, device=cd_log_probs.device),
                        cd_log_probs,
                    )

                log_probs = (
                    cd_log_probs[range(cd_log_probs.shape[0]), continue_ids]
                    .sum()
                    .item()
                )

            elif mode == "att_cd":

                scale_alpha = False

                def calculate_entropy(logits):
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

                    if scale_alpha:
                        entropy = entropy / np.log(probs.shape[-1])

                    return entropy

                expert_outputs = self.model(input_ids)
                outputs_model = expert_outputs[0].squeeze(0)  # (seq_len, vocab_size)

                premature_outputs = self.model_cd(input_ids)
                outputs_model_cd = premature_outputs[0].squeeze(
                    0
                )  # (seq_len, vocab_size)

                original_log_probs = F.log_softmax(outputs_model, dim=-1)
                premature_log_probs = F.log_softmax(outputs_model_cd, dim=-1)

                original_log_probs = original_log_probs[
                    prefix_ids.shape[-1] - 1 : -1, :
                ]
                premature_log_probs = premature_log_probs[
                    prefix_ids.shape[-1] - 1 : -1, :
                ]

                if os.environ.get("ATT_CD_MODE") == "decore":
                    # print("using alpha calculated from entropy")
                    alpha = calculate_entropy(expert_outputs.logits[0, -1])
                    cd_log_probs = (
                        1 + alpha
                    ) * original_log_probs - alpha * premature_log_probs
                else:
                    cd_log_probs = (
                        1 + cd_temperature
                    ) * original_log_probs - cd_temperature * premature_log_probs

                cd_log_probs = cd_log_probs.float()

                if post_softmax:
                    cd_log_probs = cd_log_probs.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(
                        original_log_probs, relative_top
                    )
                    cd_log_probs = torch.where(
                        relative_top_mask,
                        torch.tensor(relative_top_value, device=cd_log_probs.device),
                        cd_log_probs,
                    )

                log_probs = (
                    cd_log_probs[range(cd_log_probs.shape[0]), continue_ids]
                    .sum()
                    .item()
                )

            elif mode == "dola":
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=True,
                )

                dict_outputs = {}
                for idx, state in enumerate(outputs.hidden_states):
                    dict_outputs[idx] = self.model.lm_head(state)

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack(
                        [
                            dict_outputs[i][:, seq_i, :]
                            for i in candidate_premature_layers
                        ],
                        dim=0,
                    )

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(
                        dict_outputs[mature_layer][:, seq_i, :], dim=-1
                    )  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(
                        stacked_premature_layers, dim=-1
                    )  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (
                        softmax_mature_layer[None, :, :] + softmax_premature_layers
                    )  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(
                        dict_outputs[mature_layer][:, seq_i, :], dim=-1
                    )  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(
                        stacked_premature_layers, dim=-1
                    )  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(
                        log_softmax_mature_layer[None, :, :], M, reduction="none"
                    ).mean(
                        -1
                    )  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(
                        log_softmax_premature_layers, M, reduction="none"
                    ).mean(
                        -1
                    )  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (
                        kl1 + kl2
                    )  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[
                        int(js_divs.argmax().cpu().item())
                    ]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(
                    dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1 : -1]
                )
                for i, l in enumerate(premature_layers):
                    base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]

                final_logits = dict_outputs[mature_layer][
                    0, prefix_ids.shape[-1] - 1 : -1
                ]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits

                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                diff_logits = diff_logits.float()

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(
                        final_logits, relative_top
                    )
                    diff_logits = torch.where(
                        relative_top_mask, relative_top_value, diff_logits
                    )

                log_probs = (
                    diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                )

                return log_probs, premature_layers

            elif mode == "actd":
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}
                info_layer = candidate_premature_layers[0]
                premature_layers = []

                outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=True,
                )

                dict_outputs = {}
                for idx, state in enumerate(outputs.hidden_states):
                    dict_outputs[idx] = self.model.lm_head(state)

                final_logits = dict_outputs[mature_layer][
                    0, prefix_ids.shape[-1] - 1 : -1
                ]
                final_logits = final_logits.log_softmax(dim=-1)
                final_logits = final_logits.float()

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(
                        final_logits, relative_top
                    )
                    final_logits = torch.where(
                        relative_top_mask, relative_top_value, final_logits
                    )

                mask = (
                    final_logits[0] < relative_top_value
                )  # | (final_logits[0] == relative_top_value)

                info_layer_score = dict_outputs[info_layer][-1, :, :]
                index_nontop = torch.argwhere(mask).squeeze()
                info_layer_probs = F.softmax(
                    torch.t(info_layer_score), dim=1
                ).unsqueeze(
                    0
                )  # info_layer_score: [num_token_in_question, len_token_lib] -> e.g. [250, 32000]
                entropy = torch.distributions.Categorical(
                    probs=info_layer_probs, validate_args=False
                ).entropy()

                entropy = entropy.scatter(1, index_nontop.unsqueeze(0), float("Inf"))

                if alpha != 0:
                    # entropy: the smaller the better
                    final_logits = final_logits + alpha * (-entropy)

                log_probs = (
                    final_logits[range(final_logits.shape[0]), continue_ids]
                    .sum()
                    .item()
                )

            elif mode == "end":
                outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=True,
                )

                dict_outputs = {}
                for idx, state in enumerate(outputs.hidden_states):
                    dict_outputs[idx] = self.model.lm_head(state)

                mat_logits = dict_outputs[mature_layer][
                    0, prefix_ids.size(-1) - 1 : -1
                ]  # [S, V]

                mat_probs = F.softmax(mat_logits, dim=-1)  # [S,V]
                max_probs = mat_probs.max(dim=-1, keepdim=True).values  # [S,1]
                vhead_mask = mat_probs >= relative_top * max_probs  # [S,V]
                # pdb.set_trace()
                layer_probs = []  # to stack
                for l in candidate_premature_layers:
                    lp = F.softmax(
                        dict_outputs[l][0, prefix_ids.size(-1) - 1 : -1], dim=-1
                    )  # [S,V]
                    layer_probs.append(lp)

                layer_probs = torch.stack(layer_probs, dim=-1)  # [S,V,L]
                layer_probs *= vhead_mask.unsqueeze(-1)  # V_head 밖 0
                denom = layer_probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
                q = layer_probs / denom
                entropy = -(q * q.clamp_min(1e-20).log()).sum(dim=-1)  # [S,V]
                entropy = entropy.masked_fill(~vhead_mask, float("inf"))  # 식 5

                mat_logits_log = F.log_softmax(mat_logits, dim=-1)
                diff_logits = mat_logits_log - alpha * entropy
                diff_logits = torch.where(~vhead_mask, relative_top_value, diff_logits)

                log_probs = (
                    diff_logits[range(diff_logits.size(0)), continue_ids].sum().item()
                )

            else:
                raise ValueError(f"mode='{mode}'")

        return log_probs
