import copy
import itertools
import time
from typing import Generator, Iterable, Iterator

import jaxtyping as typ
import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch import cosine_similarity, nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import MistralForCausalLM, PreTrainedModel, PreTrainedTokenizer

from atm_utils import chunk_list

LR = 0.01
WD = 0.0
BS = 5
MBS = 200
TARGET_LAYER = -1


def create_trainable_atm_emb(new_phrase_ids, model_input_embedding: nn.Embedding, device=None, dtype=None):
    initial_atm_emb = torch.mean(model_input_embedding(new_phrase_ids), dim=0)  # .to(torch.float32)
    if device is not None:
        initial_atm_emb = initial_atm_emb.to(device)
    if dtype is not None:
        initial_atm_emb = initial_atm_emb.to(dtype)
    return nn.Parameter(initial_atm_emb).requires_grad_(True)


def batch_atm_for_many_tokens(
    new_phrases_ids,
    new_phrases_train_snippets,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_snippets_per_phrase=50,
):
    """
    We need to be efficient, so we cannot just do the ATM optimization for each new token one by one.
    """
    DEVICE = model.device
    NUM_PHRASES = len(new_phrases_ids)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model_input_embs_cpu = copy.deepcopy(model.get_input_embeddings()).cpu()

    # init atm embs as mean of constituent tokens (FVT-style). cast to float32 to avoid numerical issues -- we cast to bfloat16 in the fwd pass if necessary
    new_phrases_trainable_atm_embs = [
        create_trainable_atm_emb(new_phrase_ids, model_input_embs_cpu, DEVICE, torch.float32)
        for new_phrase_ids in new_phrases_ids
    ]

    batched_dataset = BatchedSnippetDataset(
        new_phrases_ids,
        new_phrases_train_snippets,
        # new_phrases_trainable_atm_embs,
        model_input_embs_cpu,
        MBS,
    )
    data_loader = torch.utils.data.DataLoader(batched_dataset, batch_size=None, shuffle=False, num_workers=8, pin_memory=True)
    dataloader_iter = iter(data_loader)

    loss_fn = torch.nn.MSELoss(reduction="mean")

    phrases_bar = tqdm(
        chunk_list(range(NUM_PHRASES), MBS),
        desc=f"ATM for {NUM_PHRASES} new phrases",
        total=NUM_PHRASES // MBS,
    )
    stats_per_opt_step = [
        {"error": 0, "gradnorm": 0, "embnorm": 0, "diff_to_initial_cos": 0, "diff_to_inital_norm": 0}
        for _ in range(num_snippets_per_phrase // BS)
    ]

    for cur_phrases_idxs in phrases_bar:
        cur_phrases_idxs = list(cur_phrases_idxs)
        cur_phrases_tokens = ["".join(tokenizer.convert_ids_to_tokens(new_phrases_ids[i])) for i in cur_phrases_idxs]
        phrases_bar.set_description(f"ATM for {NUM_PHRASES} new phrases | e.g. {cur_phrases_tokens[:3]}...")

        cur_new_phrase_atm_embs = [new_phrases_trainable_atm_embs[i] for i in cur_phrases_idxs]
        cur_initial_new_phrase_atm_embs = [new_phrases_trainable_atm_embs[i].clone().detach() for i in cur_phrases_idxs]

        optimizer = torch.optim.AdamW(cur_new_phrase_atm_embs, lr=LR, weight_decay=WD)
        optimizer.zero_grad()
        NUM_OPT_STEPS = num_snippets_per_phrase // BS
        # if cur_phrases_idxs[0] > 0:
        #     # `batched_dataset.num_snippets_per_phrase()` is not available until the first batch is processed
        #     assert num_snippets_per_phrase == batched_dataset.num_snippets_per_phrase()
        cur_opt_step = 1
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=int(NUM_OPT_STEPS * 0.5))
        fwd_pass_bar = tqdm(
            range(num_snippets_per_phrase),
            desc=f"Fwd passes (Step {cur_opt_step}/{NUM_OPT_STEPS})...",
            leave=False,
        )

        MistralForCausalLM
        avg_cur_step_error = 0
        ema_error = None
        total_data_fetch_time = 0
        last_snippet_idx = -1
        for fwdpass_idx in fwd_pass_bar:
            data_fetch_t0 = time.perf_counter()
            batch = next(dataloader_iter)
            cur_snippet_idx = batch["snippet_idx"]
            assert cur_snippet_idx > last_snippet_idx
            last_snippet_idx = cur_snippet_idx
            batch["snippets_toks"] = batch["snippets_toks"].to(DEVICE, non_blocking=True)
            batch["padded_atm_seqs_embs"] = batch["padded_atm_seqs_embs"].to(DEVICE, non_blocking=True)

            assert batch["phrases_idxs"] == cur_phrases_idxs
            # replace all placeholder atm embeddings with the actual trainable atm embeddings
            for i, atm_token_idx in enumerate(batch["new_phrase_in_atm_seq_idxs"]):
                batch["padded_atm_seqs_embs"][i, atm_token_idx] = cur_new_phrase_atm_embs[i].to(
                    batch["padded_atm_seqs_embs"][0].dtype
                )

            assert batch["padded_atm_seqs_embs"][0, batch["new_phrase_in_atm_seq_idxs"][0]].requires_grad
            data_fetch_t1 = time.perf_counter()
            total_data_fetch_time += data_fetch_t1 - data_fetch_t0
            # print(f"Data fetch time: {data_fetch_t1 - data_fetch_t0:.4f}")

            # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            # ground truth forward pass - no_grad!
            with torch.no_grad():
                out = model(
                    batch["snippets_toks"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                batched_gt_hiddens = out["hidden_states"][TARGET_LAYER]
                del out

            # forward pass with the merged pattern
            out = model(
                inputs_embeds=batch["padded_atm_seqs_embs"],
                output_hidden_states=True,
                return_dict=True,
            )
            batched_phrase_snippets_atm_hiddens = out["hidden_states"][TARGET_LAYER]
            del out

            # remove padding
            batched_phrase_snippets_atm_hiddens = [
                t[: batch["padded_atm_seqs_embs_orig_lengths"][i], :]
                for i, t in enumerate(batched_phrase_snippets_atm_hiddens)
            ]

            assert len(batched_gt_hiddens) == len(batched_phrase_snippets_atm_hiddens)

            for i in range(len(batched_gt_hiddens)):
                # assert batched_gt_hiddens[i].size(0) == batched_phrase_snippets_atm_hiddens[i].size(0)
                assert batched_phrase_snippets_atm_hiddens[i].size(0) == batch["padded_atm_seqs_embs_orig_lengths"][i]
                assert batched_phrase_snippets_atm_hiddens[i].size(0) == batched_gt_hiddens[i].size(0) - (
                    new_phrases_ids[cur_phrases_idxs[i]].size(0) - 1
                ) * len(batch["new_phrase_in_gt_seq_idxs"][i])

            # align gt sequence hiddens and atm merged sequence hiddens
            batched_non_pattern_hiddens_gt, batched_non_pattern_hiddens_merged = [], []
            batched_pattern_last_hiddens_gt, batched_pattern_hiddens_merged = [], []

            for pi in range(len(cur_phrases_idxs)):
                gt_hiddens = batched_gt_hiddens[pi]
                atm_hiddens = batched_phrase_snippets_atm_hiddens[pi]
                patterns_in_gt_seq_idx = batch["new_phrase_in_gt_seq_idxs"][pi]
                patterns_in_merged_seq_idx = batch["new_phrase_in_atm_seq_idxs"][pi]

                (
                    non_pattern_hiddens_gt,
                    non_pattern_hiddens_merged,
                    pattern_last_hiddens_gt,
                    pattern_hiddens_merged,
                ) = align_hidden_states_w_merged_pattern(
                    gt_hiddens,
                    atm_hiddens,
                    patterns_in_gt_seq_idx,
                    patterns_in_merged_seq_idx,
                    new_phrase_ids_len=len(new_phrases_ids[cur_phrases_idxs[pi]]),
                )
                assert non_pattern_hiddens_gt.size(0) == non_pattern_hiddens_merged.size(0)
                assert pattern_last_hiddens_gt.size(0) == pattern_hiddens_merged.size(0)
                batched_non_pattern_hiddens_gt.append(non_pattern_hiddens_gt)
                batched_non_pattern_hiddens_merged.append(non_pattern_hiddens_merged)
                batched_pattern_last_hiddens_gt.append(pattern_last_hiddens_gt)
                batched_pattern_hiddens_merged.append(pattern_hiddens_merged)

            all_hiddens_gt = torch.cat(batched_non_pattern_hiddens_gt + batched_pattern_last_hiddens_gt)
            all_hiddens_merged = torch.cat(batched_non_pattern_hiddens_merged + batched_pattern_hiddens_merged)
            error = loss_fn(all_hiddens_merged, all_hiddens_gt) / BS
            error.backward(inputs=cur_new_phrase_atm_embs)

            avg_cur_step_error += error.item()

            if (fwdpass_idx + 1) % BS == 0:
                avg_cur_step_error /= BS
                avg_pre_clip_grad_norm = (
                    torch.norm(torch.stack([emb.grad for emb in cur_new_phrase_atm_embs]), dim=-1).mean().item()
                )
                torch.nn.utils.clip_grad_norm_(cur_new_phrase_atm_embs, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                cur_opt_step += 1

                with torch.no_grad():
                    avg_diff_norms_to_initial = (
                        torch.norm(torch.stack(cur_new_phrase_atm_embs) - torch.stack(cur_initial_new_phrase_atm_embs), dim=-1)
                        .mean()
                        .item()
                    )
                    avg_cosine_sims_w_initial = (
                        cosine_similarity(
                            torch.stack(cur_new_phrase_atm_embs), torch.stack(cur_initial_new_phrase_atm_embs), dim=-1
                        )
                        .mean()
                        .item()
                    )
                    avg_atm_emb_norms = torch.norm(torch.stack(cur_new_phrase_atm_embs), dim=-1).mean().item()

                # cur_opt_step - 2 because we started at 1 and already incremented
                stats_per_opt_step[cur_opt_step - 2]["error"] += avg_cur_step_error
                stats_per_opt_step[cur_opt_step - 2]["gradnorm"] += avg_pre_clip_grad_norm
                stats_per_opt_step[cur_opt_step - 2]["embnorm"] += avg_atm_emb_norms
                stats_per_opt_step[cur_opt_step - 2]["diff_to_initial_cos"] += avg_cosine_sims_w_initial
                stats_per_opt_step[cur_opt_step - 2]["diff_to_inital_norm"] += avg_diff_norms_to_initial

                ema_error = avg_cur_step_error if ema_error is None else 0.5 * ema_error + 0.5 * avg_cur_step_error
                fwd_pass_bar.set_postfix(
                    {
                        "error": f"{avg_cur_step_error:.4f}",
                        "ema_error": f"{ema_error:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.7f}",
                        "data_fetch_time": f"{total_data_fetch_time:.4f}",
                        "gradnorm": f"{avg_pre_clip_grad_norm:.3f}",
                        "avg. embnorm": f"{avg_atm_emb_norms:.4f}",
                        "diff_to_initial": f"norm: {avg_diff_norms_to_initial:.4f} cos: {avg_cosine_sims_w_initial:.4f}",
                    }
                )
                fwd_pass_bar.set_description(f"Fwd passes (Step {cur_opt_step}/{NUM_OPT_STEPS})...")
                avg_cur_step_error = 0
    # avg_error_per_opt_step = [e / (NUM_PHRASES // MBS) for e in avg_error_per_opt_step]
    import matplotlib.pyplot as plt

    # do 5 different line plots for the stats_per_opt_step on top of each other
    # do it in a single fig with subplots, share x axis
    # make sure the y axis is trimmed to the range of the values
    fig, axs = plt.subplots(5, 1, sharex=True)
    for i, k in enumerate(stats_per_opt_step[0].keys()):
        axs[i].plot([s[k] for s in stats_per_opt_step])
        # axs[i].set_title(k)
        axs[i].set_ylabel(k)
        # trim y axis
        axs[i].set_ylim(min([s[k] for s in stats_per_opt_step]), max([s[k] for s in stats_per_opt_step]))

    # save the plot with tight layout
    plt.tight_layout()
    plt.savefig(f"stats_per_opt_step_nphrases{NUM_PHRASES}_nsnips{num_snippets_per_phrase}_nopt{NUM_OPT_STEPS}.png")
    return new_phrases_trainable_atm_embs


class BatchedSnippetDataset(torch.utils.data.IterableDataset):
    """
    Idea: we want to batch our forward apsses for efficiency.
    However, our input snippets for each new phrase are usually quite short, so we would need to use much larger bacth sizes than we want to (usually 1-10 is fine, but we need >100).

    Solution: We batch *along the phrase dimension* instead of batching multiple snippets for the same phrase.
    To get the same effect as batching snippets, we use gradient accumulation to achieve batch sizes larger than 1 for each phrase.

    Each batch therefore contains the nth snippet for each phrase in the batch. We iterate over batches ordered so that we exhaust all snippets for each "phrase group" before moving on to the next group.
    """

    def __init__(
        self,
        new_phrases_ids: list[list[int]],
        new_phrases_train_snippets: Iterator[list[list[typ.Int[torch.Tensor, "token_ids"]]]],
        # new_phrases_trainable_atm_embs: list[nn.Parameter],
        model_input_embedding: nn.Embedding,
        batch_size=10,
    ):
        self.new_phrases_ids = new_phrases_ids
        if not hasattr(new_phrases_train_snippets, "__next__"):
            new_phrases_train_snippets = iter(new_phrases_train_snippets)
        self.new_phrases_train_snippets = new_phrases_train_snippets
        # self.new_phrases_trainable_atm_embs = new_phrases_trainable_atm_embs
        self.atm_emb_placeholder = torch.zeros_like(model_input_embedding.weight[0])
        self.model_input_embedding = model_input_embedding
        assert model_input_embedding.weight.device.type == "cpu"
        self.batch_size = batch_size

    def num_snippets_per_phrase(self):
        return self._num_snippets_per_phrase

    @torch.no_grad()
    def _collate_phrase_snippets_into_batches(self):
        """
        Do the batch collation s.t. each batch contains the nth snippet for batch_size many phrases.
        """

        def chunk_generator(iterator, size):
            while True:
                chunk = list(itertools.islice(iterator, size))
                if chunk:
                    yield chunk
                else:
                    break

        # training_batch_inputs = []
        phrase_counter = 0

        # Get `self.batch_size` number of phrases and their snippets at a time
        for cur_phrases_train_snippets in chunk_generator(self.new_phrases_train_snippets, self.batch_size):
            cur_phrases_idxs = list(range(phrase_counter, phrase_counter + len(cur_phrases_train_snippets)))
            phrase_counter += len(cur_phrases_train_snippets)

            max_num_snippets = max([len(cur_phrase_snippets) for cur_phrase_snippets in cur_phrases_train_snippets])
            min_num_snippets = min([len(cur_phrase_snippets) for cur_phrase_snippets in cur_phrases_train_snippets])
            assert max_num_snippets == min_num_snippets
            self._num_snippets_per_phrase = max_num_snippets

            # Iterate over their snippets in parallel and collate s.t. each batch has the snippets at `snippet_idx` from each phrase
            for snippet_idx in range(max_num_snippets):
                cur_batch = {
                    "snippets_toks": torch.stack(
                        [phrase_snippets_toks[snippet_idx] for phrase_snippets_toks in cur_phrases_train_snippets]
                    ),
                    "phrases_idxs": cur_phrases_idxs,
                    "phrases_toks": [self.new_phrases_ids[pi] for pi in cur_phrases_idxs],
                    "snippet_idx": snippet_idx,
                }
                yield cur_batch

    @torch.no_grad()
    def __iter__(self):
        # Need to handle the multiple workers case
        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id

        _iter = self._training_batch_iter()
        workerified_iter = itertools.islice(_iter, worker_id, None, worker_total_num)
        return workerified_iter

    @torch.no_grad()
    def _training_batch_iter(self):
        """
        Additionally to the token input ids, we need to provide embedding sequbces for the ATM forward pass.
        """

        for i, batch in enumerate(self._collate_phrase_snippets_into_batches()):
            phrases_idxs = batch["phrases_idxs"]
            batched_new_phrase_snippet_token_ids = batch["snippets_toks"]

            batched_snippet_embs_w_atm = []
            batched_phrase_in_atm_seq_idxs = []
            batched_phrase_in_gt_seq_idxs = []
            for j in range(len(phrases_idxs)):
                new_phrase_tokenized_ids = batch["phrases_toks"][j]
                new_phrase_atm_emb = torch.full_like(self.atm_emb_placeholder, -100)
                new_phrase_snippet_token_ids = batched_new_phrase_snippet_token_ids[j]
                (
                    sample_embs_w_merged_pattern,
                    patterns_in_merged_seq_idx,
                    patterns_in_gt_seq_idx,
                ) = construct_merged_emb_inputs(
                    new_phrase_snippet_token_ids,
                    new_phrase_tokenized_ids,
                    new_phrase_atm_emb,
                    self.model_input_embedding,
                )
                batched_snippet_embs_w_atm.append(sample_embs_w_merged_pattern)
                batched_phrase_in_atm_seq_idxs.append(patterns_in_merged_seq_idx)
                batched_phrase_in_gt_seq_idxs.append(patterns_in_gt_seq_idx)

            padded_batched_atm_seqs_embs = pad_sequence(
                batched_snippet_embs_w_atm,
                batch_first=True,
                padding_value=-1,
            )
            lengths = torch.as_tensor([v.size(0) for v in batched_snippet_embs_w_atm])
            batch["padded_atm_seqs_embs"] = padded_batched_atm_seqs_embs
            batch["padded_atm_seqs_embs_orig_lengths"] = lengths
            batch["new_phrase_in_gt_seq_idxs"] = batched_phrase_in_gt_seq_idxs
            batch["new_phrase_in_atm_seq_idxs"] = batched_phrase_in_atm_seq_idxs

            yield batch


def construct_merged_emb_inputs(sample_token_ids, pattern, pattern_emb, input_embs):
    sample_embs_w_merged_pattern = []
    patterns_in_merged_seq_idx = []
    patterns_in_gt_seq_idx = []
    cur_tok_id_idx = 0
    while cur_tok_id_idx < (len(sample_token_ids)):
        if torch.equal(sample_token_ids[cur_tok_id_idx : cur_tok_id_idx + pattern.size(0)], pattern):
            sample_embs_w_merged_pattern.append(pattern_emb.to(input_embs.weight.dtype))
            patterns_in_gt_seq_idx.append(cur_tok_id_idx)
            patterns_in_merged_seq_idx.append(len(sample_embs_w_merged_pattern) - 1)
            cur_tok_id_idx += pattern.size(0)
        else:
            emb = input_embs(sample_token_ids[cur_tok_id_idx])
            sample_embs_w_merged_pattern.append(emb)
            cur_tok_id_idx += 1
    sample_embs_w_merged_pattern = torch.stack(sample_embs_w_merged_pattern)
    return (
        sample_embs_w_merged_pattern,
        patterns_in_merged_seq_idx,
        patterns_in_gt_seq_idx,
    )


def align_hidden_states_w_merged_pattern(
    gt_hiddens,
    atm_hiddens,
    patterns_in_gt_seq_idx,
    patterns_in_merged_seq_idx,
    new_phrase_ids_len,
):
    full_pattern_indices_in_gt = []
    for idx in patterns_in_gt_seq_idx:
        full_pattern_indices_in_gt.extend(range(idx, idx + new_phrase_ids_len))
    last_pattern_indices_in_gt = []
    for idx in patterns_in_gt_seq_idx:
        last_pattern_indices_in_gt.append(idx + new_phrase_ids_len - 1)
    pattern_indices_in_merged = patterns_in_merged_seq_idx

    non_pattern_hiddens_gt = gt_hiddens[[i for i in range(gt_hiddens.size(0)) if i not in full_pattern_indices_in_gt]]
    non_pattern_hiddens_merged = atm_hiddens[[i for i in range(atm_hiddens.size(0)) if i not in pattern_indices_in_merged]]

    pattern_last_hiddens_gt = gt_hiddens[last_pattern_indices_in_gt]
    pattern_hiddens_merged = atm_hiddens[pattern_indices_in_merged]

    return (
        non_pattern_hiddens_gt,
        non_pattern_hiddens_merged,
        pattern_last_hiddens_gt,
        pattern_hiddens_merged,
    )
