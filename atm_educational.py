import copy
from os import truncate
from typing import Iterable

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch import cosine_similarity, nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from atm_utils import SPIECE_WHITESPACE

LR = 0.05
WD = 0.0
BS = 5
MBS = 5
NUM_TRAIN_SNIPPETS = 50
NUM_ESTIMATE_PASSES = 50
NUM_ESTIMATE_PASSES = min(NUM_ESTIMATE_PASSES, NUM_TRAIN_SNIPPETS)
SNIPPET_MAX_LENGTH = 100
# TOKEN_STRING = "▁Jake▁Gyllenhaal"
NEW_PHRASE_STR = " technical debt"
# NEW_PHRASE_STR = " integer underflow"
NEW_PHRASE_STR = " thermomechanical material fatigue resistance testing"
NEW_PHRASE_STR = " hydrogen presence in the intergalactic medium"
NEW_PHRASE_STR = " is not"
# NEW_PHRASE_STR = " reionization"

TARGET_LAYER = -1

# DATA_GENERATION_PROMPT_TEMPLATE = "She{NEW_PHRASE}"
DATA_GENERATION_PROMPT_PREFIX = ""
USE_SPECIAL_TOKENS = True
CONCAT_EVERY = -1


def get_new_phrase_tokenized_ids(new_phrase, tokenizer: PreTrainedTokenizer):
    """
    Once your stare into the pits of hell for too long, you start to become the pits of hell.
    In this case the pits of hell are the incredibly convoluted and confusing tokenization logic of the Huggingface transformers library and I have to resort to this absolute mess to fix it.
    Specifically, the handling of prefix whitespaces in tokens is absolutely broken. It is very difficiult to reliably tokenize a span of text that doesn't start with a whitespace.
    `add_prefix_space=False` is supposed to handle this, but it doesn't work as expected.
    We use the following strategy to reliably split a span of text (e.g. `new_phrase`) into tokens and correctly handle the prefix whitespace:
    (1) We add the BOS token to the beginning of the text.
    (2) We tokenize the text with the tokenizer.
    (3) We remove the BOS token id from the output.
    """
    # yes, it's even difficult to get the actual whitespace token
    whitespace_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(f"{tokenizer.bos_token} ", add_special_tokens=False))[1]
    new_phrase_starts_with_whitespace = new_phrase.startswith(" ") or new_phrase.startswith(whitespace_token)

    # if eval:
    #     test_prefix_whitespace_when_adding_token(new_phrase, tokenizer)

    # 1.1) "Naive" token merging via mean as initial guess, first get the correct tokenization of new_phrase
    new_phrase_w_bos = f"{tokenizer.bos_token}{new_phrase}"
    new_phrase_tokenized_ids = tokenizer.encode(new_phrase_w_bos, return_tensors="pt", add_special_tokens=False)[0][1:]
    # print(
    #     f"New phrase string: <{new_phrase}> | New phrase ids: {new_phrase_tokenized_ids} | new phrase tokens: {tokenizer.convert_ids_to_tokens(new_phrase_tokenized_ids)} | New phrase tokenized: {tokenizer.tokenize(new_phrase)} | New phrase tokenized decoded: <{tokenizer.decode(new_phrase_tokenized_ids)}>"
    # )
    # sanity checks
    assert new_phrase_tokenized_ids[0] != tokenizer.bos_token_id
    if new_phrase_starts_with_whitespace:
        assert tokenizer.convert_ids_to_tokens(new_phrase_tokenized_ids[0].item()).startswith(whitespace_token)
    else:
        assert not tokenizer.convert_ids_to_tokens(new_phrase_tokenized_ids[0].item()).startswith(whitespace_token)

    return new_phrase_tokenized_ids, new_phrase_starts_with_whitespace


def atm_merge_token(new_phrase, tokenizer: PreTrainedTokenizer, model, train_snippets=None, eval=True):
    """
    Educational / eval / debug implementation of ATM for a single token.
    A more optimized version of ATM that batches the forward passes for multiple tokens is implemented in `batch_atm_for_many_tokens` but does not include the eval functionality for efficiency.
    """
    tokenizer = copy.deepcopy(tokenizer)

    # 1) Get the initial embedding for the new token
    new_phrase_tokenized_ids, new_phrase_starts_with_whitespace = get_new_phrase_tokenized_ids(new_phrase, tokenizer)
    new_phrase_atm_emb = torch.mean(model.get_input_embeddings()(new_phrase_tokenized_ids), dim=0).to(torch.float32)
    initial_new_phrase_merged_emb = new_phrase_atm_emb.clone().detach()

    # 2) generate snippets with the new token from the model
    data_generation_prompt_pattern_ids = []
    if USE_SPECIAL_TOKENS:
        data_generation_prompt_pattern_ids.append(tokenizer.bos_token_id)
    if DATA_GENERATION_PROMPT_PREFIX:
        data_generation_prompt_pattern_ids += get_new_phrase_tokenized_ids(DATA_GENERATION_PROMPT_PREFIX, tokenizer)[0].tolist()
    data_generation_prompt_pattern_ids += new_phrase_tokenized_ids.tolist()
    print("-----Sanity prints-----")
    print(
        f"Prompting the model with: {tokenizer.decode(data_generation_prompt_pattern_ids)} | {tokenizer.convert_ids_to_tokens(data_generation_prompt_pattern_ids)}"
    )

    if train_snippets is None:
        new_phrase_atm_train_snippets = generate_samples_with_pattern(
            model,
            tokenizer,
            data_generation_prompt_pattern_ids,
            NUM_TRAIN_SNIPPETS,
            seed=1234,
            concat_every=CONCAT_EVERY,
            max_length=SNIPPET_MAX_LENGTH,
        )
    else:
        new_phrase_atm_train_snippets = train_snippets

    if eval:
        # 3.pre) Initial eval on the train snippets
        do_error_estimation_passes(
            model,
            new_phrase_atm_train_snippets,
            new_phrase_tokenized_ids,
            new_phrase_atm_emb,
            NUM_ESTIMATE_PASSES,
            tokenizer,
            "Initial Mean Emb",
        )

    # 3) ATM optimization
    print(f"Snippet sample tokens: {tokenizer.convert_ids_to_tokens(new_phrase_atm_train_snippets[0]['input_ids'][0])}")
    print(f"Snippet sample tokens 2: {tokenizer.convert_ids_to_tokens(new_phrase_atm_train_snippets[1]['input_ids'][0])}")
    print(f"Snippet sample tokens 3: {tokenizer.convert_ids_to_tokens(new_phrase_atm_train_snippets[2]['input_ids'][0])}")
    print(f"Snippet sample tokens 4: {tokenizer.convert_ids_to_tokens(new_phrase_atm_train_snippets[3]['input_ids'][0])}")

    new_phrase_atm_emb = atm_optimization(
        new_phrase_atm_emb,
        model,
        new_phrase_atm_train_snippets,
        new_phrase_tokenized_ids,
        tokenizer,
        new_phrase_string=new_phrase,
    )

    if eval:
        # 4.1) Eval, first on the train snippets...
        do_error_estimation_passes(
            model,
            new_phrase_atm_train_snippets,
            new_phrase_tokenized_ids,
            new_phrase_atm_emb,
            NUM_ESTIMATE_PASSES,
            tokenizer,
            "Train Snippets",
        )

        # 4.2) ... then on newly generated test snippets
        new_phrase_test_snippets = generate_samples_with_pattern(
            model,
            tokenizer,
            data_generation_prompt_pattern_ids,
            NUM_ESTIMATE_PASSES,
            seed=123456,
            concat_every=-1,
            max_length=SNIPPET_MAX_LENGTH,
        )
        do_error_estimation_passes(
            model,
            new_phrase_test_snippets,
            new_phrase_tokenized_ids,
            new_phrase_atm_emb,
            NUM_ESTIMATE_PASSES,
            tokenizer,
            "Test Snippets",
        )

        def print_eval_generation_results(
            eval_generation_prompt,
            model,
            tokenizer,
            num_samples=5,
            max_length=50,
            desc="",
        ):
            prompt_ids = tokenizer(
                eval_generation_prompt,
                add_special_tokens=USE_SPECIAL_TOKENS,
                return_tensors="pt",
            )["input_ids"][0]
            data_samples_tokens = generate_samples_with_pattern(
                model,
                tokenizer,
                prompt_ids,
                num_samples=num_samples,
                max_length=max_length,
            )
            texts = [tokenizer.decode(i["input_ids"][0], skip_special_tokens=True) for i in data_samples_tokens]
            print()
            print("|||||||||||||||||||||||||||||||||||||||")
            print(f"----------Eval Generation: {desc}----------")
            print(
                f"--> Prompt: {eval_generation_prompt} | tokenized ids: {prompt_ids} | tokenized: {tokenizer.convert_ids_to_tokens(prompt_ids)}"
            )
            for text in texts:
                print("-------------------------")
                print(text)
            print(f"-------End Eval Generation: {desc} ---------")
            print()

        # 5.1) generate samples with the new token.. but first a baseline
        eval_new_phrase = new_phrase if new_phrase_starts_with_whitespace else f" {new_phrase}"
        generation_eval_prompt = f"Barack Obama{eval_new_phrase}"
        print_eval_generation_results(generation_eval_prompt, model, tokenizer, 3, 50, "Original Tokenization")

        # 5.2) generate samples with the new token but using the initial merged embedding
        tokenizer.add_tokens(new_phrase)
        new_phrase_merged_token_id = tokenizer.convert_tokens_to_ids(new_phrase)
        print(f"New phrase merged token id: {new_phrase_merged_token_id}")
        model.resize_token_embeddings(len(tokenizer))
        model.get_output_embeddings().weight[new_phrase_merged_token_id] = torch.zeros_like(
            model.get_output_embeddings().weight[0]
        )
        model.get_input_embeddings().weight[new_phrase_merged_token_id] = initial_new_phrase_merged_emb
        print_eval_generation_results(generation_eval_prompt, model, tokenizer, 3, 50, "Initial Merged Emb")

        # 5.3) generate samples with the new token but using the optimized merged embedding
        model.get_input_embeddings().weight[new_phrase_merged_token_id] = new_phrase_atm_emb
        print_eval_generation_results(generation_eval_prompt, model, tokenizer, 3, 50, "Optimized Merged Emb")

    return new_phrase_atm_emb


def atm_optimization(
    new_phrase_atm_emb,
    model: PreTrainedModel,
    new_phrase_snippets,
    new_phrase_tokenized_ids,
    tokenizer,
    new_phrase_string="",
):
    initial_new_phrase_atm_emb = new_phrase_atm_emb.clone().detach()
    new_phrase_atm_emb = nn.Parameter(new_phrase_atm_emb).requires_grad_(True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW([new_phrase_atm_emb], lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=int((NUM_TRAIN_SNIPPETS // BS) * 0.1)
    )

    forward_passes = do_forward_passes_merged_vs_normal(
        model,
        new_phrase_snippets,
        new_phrase_tokenized_ids,
        new_phrase_atm_emb,
        tokenizer,
        micro_batch_size=MBS,
    )
    bar = tqdm(
        forward_passes,
        total=len(new_phrase_snippets) // MBS,
        desc=f"ATM for {new_phrase_string}...",
    )

    avg_error_for_step = 0
    ema_error = None
    # avg_error_hist = []
    loss_fn = torch.nn.MSELoss(reduction="mean")

    snippet_counter = 0
    snippets_per_batch = MBS
    for _, new_phrase_snippet_forward_pass in enumerate(bar):
        (
            non_pattern_hiddens_gt,
            non_pattern_hiddens_merged,
            pattern_last_hiddens_gt,
            pattern_hiddens_merged,
        ) = new_phrase_snippet_forward_pass

        all_hiddens_gt = torch.cat([non_pattern_hiddens_gt, pattern_last_hiddens_gt])
        all_hiddens_merged = torch.cat([non_pattern_hiddens_merged, pattern_hiddens_merged])
        error = loss_fn(all_hiddens_merged, all_hiddens_gt) / BS

        # non_pattern_loss = loss_fn(non_pattern_hiddens_merged, non_pattern_hiddens_gt)
        # pattern_loss = loss_fn(pattern_hiddens_merged, pattern_last_hiddens_gt)

        # error = (non_pattern_loss + pattern_loss) / BS
        # print(new_phrase_atm_emb, new_phrase_atm_emb.requires_grad, new_phrase_atm_emb.grad)
        # if torch.isnan(error):
        #     print("NaN error, breaking")
        # if pattern_hiddens_merged.size(0) == 0:
        #     print("Empty pattern hiddens merged, breaking")
        #     cur_sample = new_phrase_snippets[i]
        #     print(cur_sample["input_ids"])
        # if pattern_last_hiddens_gt.size(0) == 0:
        #     print("Empty pattern hiddens gt, breaking")
        error.backward(inputs=[new_phrase_atm_emb])
        avg_error_for_step += error.item()

        del (
            all_hiddens_gt,
            all_hiddens_merged,
            non_pattern_hiddens_gt,
            non_pattern_hiddens_merged,
            pattern_last_hiddens_gt,
            pattern_hiddens_merged,
            new_phrase_snippet_forward_pass,
        )

        snippet_counter += snippets_per_batch

        # Optimizer Step
        if snippets_per_batch % BS == 0:
            pre_clip_grad_norm = torch.norm(new_phrase_atm_emb.grad).item()
            torch.nn.utils.clip_grad_norm_(new_phrase_atm_emb, 1.0)
            optimizer.step()
            with torch.no_grad():
                diff_norm_to_initial = torch.norm(new_phrase_atm_emb - initial_new_phrase_atm_emb).item()
                cosine_sim_w_initial = cosine_similarity(
                    new_phrase_atm_emb.unsqueeze(0),
                    initial_new_phrase_atm_emb.unsqueeze(0),
                ).item()
            # avg_error_hist.append(avg_error_for_step)
            ema_error = avg_error_for_step if ema_error is None else 0.5 * ema_error + 0.5 * avg_error_for_step
            bar.set_postfix(
                {
                    "error": f"{avg_error_for_step:.4f}",
                    "ema_error": f"{ema_error:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.7f}",
                    "optstep": f"{snippets_per_batch % BS}",
                    "gradnorm": f"{pre_clip_grad_norm:.3f}",
                    "embnorm": f"{torch.norm(new_phrase_atm_emb).item():.4f}",
                    "diff_to_initial": f"norm: {diff_norm_to_initial:.4f} cos: {cosine_sim_w_initial:.4f}",
                }
            )

            avg_error_for_step = 0
            optimizer.zero_grad()
            scheduler.step()
        # plot error hist and savefig
    # import matplotlib.pyplot as plt

    # plt.plot(avg_error_hist)
    # plt.savefig(
    #     f"error_hist_{model.name_or_path.replace('/', '_')}_<{new_phrase_string.replace('/', '_')}>_ntrain{NUM_TRAIN_SNIPPETS}_bs{BS}_lr{LR}_layer{TARGET_LAYER}_{'use_special' if USE_SPECIAL_TOKENS else 'no_use_special'}.png"
    # )
    return new_phrase_atm_emb


def pad_collate_fn(batch, padding_value):
    input_ids = [item["input_ids"][0] for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
    return {"input_ids": padded_input_ids}


class SnippetDataset(torch.utils.data.Dataset):
    def __init__(self, snippets):
        self.snippets = snippets

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        return self.snippets[idx]


def do_forward_passes_merged_vs_normal(
    model,
    data_samples_tokens,
    new_phrase_tokenized_ids,
    new_phrase_atm_emb,
    tokenizer,
    micro_batch_size=None,
):
    if micro_batch_size is None:
        micro_batch_size = MBS
    # assert len(data_samples_tokens) % micro_batch_size == 0
    # batched_iterator = iter(
    #     [data_samples_tokens[i : i + micro_batch_size] for i in range(0, len(data_samples_tokens), micro_batch_size)]
    # )
    dataset = SnippetDataset(data_samples_tokens)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        # collate_fn=partial(pad_collate_fn, padding_value=tokenizer.pad_token_id or tokenizer.eos_token_id),
    )

    for i, batch in enumerate(dataloader):
        batch_input_ids = batch["input_ids"].squeeze(0).squeeze(1)
        with torch.no_grad():
            # ground truth forward pass
            out = model(batch_input_ids, output_hidden_states=True, return_dict=True)
            batched_gt_hiddens_after_first_layer = out["hidden_states"][TARGET_LAYER]
            # .view(               -1, out["hidden_states"][TARGET_LAYER].shape[-1]            )
            del out

            # gt_hiddens_after_first_layer = out["logits"].view(-1, out["logits"].shape[-1])

        # forward pass with the merged pattern
        # (1) construct sequence of input embeddings with the pattern replaced by the new merged embedding
        batched_sample_token_ids = batch_input_ids
        batched_sample_embs_w_merged_pattern = []
        batched_patterns_in_merged_seq_idx = []
        batched_patterns_in_gt_seq_idx = []
        for sample_token_ids in batched_sample_token_ids:
            (
                sample_embs_w_merged_pattern,
                patterns_in_merged_seq_idx,
                patterns_in_gt_seq_idx,
            ) = construct_merged_emb_inputs(
                sample_token_ids,
                new_phrase_tokenized_ids,
                new_phrase_atm_emb,
                model.get_input_embeddings(),
            )
            batched_sample_embs_w_merged_pattern.append(sample_embs_w_merged_pattern.squeeze(0))
            batched_patterns_in_merged_seq_idx.append(patterns_in_merged_seq_idx)
            batched_patterns_in_gt_seq_idx.append(patterns_in_gt_seq_idx)

        padded_batched_sample_embs_w_merged_pattern = pad_sequence(
            batched_sample_embs_w_merged_pattern,
            batch_first=True,
            padding_value=-1,
        )
        lengths = torch.as_tensor([v.size(0) for v in batched_sample_embs_w_merged_pattern])

        # (2) forward pass with the new input embeddings
        out = model(
            inputs_embeds=padded_batched_sample_embs_w_merged_pattern,
            output_hidden_states=True,
            return_dict=True,
        )
        batched_merged_hiddens_after_first_layer = out["hidden_states"][TARGET_LAYER]
        # .view(
        #     -1, out["hidden_states"][TARGET_LAYER].shape[-1]
        # )
        del out
        # merged_hiddens_after_first_layer = out["logits"].view(-1, out["logits"].shape[-1])

        # remove paddings
        batched_merged_hiddens_after_first_layer = [
            t[: lengths[i], :] for i, t in enumerate(batched_merged_hiddens_after_first_layer)
        ]

        # (3) match the two sequences
        batched_non_pattern_hiddens_gt = []
        batched_non_pattern_hiddens_merged = []
        batched_pattern_last_hiddens_gt = []
        batched_pattern_hiddens_merged = []
        for (
            gt_hiddens_after_first_layer,
            merged_hiddens_after_first_layer,
            patterns_in_gt_seq_idx,
            patterns_in_merged_seq_idx,
        ) in zip(
            batched_gt_hiddens_after_first_layer,
            batched_merged_hiddens_after_first_layer,
            batched_patterns_in_gt_seq_idx,
            batched_patterns_in_merged_seq_idx,
        ):
            # unpadded_merged_hiddens_after_first_layer = merged_hiddens_after_first_layer[

            (
                non_pattern_hiddens_gt,
                non_pattern_hiddens_merged,
                pattern_last_hiddens_gt,
                pattern_hiddens_merged,
            ) = align_hidden_states_w_merged_pattern(
                gt_hiddens_after_first_layer,
                merged_hiddens_after_first_layer,
                patterns_in_gt_seq_idx,
                patterns_in_merged_seq_idx,
                new_phrase_ids_len=new_phrase_tokenized_ids.size(0),
            )
            batched_non_pattern_hiddens_gt.append(non_pattern_hiddens_gt)
            batched_non_pattern_hiddens_merged.append(non_pattern_hiddens_merged)
            batched_pattern_last_hiddens_gt.append(pattern_last_hiddens_gt)
            batched_pattern_hiddens_merged.append(pattern_hiddens_merged)

        yield (
            torch.cat(batched_non_pattern_hiddens_gt),
            torch.cat(batched_non_pattern_hiddens_merged),
            torch.cat(batched_pattern_last_hiddens_gt),
            torch.cat(batched_pattern_hiddens_merged),
        )


@torch.no_grad
def do_error_estimation_passes(
    model,
    new_phrase_snippets_toks,
    new_phrase_tokenized_ids,
    new_phrase_atm_emb,
    num_passes,
    tokenizer,
    desc="",
):
    error_estimate = 0
    pattern_error_estimate = 0
    num_extra_pos = new_phrase_snippets_toks[0]["input_ids"].size(1) - new_phrase_tokenized_ids.size(0)

    # print("num_extra_pos", num_extra_pos)
    # print("new_phrase_snippets_toks", len(new_phrase_snippets_toks))
    # print("new_phrase_tokenized_ids", new_phrase_tokenized_ids.size())
    # # print("num_extra_pos", num_extra_pos)
    # per_pos_error = torch.zeros(2 + num_extra_pos)

    per_position_non_pattern_cos_sim = torch.zeros(num_extra_pos)
    estimate_passes = do_forward_passes_merged_vs_normal(
        model,
        new_phrase_snippets_toks[:num_passes],
        new_phrase_tokenized_ids,
        new_phrase_atm_emb,
        tokenizer,
        micro_batch_size=1,
    )
    pattern_cosine_similarity = 0
    non_pattern_cosine_similarity = 0
    per_pos_cos_sim_counter = 0
    for i, pass_results in enumerate(tqdm(estimate_passes, desc=f"Eval: {desc}...", leave=False, total=num_passes)):
        (
            non_pattern_hiddens_gt,
            non_pattern_hiddens_merged,
            pattern_last_hiddens_gt,
            pattern_hiddens_merged,
        ) = pass_results

        # print(
        #     non_pattern_hiddens_gt.size(),
        #     non_pattern_hiddens_merged.size(),
        #     pattern_last_hiddens_gt.size(),
        #     pattern_hiddens_merged.size(),
        # )
        non_pattern_error = torch.norm(non_pattern_hiddens_gt - non_pattern_hiddens_merged, dim=1)
        pattern_error = torch.norm(pattern_last_hiddens_gt - pattern_hiddens_merged, dim=1)

        pattern_cosine_similarity += torch.cosine_similarity(pattern_last_hiddens_gt, pattern_hiddens_merged).mean().item()
        non_pattern_cosine_similarity += (
            torch.cosine_similarity(non_pattern_hiddens_gt, non_pattern_hiddens_merged).mean().item()
        )
        if non_pattern_hiddens_merged.size(0) == num_extra_pos:
            # print(
            #     per_position_non_pattern_cos_sim.shape,
            #     torch.cosine_similarity(non_pattern_hiddens_gt, non_pattern_hiddens_merged, dim=1).shape,
            # )
            # print(non_pattern_hiddens_gt[:2], non_pattern_hiddens_merged[:2])
            per_position_non_pattern_cos_sim += torch.cosine_similarity(
                non_pattern_hiddens_gt, non_pattern_hiddens_merged, dim=1
            )
            per_pos_cos_sim_counter += 1
        error_estimate += non_pattern_error.mean().item()
        pattern_error_estimate += pattern_error.mean().item()

    print()
    print(f"--------Eval Summary: {desc}--------")
    print(f"--> Num Passes: {num_passes}")
    print(
        f"--> Snippet length: {new_phrase_snippets_toks[0]['input_ids'].size(1)} | Pattern length: {new_phrase_tokenized_ids.size(0)} | BOS token: {new_phrase_snippets_toks[0]['input_ids'][0][0]}"
    )  # , example snippet: {new_phrase_snippets_toks[0]['input_ids']}")
    print(f"--> Pattern Cosine Similarity: {pattern_cosine_similarity / num_passes}")
    print(f"--> Non-Pattern Cosine Similarity: {non_pattern_cosine_similarity / num_passes}")
    print(f"--> Pattern Norm Diff: {pattern_error_estimate / num_passes}")
    print(f"--> Non-Pattern Norm Diff: {error_estimate / num_passes}")
    print(f"--> Per-Position non-pattern Cosine Sims: {per_position_non_pattern_cos_sim / per_pos_cos_sim_counter}")
    print(f"-----------------------------------")
    print()

    return (
        error_estimate / num_passes,
        pattern_error_estimate / num_passes,
        per_position_non_pattern_cos_sim / per_pos_cos_sim_counter,
        pattern_cosine_similarity / num_passes,
        non_pattern_cosine_similarity / num_passes,
    )


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


def generate_samples_with_pattern(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    pattern: Iterable[int],
    num_samples: int,
    seed=42,
    max_length=15,
    concat_every=-1,
    use_force_words=False,
):
    desired_samples = num_samples
    data = []
    # print(f"generating {desired_samples} samples with max new tokens {max_length}")

    bar = tqdm(total=desired_samples, leave=False, desc=f"Generating snippets use_force_words={use_force_words}...")
    # set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    word_ids = torch.tensor([*pattern], device=model.device).unsqueeze(0)
    pattern_starts_with_bos = pattern[0] == tokenizer.bos_token_id
    if pattern_starts_with_bos and use_force_words:
        word_ids = word_ids[:, 1:]
    while len(data) < desired_samples:
        if use_force_words:
            random_prefix = model.generate(
                torch.tensor([tokenizer.bos_token_id], device=model.device).unsqueeze(0),
                do_sample=True,
                max_length=5,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )

            # if
        new_data = model.generate(
            random_prefix if use_force_words else word_ids,
            do_sample=not use_force_words,
            num_beams=32,
            max_new_tokens=max_length,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            # top_k=50,
            num_return_sequences=min(4 if not use_force_words else 1000, desired_samples)
            if concat_every == -1
            else min(concat_every * 100, desired_samples),
            force_words_ids=word_ids.tolist() if use_force_words else None,
            # no_repeat_ngram_size=1,
            # remove_invalid_values=True,
            # num_return_sequences=4,
        )
        if concat_every != -1:
            # concat every n samples into a single sample
            new_data = [
                torch.cat([new_data[i] for i in range(j, j + concat_every)]) for j in range(0, len(new_data), concat_every)
            ]
        data.extend(new_data)
        bar.update(len(new_data))
    # print(len(data))
    return [{"input_ids": i.unsqueeze(0)} for i in data]


def test_prefix_whitespace_when_adding_token(new_phrase, tokenizer: PreTrainedTokenizer):
    tokenizer = copy.deepcopy(tokenizer)
    new_phrase_starts_with_whitespace = new_phrase.startswith(" ") or new_phrase.startswith(SPIECE_WHITESPACE)
    if new_phrase_starts_with_whitespace:
        new_phrase_w_whitespace = new_phrase
        new_phrase_no_whitespace = new_phrase[1:]
    else:
        new_phrase_w_whitespace = f" {new_phrase}"
        new_phrase_no_whitespace = new_phrase

    print(f"New phrase string: {new_phrase} | starts with whitespace: {new_phrase_starts_with_whitespace}")
    new_phrase_ws_tokenized = tokenizer.tokenize(new_phrase_w_whitespace)
    new_phrase_no_ws_tokenized = tokenizer.tokenize(new_phrase_no_whitespace)
    print(f"Tokenized with whitespace: {new_phrase_ws_tokenized} | Tokenized without whitespace: {new_phrase_no_ws_tokenized}")
    print(
        f"Decoded with whitespace: {tokenizer.decode(tokenizer.encode(new_phrase_w_whitespace))} | Decoded without whitespace: {tokenizer.decode(tokenizer.encode(new_phrase_no_whitespace))}"
    )

    new_phrase_bos_ws_tokenized = tokenizer.tokenize(f"{tokenizer.bos_token}{new_phrase_w_whitespace}")
    new_phrase_bos_no_ws_tokenized = tokenizer.tokenize(f"{tokenizer.bos_token}{new_phrase_no_whitespace}")
    print(
        f"BOS Tokenized with whitespace: {new_phrase_bos_ws_tokenized} | BOS Tokenized without whitespace: {new_phrase_bos_no_ws_tokenized}"
    )
    print(
        f"BOS Decoded with whitespace: {tokenizer.decode(tokenizer.encode(f'{tokenizer.bos_token}{new_phrase_w_whitespace}'))} | BOS Decoded without whitespace: {tokenizer.decode(tokenizer.encode(f'{tokenizer.bos_token}{new_phrase_no_whitespace}'))}"
    )

    # add new phrase to tokenizer
    print(f"-------Adding new phrase with whitespace: {new_phrase}---------")
    tokenizer.add_tokens(new_phrase)

    new_phrase_ws_tokenized = tokenizer.tokenize(new_phrase_w_whitespace)
    new_phrase_no_ws_tokenized = tokenizer.tokenize(new_phrase_no_whitespace)
    print(f"Tokenized with whitespace: {new_phrase_ws_tokenized} | Tokenized without whitespace: {new_phrase_no_ws_tokenized}")
    print(
        f"Decoded with whitespace: {tokenizer.decode(tokenizer.encode(new_phrase_w_whitespace))} | Decoded without whitespace: {tokenizer.decode(tokenizer.encode(new_phrase_no_whitespace))}"
    )

    new_phrase_bos_ws_tokenized = tokenizer.tokenize(f"{tokenizer.bos_token}{new_phrase_w_whitespace}")
    new_phrase_bos_no_ws_tokenized = tokenizer.tokenize(f"{tokenizer.bos_token}{new_phrase_no_whitespace}")
    print(
        f"BOS Tokenized with whitespace: {new_phrase_bos_ws_tokenized} | BOS Tokenized without whitespace: {new_phrase_bos_no_ws_tokenized}"
    )
    print(
        f"BOS Decoded with whitespace: {tokenizer.decode(tokenizer.encode(f'{tokenizer.bos_token}{new_phrase_w_whitespace}'))} | BOS Decoded without whitespace: {tokenizer.decode(tokenizer.encode(f'{tokenizer.bos_token}{new_phrase_no_whitespace}'))}"
    )


def main():
    torch.set_default_device("cuda:0")
    MODEL = "mistralai/Mistral-7B-v0.1"
    # MODEL = "meta-llama/Llama-2-7b-hf"
    # MODEL = "EleutherAI/pythia-2.7B"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    # model.compile()
    # NEW_PHRASE_STR = "uring"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, legacy=False, add_prefix_space=False
    )  # , legacy=False, use_fast=False, from_slow=True)
    test_prefix_whitespace_when_adding_token(NEW_PHRASE_STR, tokenizer)
    atm_merge_token(NEW_PHRASE_STR, tokenizer, model)


if __name__ == "__main__":
    main()
