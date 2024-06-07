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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)

LR = 0.05
WD = 0.0
BS = 5
MBS = 200
NUM_TRAIN_SNIPPETS = 50
NUM_ESTIMATE_PASSES = 50
NUM_ESTIMATE_PASSES = min(NUM_ESTIMATE_PASSES, NUM_TRAIN_SNIPPETS)
SNIPPET_MAX_LENGTH = 100
# TOKEN_STRING = "▁Jake▁Gyllenhaal"
NEW_PHRASE_STR = " technical debt"
# NEW_PHRASE_STR = " integer underflow"
NEW_PHRASE_STR = " thermomechanical material fatigue resistance testing"
NEW_PHRASE_STR = " hydrogen presence in the intergalactic medium"
# NEW_PHRASE_STR = " reionization"

TARGET_LAYER = -1

DATA_GENERATION_PROMPT_TEMPLATE = "{NEW_PHRASE}"
USE_SPECIAL_TOKENS = True
CONCAT_EVERY = -1
SPIECE_UNDERLINE = "▁"


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


def chunk_list(lst, n):
    # Split the list into chunks of size n
    return (lst[i : i + n] for i in range(0, len(lst), n))


def create_trainable_atm_emb(new_phrase_ids, model_input_embedding: nn.Embedding, device=None):
    initial_atm_emb = torch.mean(model_input_embedding(new_phrase_ids), dim=0)  # .to(torch.float32)
    if device is not None:
        initial_atm_emb = initial_atm_emb.to(device)
    return nn.Parameter(initial_atm_emb).requires_grad_(True)


class BatchedSnippetDataset(torch.utils.data.IterableDataset):
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

        # self.training_batch_inputs = self._initial_collate_phrases_snippets()

    # def __len__(self):
    #     return len(self.training_batch_inputs)

    # def _num_batches(self):
    #     return len(self.training_batch_inputs)

    def num_snippets_per_phrase(self):
        return self._num_snippets_per_phrase

    # def num_phrases(self):
    #     return self._num_phrases

    @torch.no_grad()
    def _collate_phrase_snippets_into_batches(self):
        def chunk_generator(iterator, size):
            # I prefer the argument order to be the reverse of yours.
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
        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id

        _iter = self._training_batch_iter()
        workerified_iter = itertools.islice(_iter, worker_id, None, worker_total_num)
        return workerified_iter

    @torch.no_grad()
    def _training_batch_iter(self):
        # We want to batch the forward passes for multiple new_phrases. In each batch, we stack inputs for N new_phrases. We use grad accumulation to set the actual batch size for each new phrases.

        # model_input_embbeding = self.model.get_input_embeddings()
        for i, batch in enumerate(self._collate_phrase_snippets_into_batches()):
            phrases_idxs = batch["phrases_idxs"]
            snippet_idx = batch["snippet_idx"]
            snippets_toks = batch["snippets_toks"]

            batched_new_phrase_snippet_token_ids = snippets_toks
            # batched_new_phrase_tokenized_ids = [self.new_phrases_ids[pi] for pi in phrases_idxs]
            # batched_new_phrase_atm_embs = [self.new_phrases_trainable_atm_embs[pi] for pi in phrases_idxs]

            batched_sample_embs_w_merged_pattern = []
            batched_patterns_in_merged_seq_idx = []
            batched_patterns_in_gt_seq_idx = []

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
                batched_sample_embs_w_merged_pattern.append(sample_embs_w_merged_pattern)
                batched_patterns_in_merged_seq_idx.append(patterns_in_merged_seq_idx)
                batched_patterns_in_gt_seq_idx.append(patterns_in_gt_seq_idx)

            padded_batched_sample_embs_w_merged_pattern = pad_sequence(
                batched_sample_embs_w_merged_pattern,
                batch_first=True,
                padding_value=-1,
            )
            lengths = torch.as_tensor([v.size(0) for v in batched_sample_embs_w_merged_pattern])
            batch["padded_emb_inputs_with_atm"] = padded_batched_sample_embs_w_merged_pattern
            batch["padded_emb_inputs_with_atm_orig_lengths"] = lengths
            batch["new_phrase_in_gt_seq_idxs"] = batched_patterns_in_gt_seq_idx
            batch["new_phrase_in_atm_seq_idxs"] = batched_patterns_in_merged_seq_idx

            yield batch


def batch_atm_for_many_tokens(
    new_phrases_ids,
    new_phrases_train_snippets,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_snippets_per_phrase=50,
):
    DEVICE = model.device
    NUM_PHRASES = len(new_phrases_ids)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model_input_embs_cpu = copy.deepcopy(model.get_input_embeddings()).cpu()
    new_phrases_trainable_atm_embs = [
        create_trainable_atm_emb(new_phrase_ids, model_input_embs_cpu, DEVICE) for new_phrase_ids in new_phrases_ids
    ]

    batched_dataset = BatchedSnippetDataset(
        new_phrases_ids,
        new_phrases_train_snippets,
        # new_phrases_trainable_atm_embs,
        model_input_embs_cpu,
        MBS,
    )
    data_loader = torch.utils.data.DataLoader(batched_dataset, batch_size=None, shuffle=False, num_workers=4, pin_memory=True)
    dataloader_iter = iter(data_loader)

    loss_fn = torch.nn.MSELoss(reduction="mean")

    phrases_bar = tqdm(
        chunk_list(range(NUM_PHRASES), MBS),
        desc=f"ATM for {NUM_PHRASES} new phrases",
        total=NUM_PHRASES // MBS,
    )
    for cur_phrases_idxs in phrases_bar:
        cur_phrases_idxs = list(cur_phrases_idxs)
        cur_phrases_tokens = ["".join(tokenizer.convert_ids_to_tokens(new_phrases_ids[i])) for i in cur_phrases_idxs]
        phrases_bar.set_description(f"ATM for {NUM_PHRASES} new phrases | e.g. {cur_phrases_tokens[:3]}...")

        cur_new_phrase_atm_embs = [new_phrases_trainable_atm_embs[i] for i in cur_phrases_idxs]
        cur_initial_new_phrase_atm_embs = [new_phrases_trainable_atm_embs[i].clone().detach() for i in cur_phrases_idxs]

        optimizer = torch.optim.AdamW(cur_new_phrase_atm_embs, lr=LR, weight_decay=WD)
        NUM_OPT_STEPS = num_snippets_per_phrase // BS
        if cur_phrases_idxs[0] > 0:
            # `batched_dataset.num_snippets_per_phrase()` is not available until the first batch is processed
            assert num_snippets_per_phrase == batched_dataset.num_snippets_per_phrase()
        cur_opt_step = 1
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=int(NUM_OPT_STEPS * 0.3))
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
            batch["padded_emb_inputs_with_atm"] = batch["padded_emb_inputs_with_atm"].to(DEVICE, non_blocking=True)

            atm_token_idxs = batch["new_phrase_in_atm_seq_idxs"]
            phrases_idxs = batch["phrases_idxs"]
            assert phrases_idxs == cur_phrases_idxs
            # replace all placeholder atm embeddings with the actual atm embeddings
            # with torch.no_grad():
            for i, atm_token_idx in enumerate(atm_token_idxs):
                batch["padded_emb_inputs_with_atm"][i, atm_token_idx] = cur_new_phrase_atm_embs[i]
                # assert batch["padded_emb_inputs_with_atm"][i, atm_token_idx].requires_grad
                # assert batch["padded_emb_inputs_with_atm"][i, atm_token_idx[0] + 1].requires_grad is False

            assert batch["padded_emb_inputs_with_atm"][0, atm_token_idxs[0]].requires_grad
            data_fetch_t1 = time.perf_counter()
            total_data_fetch_time += data_fetch_t1 - data_fetch_t0
            # print(f"Data fetch time: {data_fetch_t1 - data_fetch_t0:.4f}")

            batch_phrase_snippet_token_ids = batch["snippets_toks"]

            # ground truth forward pass - no_grad!
            with torch.no_grad():
                out = model(
                    batch_phrase_snippet_token_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                batched_gt_hiddens = out["hidden_states"][TARGET_LAYER]
                del out

            # forward pass with the merged pattern
            batch_phrase_snippets_atm_padded_input_embs = batch["padded_emb_inputs_with_atm"]
            out = model(
                inputs_embeds=batch_phrase_snippets_atm_padded_input_embs,
                output_hidden_states=True,
                return_dict=True,
            )
            batched_phrase_snippets_atm_hiddens = out["hidden_states"][TARGET_LAYER]
            del out

            # remove padding
            batched_phrase_snippets_atm_hiddens = [
                t[: batch["padded_emb_inputs_with_atm_orig_lengths"][i], :]
                for i, t in enumerate(batched_phrase_snippets_atm_hiddens)
            ]

            assert len(batched_gt_hiddens) == len(batched_phrase_snippets_atm_hiddens)

            for i in range(len(batched_gt_hiddens)):
                # assert batched_gt_hiddens[i].size(0) == batched_phrase_snippets_atm_hiddens[i].size(0)
                assert batched_phrase_snippets_atm_hiddens[i].size(0) == batch["padded_emb_inputs_with_atm_orig_lengths"][i]
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
                # pre_clip_grad_norm = torch.norm(cur_new_phrase_atm_embs.grad).item()
                avg_pre_clip_grad_norm = torch.norm(torch.stack([emb.grad for emb in cur_new_phrase_atm_embs]), dim=-1).mean()
                torch.nn.utils.clip_grad_norm_(cur_new_phrase_atm_embs, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                cur_opt_step += 1

                fwd_pass_bar.set_description(f"Fwd passes (Step {cur_opt_step}/{NUM_OPT_STEPS})...")

                with torch.no_grad():
                    avg_diff_norms_to_initial = (
                        torch.norm(torch.stack(cur_new_phrase_atm_embs) - torch.stack(cur_initial_new_phrase_atm_embs))
                        .mean()
                        .item()
                    )
                    avg_cosine_sims_w_initial = (
                        cosine_similarity(
                            torch.stack(cur_new_phrase_atm_embs),
                            torch.stack(cur_initial_new_phrase_atm_embs),
                        )
                        .mean()
                        .item()
                    )
                    avg_atm_emb_norms = torch.norm(torch.stack(cur_new_phrase_atm_embs), dim=1).mean().item()

                # avg_error_hist.append(avg_error_for_step)
                avg_cur_step_error /= BS
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
                avg_cur_step_error = 0

    return new_phrases_trainable_atm_embs


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
    data_generation_prompt_pattern_ids = [tokenizer.bos_token_id] + new_phrase_tokenized_ids.tolist()
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
        generation_eval_prompt = f"{eval_new_phrase}"
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
):
    desired_samples = num_samples
    data = []
    # print(f"generating {desired_samples} samples with max new tokens {max_length}")

    bar = tqdm(total=desired_samples, leave=False, desc="Generating snippets")
    # set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    while len(data) < desired_samples:
        new_data = model.generate(
            torch.tensor([*pattern], device=model.device).unsqueeze(0),
            do_sample=True,
            max_new_tokens=max_length,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            # top_k=50,
            num_return_sequences=min(1000, desired_samples) if concat_every == -1 else min(concat_every * 100, desired_samples),
        )
        if concat_every != -1:
            # concat every n samples into a single sample
            new_data = [
                torch.cat([new_data[i] for i in range(j, j + concat_every)]) for j in range(0, len(new_data), concat_every)
            ]
        data.extend(new_data)
        bar.update(len(new_data))
    # print(len(data))
    return [{"input_ids": i.unsqueeze(0)} for i in new_data]


def test_prefix_whitespace_when_adding_token(new_phrase, tokenizer: PreTrainedTokenizer):
    tokenizer = copy.deepcopy(tokenizer)
    new_phrase_starts_with_whitespace = new_phrase.startswith(" ") or new_phrase.startswith(SPIECE_UNDERLINE)
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
