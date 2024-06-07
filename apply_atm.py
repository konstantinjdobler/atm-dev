import copy
import os
import pickle
from functools import partial

from sympy import N
import torch
from datasets import Dataset, load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralForCausalLM,
    PreTrainedModel,
)

from ahocorasick import collect_snippets_with_patterns_from_dataset, hash_int_seq
from atm import batch_atm_for_many_tokens
from atm_utils import SPIECE_WHITESPACE, get_new_phrase_tokenized_ids

DATA_SUFFIX = "ERROR_NOT_YET_SET"


def tokenize_dataset(tokenizer, batch_size=16_000):
    """
    TODO: make more general without hacky global DATA_SUFFIX
    """
    if os.path.exists(f"./tokenized_dataset_{DATA_SUFFIX}/"):
        return
    if DATA_SUFFIX == "german":
        dataset = load_dataset("uonlp/CulturaX", "de", split="train", streaming=True)
    else:
        dataset = load_dataset("ncbi/pubmed", split="train", streaming=True)
        dataset = dataset.map(
            lambda x: {"text": x["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]},
            batched=False,
            # batch_size=batch_size,
            remove_columns=dataset.column_names,
        )
        "MedlineCitation" > "Article" > "Abstract" > "AbstractText"

    tokenized_dataset = dataset.map(
        lambda x: {"tokens": tokenizer(x["text"], truncation=False, padding=False, add_special_tokens=False)["input_ids"]},
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
    )

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    ds = Dataset.from_generator(
        partial(gen_from_iterable_dataset, tokenized_dataset.take(3_000_000)),
        features=tokenized_dataset.features,
    )

    ds.save_to_disk(f"./tokenized_dataset_{DATA_SUFFIX}/")


def main(
    model_path: str = "mistralai/Mistral-7B-v0.1",
    tokenizer: str = "./tokenizers/de/de32k/",
    out_path: str = "./testdata/",
):
    # TODO: make less hacky -- but it works for now
    global DATA_SUFFIX
    if "de32k" in tokenizer:
        DATA_SUFFIX = "german"
    else:
        DATA_SUFFIX = "pubmed"
    torch.set_float32_matmul_precision("high")
    # we need to use these kwargs - especially `add_prefix_space=False` to get the correct tokenization of single tokens from target tokenizer
    source_tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, add_prefix_space=False, from_slow=True)
    target_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.compile()

    # Collect overlapping tokens -- copy their embeddings (we can also try ATM, explore this later)
    new_vocab = {}
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = sorted(target_tokenizer.get_vocab().items())
    todo_tokens = []
    for token, token_id in tqdm(target_vocab):
        token = token.replace("Ġ", SPIECE_WHITESPACE)
        if source_vocab.get(token) is not None:
            new_vocab[token_id] = model.get_input_embeddings().weight[source_vocab[token]]
        elif token in target_tokenizer.all_special_tokens:
            continue
        else:
            todo_tokens.append((token, token_id))
    print(f"Found {len(new_vocab)} tokens in source vocab")

    # use standard tokenizer without prefix_space modification for tokenizing source data (for snippet extraction)
    source_tokenizer_fast = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    patterns_ids = [get_new_phrase_tokenized_ids(t[0], source_tokenizer)[0] for t in todo_tokens]
    stopping_cond = "num_docs:1_000_000"
    snippet_cache = f"collected_snippets_{DATA_SUFFIX}_{stopping_cond}.pkl"
    if os.path.exists(snippet_cache):
        with open(snippet_cache, "rb") as f:
            collected_snippets = pickle.load(f)
    else:
        tokenize_dataset(source_tokenizer_fast, batch_size=16_000)
        dataset = Dataset.load_from_disk(f"./tokenized_dataset_{DATA_SUFFIX}/")
        collected_snippets = collect_snippets_with_patterns_from_dataset(
            patterns_ids, source_tokenizer_fast, dataset, stopping_condition=stopping_cond
        )

        # save collected snippets
        with open(snippet_cache, "wb") as f:
            pickle.dump(collected_snippets, f)
    print("Collected snippets for all tokens")

    SNIPPET_LEN = 100
    NUM_SNIPPETS_PER_TOKEN = 50
    new_phrases_ids = []
    new_phrases_snippets_ids = []
    new_phrases_token_ids_in_target_vocab = []
    todo_tokens_not_enough_snippets = []
    """
    Filter out tokens that do not have enough snippets of the desired length.
    """
    for token, token_id in tqdm(
        sorted(
            todo_tokens,
            key=lambda x: len(collected_snippets[hash_int_seq(get_new_phrase_tokenized_ids(x[0], source_tokenizer)[0])]),
        )
    ):
        snippets = collected_snippets[hash_int_seq(get_new_phrase_tokenized_ids(token, source_tokenizer)[0])]
        if len(snippets) < NUM_SNIPPETS_PER_TOKEN:
            todo_tokens_not_enough_snippets.append((token, token_id))
            continue
        snippets = [s[0][s[1] : s[1] + SNIPPET_LEN] for s in snippets]
        snippets = [[source_tokenizer.bos_token_id] + s[:SNIPPET_LEN] for s in snippets if len(s) >= SNIPPET_LEN][:NUM_SNIPPETS_PER_TOKEN]
        snippets = [{"input_ids": torch.tensor(s).unsqueeze(0)} for s in snippets]
        if len(snippets) < NUM_SNIPPETS_PER_TOKEN:
            todo_tokens_not_enough_snippets.append((token, token_id))
            continue
        new_phrases_ids.append(get_new_phrase_tokenized_ids(token, source_tokenizer)[0])
        new_phrases_snippets_ids.append([s["input_ids"].squeeze(0) for s in snippets])
        new_phrases_token_ids_in_target_vocab.append(token_id)
    print(f"Enough snippets for {len(new_phrases_ids)} new tokens")

    """Do ATM for new tokens."""
    new_phrases_atm_embs = batch_atm_for_many_tokens(new_phrases_ids, new_phrases_snippets_ids, model, source_tokenizer, NUM_SNIPPETS_PER_TOKEN)

    for token_id, atm_embedding in zip(new_phrases_token_ids_in_target_vocab, new_phrases_atm_embs):
        new_vocab[token_id] = atm_embedding

    """Handle tokens that do not have enough snippets."""
    for token, token_id in todo_tokens_not_enough_snippets:
        # init as mean of constituent tokens
        embs = model.get_input_embeddings()
        token_ids = get_new_phrase_tokenized_ids(token, source_tokenizer)[0].to(embs.weight.device)
        new_vocab[token_id] = torch.mean(embs(token_ids), dim=0)

    """Handle special tokens."""
    if target_tokenizer.bos_token_id:
        new_vocab[target_tokenizer.bos_token_id] = model.get_input_embeddings().weight[source_tokenizer.bos_token_id]
    if target_tokenizer.eos_token_id:
        new_vocab[target_tokenizer.eos_token_id] = model.get_input_embeddings().weight[source_tokenizer.eos_token_id]
    if target_tokenizer.pad_token_id:
        new_vocab[target_tokenizer.pad_token_id] = model.get_input_embeddings().weight[source_tokenizer.pad_token_id]
    if target_tokenizer.unk_token_id:
        new_vocab[target_tokenizer.unk_token_id] = model.get_input_embeddings().weight[source_tokenizer.unk_token_id]

    """
    Replace old input embeddings with new input embeddings.
    Right now, I'm exploring only adapting the input embs while keeping the output embs as they are originally -- need to test this.
    """
    # model.config.tie_word_embeddings = True  # hack s.t. only input embedding is resized. [KD]: doesn't seem to work
    old_output_embs = copy.deepcopy(model.get_output_embeddings())
    model.resize_token_embeddings(max(len(target_tokenizer), len(source_tokenizer)))
    # model.config.tie_word_embeddings = False

    new_embedding_weights = model.get_input_embeddings().weight.data
    for token_id, embedding in new_vocab.items():
        new_embedding_weights[token_id] = embedding
    model.get_input_embeddings().weight.data = new_embedding_weights
    # model.set_output_embeddings(old_output_embs)
    model.get_output_embeddings().weight.data = torch.zeros_like(model.get_output_embeddings().weight)
    model.get_output_embeddings().weight.data[: len(source_tokenizer)] = old_output_embs.weight
    # model.get_output_embeddings().weight.data = new_embedding_weights
    print(
        f"New input emb shape: {model.get_input_embeddings().weight.shape} | New output emb shape: {model.get_output_embeddings().weight.shape}"
    )
    model.save_pretrained(out_path)
    target_tokenizer.save_pretrained(out_path)


if __name__ == "__main__":
    Fire(main)
