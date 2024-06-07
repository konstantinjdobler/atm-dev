import os
import pickle
from functools import partial

import torch
from datasets import Dataset, load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from ahocorasick import collect_snippets_with_patterns_from_dataset, hash_int_seq
from atm import batch_atm_for_many_tokens, get_new_phrase_tokenized_ids


def tokenization_generator(tokenizer, hf_dataset, batch_size=16_000):
    if os.path.exists("./tokenized_dataset/"):
        # ds = Dataset.load_from_disk("./tokenized_dataset/")
        # yield from ds.iter(batch_size=1)
        return
    dataset = load_dataset("uonlp/CulturaX", "de", split="train", streaming=True)

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

    ds.save_to_disk("./tokenized_dataset/")

    # yield from ds.to_iterable_dataset()


def main(
    model_path: str = "mistralai/Mistral-7B-v0.1",
    tokenizer: str = "./tokenizers/de/de32k/",
    out_path: str = "./testdata/",
):
    # torch.set_default_device("cuda:0")  # TODO: fix this better
    torch.set_float32_matmul_precision("high")
    source_tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, add_prefix_space=False, from_slow=True)
    target_tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False, add_prefix_space=False, from_slow=True)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    # MistralForCausalLM
    model.compile()

    new_vocab = {}
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = sorted(target_tokenizer.get_vocab().items())
    todo_tokens = []
    for token, token_id in tqdm(target_vocab):
        if source_vocab.get(token) is not None:
            new_vocab[token_id] = model.get_input_embeddings().weight[source_vocab[token]]
            # print(f"Found {token} in source vocab")
        elif token in target_tokenizer.all_special_tokens:
            continue
        else:
            todo_tokens.append((token, token_id))
    print(f"Found {len(new_vocab)} tokens in source vocab")
    #     atm_embedding = atm_merge_token(model, token)
    # new_vocab[token_id] = atm_embedding

    # get data snippets for new tokens
    source_tokenizer_fast = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    patterns_ids = [get_new_phrase_tokenized_ids(t[0], source_tokenizer)[0] for t in todo_tokens]

    if os.path.exists("collected_snippets2.pkl"):
        with open("collected_snippets2.pkl", "rb") as f:
            collected_snippets = pickle.load(f)
    else:
        tokenization_generator(source_tokenizer_fast, None, batch_size=16_000)
        dataset = Dataset.load_from_disk("./tokenized_dataset/")
        collected_snippets = collect_snippets_with_patterns_from_dataset(
            patterns_ids, source_tokenizer_fast, dataset, stopping_condition="num_docs:500"
        )

        # save collected snippets
        with open("collected_snippets2.pkl", "wb") as f:
            pickle.dump(collected_snippets, f)
    print("Collected snippets for all tokens")

    
    SNIPPET_LEN = 50
    NUM_SNIPPETS_PER_TOKEN = 50 
    new_phrases_ids = []
    new_phrases_snippets_ids = []
    new_phrases_token_ids_in_target_vocab = []
    for token, token_id in tqdm(
        sorted(
            todo_tokens,
            key=lambda x: len(collected_snippets[hash_int_seq(get_new_phrase_tokenized_ids(x[0], source_tokenizer)[0])]),
        )
    ):
        snippets = collected_snippets[hash_int_seq(get_new_phrase_tokenized_ids(token, source_tokenizer)[0])]
        if len(snippets) < 50:
            # print(f"Only {len(snippets)} snippets for {token}")
            continue
        snippets = [s[0][s[1] : s[1] + 50] for s in snippets]
        snippets = [[source_tokenizer.bos_token_id] + s[:50] for s in snippets if len(s) >= 50][:50]

        snippets = [{"input_ids": torch.tensor(s).unsqueeze(0)} for s in snippets]

        # print(f"Found {len(snippets)} snippets for {token}")
        if len(snippets) < 50:
            # print(f"Only {len(snippets)} snippets for {token}")
            continue
        # atm_embedding = atm_merge_token(model, token, snippets)
        # atm_embedding = atm_merge_token(token, source_tokenizer, model, train_snippets=snippets, eval=False)
        # new_vocab[token_id] = atm_embedding
        new_phrases_ids.append(get_new_phrase_tokenized_ids(token, source_tokenizer)[0])
        new_phrases_snippets_ids.append([s["input_ids"].squeeze(0) for s in snippets])
        new_phrases_token_ids_in_target_vocab.append(token_id)
    print(f"Enough snippets for {len(new_phrases_ids)} new tokens")

    new_phrases_atm_embs = batch_atm_for_many_tokens(new_phrases_ids, new_phrases_snippets_ids, model, source_tokenizer)

    for token_id, atm_embedding in zip(new_phrases_token_ids_in_target_vocab, new_phrases_atm_embs):
        new_vocab[token_id] = atm_embedding

    model.config.tie_word_embeddings = True  # hack s.t. only input embedding is resized
    model.resize_token_embeddings(len(target_tokenizer))
    model.config.tie_word_embeddings = False

    new_embedding_weights = model.get_input_embeddings().weight.data
    for token_id, embedding in new_vocab.items():
        new_embedding_weights[token_id] = embedding
    model.get_input_embeddings().weight.data = new_embedding_weights
    # model.get_output_embeddings().weight.data = new_embedding_weights
    print(
        f"New input emb shape: {model.get_input_embeddings().weight.shape} | New output emb shape: {model.get_output_embeddings().weight.shape}"
    )
    model.save_pretrained(out_path)


if __name__ == "__main__":
    Fire(main)
