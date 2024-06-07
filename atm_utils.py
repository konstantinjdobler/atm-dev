from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

SPIECE_WHITESPACE = "▁"
GPT_BPE_WHITESPACE = "Ġ"


def chunk_list(lst, n):
    # Split the list into chunks of size n
    return (lst[i : i + n] for i in range(0, len(lst), n))


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
