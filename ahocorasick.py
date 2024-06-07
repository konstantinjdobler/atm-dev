import time

import ahocorasick_rs
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer


def hash_int_seq(sequence):
    # Use chr(1000 + x) to avoid conflicts with ASCII characters # TODO check if necessary [KD]: I don't think it is
    return "".join(chr(1000 + x) for x in sequence)


def unhash_int_seq(string):
    return [ord(x) - 1000 for x in string]


def collect_snippets_with_patterns_from_dataset(
    patterns_ids, tokenizer: PreTrainedTokenizer, dataset: Dataset, stopping_condition="num_docs:16000"
) -> dict[str, list[tuple[list[int], int]]]:
    """
    Cast the problem of detecting occurences of a sequence of tokens in a larger document as a string matching problem.
    We use an optimized Rust-based Aho-Corasick implementation with iterative pruning of the search patterns.

    The main performance bottlenecks are loading the data into memory and allocating memory for the results. The actual Aho-Corasick search is very fast, especially with our iterative pruning of the search patterns.

    TODO: implement more memory efficient results storage / pre-allocate memory once (not possible with python lists in a sensible way).

    Returns: A dictionary with `hash_int_seq(pattern)` as keys and the found snippets as values, along with the start position of the pattern in the snippet.
    """
    print("Starting Aho-Corasick snippet for patterns from `pattern_ids` in `dataset`...")
    stringified_patterns = [hash_int_seq(pattern) for pattern in patterns_ids]

    max_pattern_len = max(map(len, patterns_ids))
    print(f"Max pattern len: {max_pattern_len}")
    OFFSET_BEFORE = 300
    OFFSET_AFTER = 300

    min_num_of_collected_snippets = 200
    DONE = False

    collected_snippets = {pattern: [] for pattern in stringified_patterns}

    def collection_ahocorasick_f(tokenized_docs: list[list[int]]):
        # rebuild every new iter since we can prune patterns that have enough snippets
        t_before_init = time.perf_counter()
        ac = ahocorasick_rs.AhoCorasick(stringified_patterns, implementation=ahocorasick_rs.Implementation.DFA)
        t_after_init = time.perf_counter()
        print(f"Init time for Aho-Corasick: {t_after_init - t_before_init}")

        # timings = {"hashing": [], "find_matches": [], "match_proc": [], "append": []}
        t0 = time.perf_counter()
        for doc in tqdm(tokenized_docs, leave=False):
            # t0 = time.perf_counter()
            stringified_doc = hash_int_seq(doc)
            # t1 = time.perf_counter()
            matches = ac.find_matches_as_indexes(stringified_doc, overlapping=True)
            # t2 = time.perf_counter()

            for match in matches:
                pattern_id, start, end = match
                pattern_hash = stringified_patterns[pattern_id]

                snippet = doc[max(0, start - OFFSET_BEFORE) : min(end + OFFSET_AFTER, len(doc))]

                start_pos_of_pattern_in_snippet = start - max(0, start - OFFSET_BEFORE)

                # t_before_append = time.perf_counter()
                collected_snippets[pattern_hash].append((snippet, start_pos_of_pattern_in_snippet))
                # t_after_append = time.perf_counter()
                # timings["append"].append(t_after_append - t_before_append)

            # t3 = time.perf_counter()

        #     timings["hashing"].append(t1 - t0)
        #     timings["find_matches"].append(t2 - t1)
        #     timings["match_proc"].append(t3 - t2)
        # # print(
        #     f"Average times -- hash doc {avg_diffs[0] / len(tokenized)} | find matches {avg_diffs[1] / len(tokenized)} | match proc {avg_diffs[2] / len(tokenized)}"
        # )
        # print(
        #     f"Average times -- hash doc {sum(timings['hashing']) / len(tokenized)} | find matches {sum(timings['find_matches']) / len(tokenized)} | match proc {sum(timings['match_proc']) / len(tokenized)}"
        # )
        print(f"Collection for {len(tokenized_docs)} done in {time.perf_counter() - t0:.4f} seconds")

    proc_bs = 500
    print("Starting collection")
    iter_counter = 0

    if stopping_condition.startswith("num_docs:"):
        MAX_NUM_DOCS = int(stopping_condition.split(":")[1])
    else:
        raise NotImplementedError(f"Stopping condition {stopping_condition} not implemented")
    # MAX_NUM_DOCS = 16_000
    MAX_BS = 256_000
    # tokenized_dataset = tokenized_dataset.take(MAX_NUM_DOCS)

    # def take_n_from
    total_docs_processed = 0
    total_t0 = time.perf_counter()
    while not DONE:
        print(f"\n\n--------------\nStarting iteration {iter_counter} with {proc_bs} documents\n\n----------")
        data_fetch_t0 = time.perf_counter()
        print("Fetching data...")
        batch = dataset.select(list(range(total_docs_processed, total_docs_processed + proc_bs)), keep_in_memory=True)["tokens"]
        data_fetch_t1 = time.perf_counter()
        print(f"Data fetch time: {data_fetch_t1 - data_fetch_t0:.4f} seconds")

        collection_ahocorasick_f(batch)
        total_docs_processed += proc_bs
        iter_counter += 1
        print(f"----Summary for processed {total_docs_processed} documents")
        collected_docs_lens = {k: len(v) for k, v in collected_snippets.items()}
        least_num_snippets = min(collected_docs_lens.values())
        if least_num_snippets >= min_num_of_collected_snippets:
            print("All tokens have enough snippets")
            DONE = True
            break
        print(
            f"Num tokens with enough snippets: {len([k for k, v in collected_docs_lens.items() if v >= min_num_of_collected_snippets])}"
        )

        # Prune patterns that have enough snippets
        stringified_patterns = [k for k, v in collected_docs_lens.items() if v < min_num_of_collected_snippets]
        print(
            f"Hot patterns: {len(stringified_patterns)} | new num snippets: {sum([v for k, v in collected_docs_lens.items() if v < min_num_of_collected_snippets])}"
        )

        dict_num_snipppets_to_count = {}
        for k, v in collected_docs_lens.items():
            dict_num_snipppets_to_count[v] = dict_num_snipppets_to_count.get(v, 0) + 1
        print("Collected counts", sorted(dict_num_snipppets_to_count.items())[:30])
        print(
            f"Num tokens with least snippets ({least_num_snippets}): {len([k for k, v in collected_docs_lens.items() if v == least_num_snippets])}"
        )
        print(f"Num tokens with fewer than 50 snippets: {len([k for k, v in collected_docs_lens.items() if v < 50])}")
        print(
            "Example tokens with least snippets:",
            [
                tokenizer.convert_ids_to_tokens(unhash_int_seq(k))
                for k, v in collected_docs_lens.items()
                if v == least_num_snippets
            ][:10],
        )
        print(f"Total time for {iter_counter} iterations: {time.perf_counter() - total_t0:.4f} seconds")
        if DONE or total_docs_processed >= MAX_NUM_DOCS:
            break
        if proc_bs < MAX_BS:
            proc_bs *= 2
    return collected_snippets
