# atm-dev


## Setup
We use `rye` for reproducible and fast dependecy management.
You can install `rye` with the following snippet (Linux/MacOS):
```sh
curl -sSf https://rye.astral.sh/get | bash
```

The run `rye sync --no-lock --force` to install the dependencies from our `requirements.lock`. The dependencies are installed in `.venv/`. You can actiavte the virtual environment with `source .venv/bin/activate`.

## Running ATM
```sh
python apply_atm.py --model_path=mistralai/Mistral-7B-v0.1 --tokenizer ./tokenizers/de/de32k/ --out_path ./testdata/
```
