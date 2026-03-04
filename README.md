
## QK-MFA Training

`train_qk_single.py` supports two data sources:

- `llm`: extract query/key vectors from an LLM using supervised text data.
- `toy_mixture`: generate synthetic `(q, k)` pairs from `dgp/mixture.py`.

Run a toy training job:

```bash
python train_qk_single.py --config configs/qk_single.toy.example.yml
```

Run an LLM-based training job:

```bash
python train_qk_single.py --config configs/qk_single.example.yml
```

Optional overrides for either mode:

```bash
python train_qk_single.py \
  --config configs/qk_single.toy.example.yml \
  --run-name my_run \
  --output-dir ./runs
```
