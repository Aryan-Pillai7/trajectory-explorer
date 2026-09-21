# trajectory-explorer

**Weight-only diffs between model checkpoints: see *when* each part of a model learns.**

> Status: pre-alpha, under active development. Only `--version` works so far.

trajectory-explorer compares checkpoints of the same architecture tensor by tensor, without
running the model. Its headline use case is the [Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m)
training trajectory: EleutherAI publishes 154 intermediate checkpoints as Hugging Face
revisions (`step0` … `step143000`). It also diffs any two same-architecture checkpoints,
e.g. `pythia-70m` vs `pythia-70m-deduped`, or `SmolLM2-135M` vs `SmolLM2-135M-Instruct`.

## Planned features

- Per-tensor relative delta norm `‖ΔW‖ / ‖W‖` and effective rank of each delta, grouped by
  layer and component (attention QKV, attention output, MLP in/out, embeddings, layer norms).
- Layer × component heatmap for a pair of checkpoints, plus a trajectory view across steps.
- A ranked "what moved most" table with labels: low-rank vs dense, concentrated vs spread out.
- An honest noise floor: effects below it are visually muted, and identical inputs report
  "no significant difference".
- Output: one self-contained, offline HTML file with inline SVG. No JavaScript framework.
- Low resource use: tensors stream one at a time, and at most two checkpoints are on disk.

## License

[Apache-2.0](LICENSE)
