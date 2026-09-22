# Metrics

What each number in a report means, how it is computed, and what it can't tell you. The code
is in `src/trajectory_explorer/metrics.py` (measurements) and `noise.py` (floor, status,
labels).

For every tensor present in both checkpoints, with A the reference and B the one compared:

    ΔW = B − A        (computed in float32; sums accumulated in float64)

## Relative and absolute change

- **Absolute change** `‖ΔW‖`: the Frobenius norm (square root of the sum of squared elements).
- **Starting norm** `‖W_A‖`.
- **Relative change** `‖ΔW‖ / ‖W_A‖`. This is the main number: a heatmap cell's colour, and
  the ranking in the tables.

A large relative change can come from a tiny starting size, which is why the tables also show
the starting norm. If A is exactly zero and B is not (a bias initialised to zero, for
example), there is no relative change. The tensor gets the status `from_zero`: it counts as a
real change and is drawn as a dashed outline marked "0→", never as a coloured cell.

## The noise floor

Two stored copies of the same weights can differ just from rounding to their storage formats.
The noise floor is the relative change that rounding alone could plausibly produce:

    floor = k × √(u_A² + u_B²) / √3,   k = 3

`u` is the unit roundoff of the precision each side carries:

| Precision | u |
|---|---|
| float32 | 2⁻²⁴ ≈ 5.96e-8 |
| float16 | 2⁻¹¹ ≈ 4.88e-4 |
| bfloat16 | 2⁻⁸ ≈ 3.91e-3 |

`u / √3` is a conservative estimate of the root-mean-square relative rounding error per element.
For round-to-nearest the exact value is about 0.42 u, a bit below `1/√3 ≈ 0.58`. `k = 3` is a
safety factor. A change at or below the floor is consistent with rounding alone. It is drawn
grey and hatched and never counts as significant.

### Effective precision: float32 files that hold float16 values

What counts is the precision a tensor really carries, not its file format. Pythia's step
checkpoints are stored as float32, but every value in them is exactly a float16 number
(measured on step1000, step142000 and step143000: 76 of 76 tensors each). Using the float32
floor would call float16-sized rounding differences "significant".

The rule:

- It only affects sides stored as float32.
- A float32 side counts as float16 only if, **in both checkpoints**, every value of that tensor
  is exactly a float16 number: float32 → float16 → float32 gives the same bits. A NaN counts if
  it round-trips to NaN, ±inf round-trips, and finite values beyond float16's range fail.
- A side stored as float16 is float16 by definition.
- There is no bfloat16 check. A bfloat16 value stored as float32 cannot be told apart by value,
  and files stored as bfloat16 already get the bfloat16 floor.

The check runs in the same chunked pass as the norms, so it costs no extra copy of the tensor.
Each report's noise-floor section says whether the rule applied and to how many tensors. For
two float32 Pythia checkpoints the floor goes from 1.46e-7 (float32) to 1.20e-3, i.e. 0.12%
(float16).

### Status of a tensor

In order:

| Status | When |
|---|---|
| `no_change` | `‖ΔW‖ = 0` |
| `from_zero` | A is exactly zero and B is not |
| `below_floor` | relative change ≤ floor |
| `below_control` | relative change ≤ the control pair's relative change for this tensor (only with `diff --control`) |
| `significant` | anything else |

`significant` and `from_zero` count as changes. The control is a reference scale, not a null:
see [below](#adjacent-step-diffs-are-not-a-null).

## Effective rank and r90 (matrices only)

For a tensor with two or more dimensions, ΔW is reshaped to (rows, everything else) and its
singular values σ₁ ≥ σ₂ ≥ … are computed. To keep this cheap for big matrices, the tool builds
the smaller Gram matrix (ΔWᵀΔW or ΔWΔWᵀ) in float64, accumulated over row chunks, and takes
the square roots of its eigenvalues. For the 50304 × 512 Pythia embedding that is a 512 × 512
problem. Squaring loses singular values below about 1e-8 of the largest, and those carry no
weight in either measure below.

- **Effective rank** (Roy & Vetterli, 2007): `exp(H)`, where H is the Shannon entropy of the
  normalised singular values `pᵢ = σᵢ / Σσ`. It is 1 for a rank-1 change and equals the full
  rank when all singular values are equal. It is not an integer.
- **r90**: the smallest k such that the top k singular values hold 90% of the change's energy
  `Σσ²`. It is an integer.
- The report shows both as "effective rank X of N, r90 Y", where N = min(rows, cols) is the
  largest possible rank.

The two can disagree a lot, because they weight the spectrum differently: effective rank uses
σ, r90 uses σ². One real example, step100000 → step143000 of Pythia-70M, the unembedding
(`embed_out.weight`, 50304 × 512): effective rank 222 of 512, but r90 = 1. One direction holds
90% of the energy, and the other 511 are small but many, so the entropy stays high. Read the
two together. Neither one alone tells the whole story.

### The low-rank / dense label

A random (Gaussian) matrix doesn't have full effective rank either, so the comparison is
against a same-shape random matrix. Its expected effective rank comes from the
Marchenko–Pastur distribution of its singular values, computed numerically for the matrix's
aspect ratio.

- **low-rank** if `effective rank < 0.25 × effective rank of a same-shape Gaussian matrix`;
- **dense** otherwise;
- `n/a` for vectors, and for matrices whose change is zero or that have a dimension of 1.

## Concentration: the concentrated / spread label

This measures how much of the squared change `‖ΔW‖²` is held by the top 5% of rows. Rows are the
first axis, and at least one row is always counted.

- **concentrated** if those rows hold ≥ 50% of the squared change;
- **spread** otherwise;
- `n/a` for matrices with fewer than 20 rows (5% would be less than one row), and for all
  vectors.

For vectors (biases, norm scales), each element is a "row". The share is shown as a number,
"top 5% of elements hold X%", in the vectors table and tooltips, instead of a label.

## Thresholds are heuristics

| Constant | Value | In `noise.py` |
|---|---|---|
| Floor safety factor | k = 3 | `FLOOR_K` |
| Low-rank cut-off | 0.25 × Gaussian effective rank | `LOW_RANK_RATIO` |
| Rows counted for concentration | top 5% | `TOP_ROW_FRACTION` |
| Concentrated cut-off | ≥ 50% of the squared change | `CONCENTRATED_SHARE` |
| Minimum rows for the concentration label | 20 | `MIN_ROWS_FOR_CONCENTRATION` |
| Energy fraction for r90 | 90% | `ENERGY_FRACTION` (in `metrics.py`) |

These are reasonable, fixed choices, not universal constants. So far they have been checked
mainly on one architecture, Pythia-70M (GPT-NeoX, float32 files holding float16 values). They
have not been validated on other model families, other sizes or fine-tuning deltas. Treat a
label as a pointer to look at the numbers next to it, not as a finding on its own. The floor is
the most principled of them, because it follows from the storage format. The rank and
concentration labels are the least tested.

## Groups: heatmap cells

Tensors are grouped into cells by (layer, component, kind):

- **layer**: the block index, or none for the embedding, the final norm and the unembedding;
- **component**: `embed`, `attn_qkv`, `attn_out`, `mlp_in`, `mlp_out`, `norm`, `unembed`, or
  `other`;
- **kind**: `matrix` (2-D and up) or `vector` (1-D: biases, norm scales).

Matrices and vectors are never mixed. A small bias with a huge relative change must not make a
cell look as if it describes a weight matrix. The pair report and the trajectory report show
them in two separate panels, each with its own colour scale, so colours are not comparable
between the panels.

Within a cell, sums of squares are pooled:

    cell relative change = √(Σ ‖ΔW_i‖²) / √(Σ ‖W_A,i‖²)
    cell floor           = √(Σ (floor_i × ‖W_A,i‖)²) / √(Σ ‖W_A,i‖²)

The cell floor is the rounding noise the members would show together, weighted by their size.
A cell whose members are all at or below their own floors is at or below the cell floor too.
With `--control`, the control pair is grouped the same way and each cell is compared with the
matching control cell. A cell is `from_zero` only if every member starts at exactly zero.

Colour: one blue hue, five bins spaced on a log scale between the smallest and largest
significant value in that panel. In a trajectory, a panel's scale is fixed across all intervals,
so columns within a panel are comparable.

## Adjacent-step diffs are not a null

A diff between neighbouring training checkpoints, such as step142000 → step143000, is a real
change, not noise: the model is still learning. In that pair, 71 of 76 Pythia-70M tensors are
above the noise floor. So it is useful as a **reference scale**: `diff --control A2:B2` mutes
tensors and cells that moved no more than the control pair did. Calling everything smaller than
it "nothing" would be wrong.

The nulls the tool does rely on:

- the rounding floor above;
- byte-identical files (sha256 match), which are reported as "No significant difference" without
  reading any weights;
- checkpoints verified to hold the same weights. Pythia-70M `main` is bit for bit the float16
  rounding of `step143000` (76 of 76 tensors), and the tool reports that pair as "No significant
  difference".

Trajectory intervals also differ in length: early ones can be a single step and late ones tens
of thousands of steps. A bigger number in a longer interval is not a faster change. The
trajectory report has a section on this.
