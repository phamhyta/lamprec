# LAMP-Rec

LAMP-Rec is a label-free, anytime-valid monitor of a deployed recommender's
running ranking quality. It tracks the value of the currently serving policy
without querying labels: the feedback that does arrive is logged-bandit,
position-biased, and delayed — the monitor combines an immediate model-based
proxy with a delayed, examination-corrected residual and maintains a point
estimate, a time-uniform confidence sequence, and a degradation alarm through
drift, out-of-order maturation, and censoring.

## Status: public interface skeleton

This repository is the **public interface** for the paper, which is currently
under review. It ships the full package structure — every module, class, and
function signature, with documentation — so the pipeline can be read and
reviewed end to end. **It is only partially runnable:** the schema, the basic
rectifier, the forgetting-rate theory, and the simple baselines are complete;
the body of each withheld component prints a release notice and raises
`NotImplementedError`. The executable implementation, the experiment runners,
and the reproduction artifacts are released here once the paper is accepted.

Importing the package and inspecting the API works; calling a withheld
component surfaces:

```
LAMP-Rec reference implementation is not public yet. The full source of this
component will be released in this repository once the paper is accepted; the
public release currently ships the interface and documentation only.
```

## Monitor pipeline

A logged round flows through four components (`lamprec/core/monitor.py`):

1. **Proxy layer** (`core/estimator.py`) — every served context contributes its
   free model score immediately; the reward is withheld until maturation.
2. **Rectifier** (`core/estimator.py`, `core/slate.py`) — when a round matures
   after its delay, the position-debiased doubly-robust residual re-anchors the
   estimate, keeping the bias non-accumulating under drift.
3. **Confidence sequence** (`core/confseq.py`) — each completed pseudo-outcome
   is appended once, in maturation order, to a betting confidence sequence that
   is valid at every query time.
4. **Forgetting + alarm** (`core/drift.py`, `core/confseq.py`) — geometric
   recency weights define the monitored window; the alarm fires only when the
   whole interval sits below the quality floor.

## Package map

| Module | Role |
|--------|------|
| `lamprec.data` | Common `Stream` schema and the OBP / KuaiRand adapters |
| `lamprec.core` | The online monitor: estimator, confidence sequence, forgetting theory, examination factorization |
| `lamprec.sim` | Synthetic stream generator with exact ground-truth quality |
| `lamprec.baselines` | Delay-aware and label-free baselines compared in the paper |
| `lamprec.metrics` | Tracking error, time-uniform coverage, width, alarm metrics |
| `experiments/` | Per-question runners, equal-labels protocol, report builder |
| `tests/` | Public portion of the test suite (withheld claims are skipped) |

## Input / output contract

Every data source emits a `lamprec.data.event.Stream` — a columnar log with one
row per served round:

```python
Stream(g=..., p=..., e=..., r_obs=..., delay=..., theta=...)
```

Fields: `g` (immediate label-free proxy), `p` (logging propensity), `e`
(examination propensity), `r_obs` (delayed observed reward), `delay`
(maturation delay in rounds), plus optional `g_action`, `pi_target`, `rank`,
`context`. `theta` is the gold running value, used for evaluation only.

The monitor returns per-round point estimates `theta_hat`, per-query interval
bounds `lo` / `hi`, and `alarm` — the first query time the interval certifies
quality below the operating floor (`-1` if never).

## Install

```bash
uv sync
```

The package imports with `numpy` alone; the released implementation adds the
stack listed under the `full` extra in `pyproject.toml`. The public tests run
with `uv run --extra dev pytest -q`.

## Release status

This is a pre-acceptance public skeleton. The private research repository keeps
the implementation, manuscript, experiment scripts, result bundles, and data
manifests until the review process is complete.

## License

MIT. See `LICENSE`.
