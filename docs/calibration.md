# Calibration experiment

```bash
uv run calm calibration --config configs/calibration.yaml
```

Uses the existing random-mask and channel-mask checkpoints without retraining.
Requires the frozen preprocessing selection, audit manifest, and local recordings.
The command verifies checkpoint subject/preprocessing compatibility, channel order,
and raw-file hashes. It refuses to overwrite an existing output directory; use
`--output-dir artifacts/results/calibration_repeat` for an independent rerun.

The protocol in `configs/calibration.yaml` evaluates subjects 46–60, holding out
one run and calibrating on the other two, rotating runs 4/8/12. Budgets are 5 or
10 labeled trials per class and all 30 available trials. Ten sampling seeds use
identical examples across methods, with each 5/class subset nested in its
10/class counterpart. The full-data condition is evaluated once per rotation.
A single run has too few trials to support 10/class, hence the two-run pool.

Comparisons include majority, bandpower, CSP–LDA, flattened full log-STFT,
flattened pooled MAE-input tokens, a random encoder, and both pretrained encoders.
Every logistic probe uses StandardScaler fitted **only on the selected calibration
examples**, then logistic regression with fixed C=1 (no test-based tuning).
Frozen encoder features use the original development-fitted normalizer. Three
additional full-data unscaled probes isolate the effect of standardization.

Outputs in `artifacts/results/calibration_v1/` include `report.md`,
`calibration_curve.png`, all fold metrics and predictions, shared sample IDs,
subject means, paired subject differences and bootstrap intervals, and a protocol
record with configuration, dependency versions, source/checkpoint/data hashes.
Full run artifacts remain local. Selected tables and the curve are published in
`reports/calibration_v1/`.

This is an **exploratory follow-up on previously inspected subjects**, not a new
locked test or zero-shot transfer evaluation. Subject means average sampling
seeds within rotations, then rotations. Bootstrap intervals resample subjects;
seeds and rotations are not independent subjects. Sampling repetitions do not
measure pretraining variability: each masking strategy uses one saved checkpoint.

### Calibration results (15 subjects)

Balanced accuracy (%); each logistic probe is standardized. All = 30 total trials.

| Method | 5/class | 10/class | All |
|---|---:|---:|---:|
| Majority | 50.00 | 50.00 | 50.00 |
| Bandpower + logistic regression | 56.77 | 60.61 | 63.73 |
| CSP–LDA | 59.53 | 62.71 | 66.31 |
| Full flattened log-STFT | 52.10 | 53.09 | 52.78 |
| Flattened MAE-input tokens | 54.56 | 57.06 | 59.42 |
| Random frozen encoder | 52.07 | 51.96 | 49.98 |
| Random-mask pretrained encoder | 51.13 | 52.43 | 52.00 |
| Channel-mask pretrained encoder | 51.53 | 52.24 | 52.60 |

Standardization did not rescue the random-mask pretrained encoder: full-data
balanced accuracy was 52.20% unscaled and 52.00% standardized (paired change
−0.20 percentage points; subject-bootstrap 95% CI −3.65 to +3.04). The
channel-mask encoder changed from 52.04% to 52.60% (+0.56 points; CI −3.15 to +4.15).

The exact pooled input tokens carry more linearly decodable information under
this protocol than the pretrained mean-pooled embedding: 59.42% versus 52.00%
with all labels, a paired advantage of 7.42 points (CI +1.23 to +13.69).
Flattening the full STFT is not a rescue either (52.78%); this uses 24,192
features with at most 30 calibration trials and fixed regularization. That
failure does not establish that the raw spectral representation lacks signal.

Random-mask pretraining minus the standardized random encoder is −0.93 points
at 5/class (CI −2.44 to +0.56), +0.46 at 10/class (−0.86 to +1.83), and +2.02
with all labels (−1.07 to +5.12). No budget demonstrates a pretraining benefit.
CSP–LDA exceeds the random-mask pretrained probe by 8.39, 10.29, and 14.31
points at the respective budgets; all three paired intervals exclude zero.
Neither pretrained encoder matches the mean full-data CSP baseline at any
budget tested.

**Conclusion:** these fixed frozen embeddings do not demonstrate reduced
calibration requirements. The evidence points to the encoder/mean-pooling/readout
combination as a bottleneck relative to its pooled inputs, rather than missing
probe standardization as the explanation. It does not isolate the causal role
of the reconstruction objective, architecture, pooling, or regularization, and
does not show that all EEG pretraining methods fail. No inference here treats
sampling seeds as independently pretrained models. Confidence intervals are
exploratory and unadjusted for multiple comparisons.

Local generated report: `artifacts/results/calibration_v1/report.md`; paired subject
differences: `paired_per_subject.csv`; paired intervals: `paired_comparisons.csv`.
All 7,695 classifier fits completed. Independent output checks verified shared
sampling, nesting, run isolation, predictions against metrics, and reproduction
of 225 original full-data classical/unscaled fold results.
