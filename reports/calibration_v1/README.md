# Calibration results

Selected outputs from the exploratory calibration experiment on subjects 46–60.
Balanced accuracy is stored as a fraction in the CSV files.

- `aggregate.csv`: subject means and bootstrap confidence intervals.
- `per_subject.csv`: scores after averaging sampling seeds and run rotations.
- `paired_per_subject.csv`: matched differences for each subject.
- `paired_comparisons.csv`: mean differences and subject-bootstrap intervals.
- `calibration_curve.png`: mean scores at 10, 20, and 30 calibration trials.

See [experiment notes](../../docs/calibration.md) and the
[configuration](../../configs/calibration.yaml) for the protocol and limitations.
Full predictions, sample IDs, checkpoint hashes, and the original source hashes
are generated locally under `artifacts/results/calibration_v1/`. Those hashes
record the source used for the experiment, before later documentation cleanup.
