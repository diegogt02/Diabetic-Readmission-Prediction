# Diabetic Readmission Prediction

Binary classification of hospital readmission for diabetic patients, comparing seven
classical machine learning models end-to-end: EDA, preprocessing, model selection,
hyperparameter tuning, and evaluation on a held-out test set.

Coursework for *Applied Data Analysis and Machine Learning* (Master's in Complex
Systems Physics, UIB), adapted here as a standalone portfolio project.

## Data

5,000 encounter records for diabetic patients, a subset of the dataset described in:

> B. Strack, J. P. DeShazo, C. Gennings, J. L. Olmo, S. Ventura, K. J. Cios, and J. N.
> Clore, "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of
> 70,000 Clinical Database Patient Records," *BioMed Research International*, vol.
> 2014, Article ID 781670, 2014.

38 columns mixing demographics (race, gender, age range), encounter information
(admission/discharge codes, length of stay, procedures, medications), ICD-9 diagnosis
codes, and diabetes-specific medication indicators. Target: `readmitted` (yes/no).

## Method

- Exploratory analysis of class balance, missing values, and feature distributions.
- Preprocessing pipeline (`scikit-learn` `ColumnTransformer`): standard scaling for
  numeric features, ordinal encoding for the age bucket, one-hot encoding for
  nominal categoricals.
- Seven models compared via cross-validated ROC-AUC: dummy baseline, logistic
  regression, k-NN, SVM, decision tree, random forest, gradient boosting.
- Hyperparameter tuning (`GridSearchCV`) on the two strongest candidates.
- Final evaluation on a held-out test set + impurity-based feature importance.

## Results

The tuned random forest (`n_estimators=400`, `max_depth=10`, `min_samples_leaf=1`) was
the best model, with cross-validated ROC-AUC 0.679. On the untouched test set (1,000
patients) it reached **ROC-AUC 0.696, accuracy 0.64** — consistent with the
cross-validation estimate, so no sign of overfitting. It correctly flags 80% of
non-readmitted patients but only ~45% of patients who are actually readmitted,
typical when the available features only partially explain the outcome.

The most predictive features are all measures of prior hospital utilisation —
`number_inpatient`, `num_lab_procedures`, `num_medications`, `number_emergency`,
`number_diagnoses`, `number_outpatient`, `time_in_hospital` — together with `age`.
Clinically intuitive: patients with heavier recent hospital use and more diagnoses are
the most likely to be readmitted.

Full write-up with figures: [`reports/report.pdf`](reports/report.pdf).

## Repository structure

```
notebooks/   readmission_classification.ipynb   # full analysis, run top to bottom
data/        diabetic_readmission.csv            # the 5,000-patient subset
reports/     report.pdf, report.tex, figures/    # write-up and generated figures
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/readmission_classification.ipynb
```

## License

MIT — see [LICENSE](LICENSE).
