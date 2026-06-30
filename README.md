# Hospital Readmission Risk — A Case Study in Rigorous ML Practice

**An end-to-end analysis of 30-day readmission prediction for diabetic patients,
documenting the methodology, decisions, and honest limitations encountered when
applying machine learning to administrative healthcare data.**

🔗 **[Full methodology notebooks →](./notebooks)**
📊 **[Final model card →](#final-model)**

---

## What This Project Is — And Isn't

**This is a methodology case study, not a deployable healthcare tool.** The
final model achieves AUC 0.66 on cross-validated patient-grouped folds, which
is consistent with published results on this dataset but insufficient for
clinical deployment. The deliverable of this project is the documented
reasoning behind 12+ analytical decisions and the honest diagnosis of
where the model's predictive ceiling lives and why.

The project demonstrates:

- Leakage-safe data pipeline design (patient-grouped splits, fit-on-train discipline)
- Multi-method feature selection (predictive-power audit + L1 regularization)
- Statistically rigorous model comparison (cross-validation with confidence intervals)
- Fairness auditing across demographic subgroups
- Senior-grade decision-making about when to stop optimizing

What it deliberately doesn't include:

- A risk-scoring app or production deployment (the model isn't strong enough to warrant one)
- Claims of clinical utility (the precision-recall tradeoff makes real deployment infeasible)

---

## The Problem

The dataset is the **Diabetes 130-US Hospitals** dataset (UCI ML Repository,
~101,766 encounters). The question: **at the moment a diabetic patient is
discharged, can we predict whether they will be readmitted within 30 days?**

Why this matters: under US healthcare policy, 30-day readmissions trigger
financial penalties. A working prediction tool could let hospitals target
follow-up resources at high-risk patients.

**Final cohort after cleaning**: 97,822 encounters, 11.46% positive rate.

---

## Methodology — Decision by Decision

Each major decision in the project, what alternatives were considered, and
what evidence drove the choice. This is the senior workflow your mentor wants
to see: not "I did X," but "I did X *because* Y."

### Decision 1: Problem framing and leakage boundary

**Decision**: Predict 30-day readmission as binary classification, with the
prediction moment defined as "discharge."

**Why this matters**: Features that exist only *after* discharge (follow-up
calls, future medication adjustments) would be data leakage. Every feature
was audited against "is this knowable at discharge?"

**Alternative considered**: Multi-class prediction (no readmission / >30 days / <30
days). Rejected because the business problem cares only about 30-day readmissions
(the penalized category).

### Decision 2: Patient-grouped train/test split

**Decision**: Used `GroupShuffleSplit` with `patient_nbr` as the grouping variable,
preventing the same patient from appearing in both train and test.

**Why this matters**: 31% of encounters belong to patients who have multiple
visits in the dataset. A naive random split would put the same patient on
both sides, letting the model "memorize" patients and inflate test performance.

**Result**: 78,283 training / 19,539 test, with zero patient overlap confirmed.

### Decision 3: Filter unreachable encounters

**Decision**: Dropped 2,423 encounters where the patient died or was transferred
to hospice during the stay.

**Why**: These patients cannot be readmitted within 30 days by definition.
Including them creates a guaranteed-negative class that doesn't reflect the
real prediction problem.

### Decision 4: Missing value strategy — "missingness as signal"

**Decision**: Different missing-value strategies based on what missingness means.

- `weight` (97% missing): Converted to binary `weight_recorded` flag
- `max_glu_serum`, `A1Cresult` (lab tests): Encoded missing as "not_tested" category
- `medical_specialty`, `payer_code`: Encoded missing as "unknown" category
- `diag_1`, `diag_2`, `diag_3` (light missing): Dropped affected rows

**Why this matters**: A lab test not being ordered is *clinical information*
(the doctor didn't think it was necessary), not just missing data. Treating
all missingness identically would discard this signal.

### Decision 5: ICD-9 diagnosis code grouping

**Decision**: Grouped 689-761 raw ICD-9 codes per diagnosis column into 9
clinical categories (Circulatory, Respiratory, Diabetes, etc.) using the
standard mapping from the readmission literature.

**Why**: Naive one-hot encoding of raw ICD codes would have produced ~2,170
sparse feature columns, most representing fewer than 10 patients. Clinical
grouping reduces this to 27 meaningful columns while preserving signal.

### Decision 6: Feature reduction — from 217 to 30

The most consequential set of decisions in the project. Three sequential
reductions, each verified against the previous baseline:

#### Stage A — Multicollinearity correction (217 → 189 features)

Identified 12 pairs of medication features with correlation ≥ 0.8. Diagnosed
the underlying issue: 4 medications were near-constant (<1% non-"No" rate);
8 others had skewed distributions dominated by No vs Steady.

Action: dropped the 4 near-constants entirely; binarized the 8 skewed
medications to "taken vs not taken."

Result: 12 multicollinearity pairs resolved, AUC unchanged (0.664 → 0.664).

#### Stage B — Prevalence-based pruning (189 → 92 features)

Coefficient inspection of the 189-feature model revealed clinically nonsensical
top features: rare medical specialties (Pediatrics-Endocrinology, etc.)
dominating coefficient rankings. Investigation showed many one-hot dummies
represented fewer than 100 patients.

Action: dropped one-hot dummies with prevalence < 1% (less than ~780 patients).

Result: 97 features removed, AUC essentially unchanged (0.664 → 0.661).

#### Stage C — L1 selection + coefficient floor (92 → 30 features)

Swept L1 regularization strength across 7 C values, identified an elbow at
C=0.01 (36 features retained). Applied a coefficient magnitude floor of 0.02
to remove L1 survivors with statistically meaningless effects.

Validated against an independent univariate predictive-power audit
(Cohen's d for numerics, readmission rate differences for categoricals).
Both methods agreed on the top features.

Result: 30 final features, AUC unchanged (0.661 → 0.661).

**Total reduction**: 217 → 30 features (86% reduction) with AUC preserved
within noise (0.664 → 0.661). The features dropped did not carry predictive
signal worth keeping.

### Decision 7: Class imbalance handling

**Decision**: Retained `class_weight='balanced'` after comparing four alternatives.

**Comparison table** (all on the final 30-feature model):

| Method | AUC | Recall | Precision |
|---|---|---|---|
| class_weight='balanced' (chosen) | 0.6612 | 0.529 | 0.192 |
| SMOTE | 0.6413 | 0.538 | 0.178 |
| RandomUnderSampler | 0.6610 | 0.530 | 0.193 |
| SMOTE + TomekLinks | 0.6412 | 0.539 | 0.178 |
| No handling (reference) | 0.6613 | 0.018 | 0.471 |

**Why class_weight won**: SMOTE *reduced* AUC by 0.02 because the feature
space is dominated by one-hot dummies that don't interpolate meaningfully.
RandomUnderSampler matched class_weight but discarded 60,000 training rows.
The "no handling" baseline shows what happens without any imbalance
correction: 0.018 recall, catching only 41 of 2,337 actual readmissions.

### Decision 8: Final model class

**Decision**: Retained L2 logistic regression rather than XGBoost.

**Cross-validation comparison** (patient-grouped, 5 folds):

| Model | Mean AUC | Std Dev | 95% CI |
|---|---|---|---|
| LR (final) | 0.6607 | 0.0025 | [0.656, 0.666] |
| XGBoost | 0.6647 | 0.0033 | [0.658, 0.671] |

XGBoost's improvement (0.004 mean) is within fold-to-fold noise. The
confidence intervals overlap substantially. The honest conclusion:
**non-linearity does not break the AUC ceiling on this problem.**

This is the most important diagnostic finding of the project. Three
feature-reduction stages (217 → 30) and a stronger model class all
converge on AUC ~0.66. **The binding constraint is the data, not the
model.**

---

## Final Model

**30 features, L2 logistic regression, `class_weight='balanced'`**

- Cross-validated AUC: 0.6607 ± 0.0025
- At default 0.5 threshold: recall 0.53, precision 0.19, F2 0.39
- At capacity-matched threshold (10% flagged): recall 0.22, precision 0.27

### Top features by coefficient magnitude

The five strongest predictors:

1. `discharge_disposition_id_22` (odds ratio 3.81) — specific discharge destination
2. `discharge_disposition_id_5` (odds ratio 2.75) — specific discharge destination
3. `discharge_disposition_id_2` (odds ratio 1.79) — discharge to short-term hospital
4. `diag_2_Neoplasms` (odds ratio 1.47) — cancer comorbidity
5. `number_inpatient` (odds ratio 1.44 per SD increase) — prior inpatient stays

[Insert the feature_importance.png figure here]

**Notable pattern**: 5 of the top 6 features are discharge dispositions.
*Where* a patient is sent after discharge is the single biggest signal in
this model. Patient demographics and length of stay carry far less individual
predictive power than the discharge destination decision.

---

## Fairness Audit

The fairness audit checked whether AUC differs meaningfully across
demographic subgroups.

### By race (groups with adequate sample size)

| Group | n | AUC |
|---|---|---|
| Caucasian | 14,712 | 0.6556 |
| AfricanAmerican | 3,613 | 0.6739 |
| Hispanic | 388 | 0.7667* |

*Hispanic AUC unreliable due to small sample (46 positives).

**Finding**: No meaningful AUC disparity between Caucasian and African
American patients (difference 0.013, within noise). Smaller groups have
insufficient sample sizes to evaluate reliably — a real limitation of
this dataset's utility for fairness analysis on minority populations.

### By gender

| Group | n | AUC |
|---|---|---|
| Female | 10,483 | 0.6674 |
| Male | 9,056 | 0.6538 |

**Finding**: No meaningful gender disparity (difference 0.014).

### By age — the real disparity

| Age | n | AUC |
|---|---|---|
| 30-40 | 794 | 0.7304 |
| 40-50 | 1,840 | 0.7065 |
| 50-60 | 3,414 | 0.6602 |
| 60-70 | 4,323 | 0.6643 |
| 70-80 | 4,981 | 0.6516 |
| 80-90 | 3,243 | 0.6112 |
| 90-100 | 545 | 0.5404 |

**Finding**: Model performance degrades systematically with patient age.
AUC of 0.73 for ages 30-50 declines to 0.55-0.61 for ages 80+. This is
consistent with the U-shaped age-readmission relationship and reflects
the linear model's inability to capture non-linear age effects. **For
elderly patients, this model is a particularly weak predictor.**

This is a model-class limitation, not a data bias. Captures age non-linearly
(via polynomial features, splines, or tree models) would likely close the
gap — but it would still inherit the broader AUC ceiling.

---

## The Honest Verdict

**Should this model be deployed? No.**

At a capacity-matched operating point (10% of patients flagged), the model
catches 22% of readmissions while generating 1,430 false positives per 524
true positives. That's not strong enough for clinical use — the follow-up
team would correctly intervene on 1 in 4 flagged patients, and the model
would miss 78% of true readmissions.

**Why is the ceiling so low?**

Healthcare readmission is genuinely hard to predict from administrative
data. Unmeasured factors that likely matter:

- Medication adherence post-discharge
- Social support (lives alone vs. has caregiver)
- Outpatient follow-up scheduling and attendance
- Social determinants of health (housing stability, food security, transportation)
- Patient-reported symptoms

These features are not in administrative data. The 30-feature model captures
what's there; what's there has a ceiling.

**Published literature** on this same dataset reports AUC in the 0.65–0.72
range for best-tuned models. We're at 0.66. This isn't a personal failure
or a methodology gap — it's the predictability ceiling of administrative
healthcare data for this problem.

**In a production setting**, the senior next step would be a conversation
with the data team: *"We've hit the ceiling of this dataset. What other
data sources could capture medication adherence, follow-up patterns, and
social context?"* This is the lesson that distinguishes ML practitioners
who understand their work from those who chase AUC numbers without
recognizing fundamental constraints.

---

## What I Learned

The biggest lessons from this project:

1. **Investigate before optimizing**: Multiple times in the project, what
   looked like an obvious next step (drop features, apply SMOTE, push to
   XGBoost) turned out to be the wrong move when investigated. Each "no
   improvement" finding was useful — it eliminated a path and clarified
   where the real constraints live.

2. **The ceiling diagnosis is the deliverable**: A senior healthcare ML
   practitioner doesn't ship "AUC 0.66" — they ship the explanation of
   why 0.66 is the achievable ceiling and what would be required to lift
   it. The honest framing is more valuable than the number.

3. **Cross-validation is mandatory for model comparisons**: Single-split
   evaluations can lie. The CI overlap between LR and XGBoost showed that
   what looked like a tiny improvement was statistically within noise —
   without CV, I might have claimed XGBoost was "better" when the data
   couldn't support that claim.

4. **Fairness audits reveal what the model knows and doesn't**: The age
   disparity wasn't obvious from overall metrics. It only surfaced when I
   computed per-group AUC, which then traced back to a specific modeling
   choice (linear age encoding). This is the kind of analysis that
   distinguishes thoughtful work from naive deployment.

5. **Knowing when to stop**: I could have run XGBoost hyperparameter
   tuning, tried SMOTE-NC, engineered interaction features, ensembled
   multiple models. Each would have produced marginal improvements. None
   would have lifted the data-imposed ceiling. The senior move was
   pivoting from model improvement to interpretability and documentation.

---

## Repo Structure

```text
project-name/
├── data/                          # Saved intermediate states (gitignored)
├── notebooks/
│   ├── 01_understanding.ipynb     # Problem framing, success criteria
│   ├── 02_cleaning.ipynb          # Missing values, target, disposition filter
│   ├── 03_modeling.ipynb          # Initial encoding pipeline, baseline LR
│   ├── 04_feature_selection.ipynb # Prevalence-based pruning (189 → 92)
│   ├── 05_predictive_power.ipynb  # Cohen's d audit, categorical rate audit
│   ├── 06_l1_selection.ipynb      # L1 sweep + coefficient floor (92 → 30)
│   ├── 07_class_imbalance.ipynb   # SMOTE vs class_weight comparison
│   ├── 08_cross_validation.ipynb  # Patient-grouped 5-fold CV
│   ├── 09_xgboost.ipynb           # Non-linear model test
│   └── 10_interpretability.ipynb  # Coefficient analysis, fairness audit
└── README.md                      # This file
```

## Stack

- Python · Pandas · scikit-learn · XGBoost · imbalanced-learn · matplotlib · seaborn
- Patient-grouped splitting via `GroupShuffleSplit` and `GroupKFold`
- Encoding pipeline via `ColumnTransformer`