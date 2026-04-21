# early-literacy-risk-classifier

> A supervised ML pipeline that identifies **children at risk for reading difficulties by Grade 4** — using kindergarten-entry cognitive and demographic features — with full SHAP interpretability so researchers and educators can understand *why* each child was flagged.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![sklearn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-red)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Motivation

In Dr. Jessica Logan's research at Vanderbilt's Peabody College, a central question is: **which children are at risk for learning disabilities, and what factors drive that risk?**

Traditional statistical approaches model average effects. This project brings supervised machine learning to the same question — specifically **Random Forest** and **Gradient Boosting** classifiers — to predict which children are in the bottom 20th percentile of reading achievement by Grade 4, using only information available at kindergarten entry.

Crucially, this project treats **interpretability as non-negotiable**. Using SHAP (SHapley Additive exPlanations), every prediction is explained at both the global (population) and individual (child) level — making the model useful to education researchers, not just data scientists.

---

## Key Research Question

> *Can kindergarten-entry cognitive and demographic features predict which children will struggle most with reading by Grade 4? And which features drive the highest risk?*

---

## ML Pipeline Overview

```
Raw ECLS-K style data
        │
        ▼
Feature Engineering
  ├── Compound disadvantage index (SES × parental education)
  ├── Cognitive composite (vocab + phonological + working memory)
  ├── SES × Vocabulary interaction
  ├── Phonological × Special Ed interaction
  └── Classroom risk score (class size + teacher inexperience)
        │
        ▼
Risk Label Construction
  └── Bottom 20th percentile on Grade 4 reading IRT = at-risk
        │
        ▼
Preprocessing (StandardScaler + Imputation)
        │
        ▼
5-Fold Stratified Cross-Validation
  ├── Random Forest ✓
  ├── Gradient Boosting
  └── Logistic Regression (interpretable baseline)
        │
        ▼
Test Set Evaluation
  ├── ROC-AUC
  ├── Precision-Recall (handles class imbalance)
  └── Confusion Matrix
        │
        ▼
SHAP Interpretability
  ├── Beeswarm (global feature importance)
  ├── Bar plot (mean |SHAP|)
  └── Waterfall (individual child explanation)
```

---

## Project Structure

```
early-literacy-risk-classifier/
│
├── data/
│   └── synthetic_ecls.csv          # Auto-generated synthetic data
│
├── src/
│   ├── features.py                 # Feature engineering + preprocessing
│   ├── model.py                    # Training, evaluation, SHAP
│   └── run_classifier.py           # End-to-end runner
│
├── outputs/
│   ├── test_evaluation.png         # ROC, PR curve, confusion matrix
│   ├── shap_beeswarm.png           # Global SHAP summary
│   ├── shap_bar.png                # Mean |SHAP| bar chart
│   ├── shap_waterfall_high_risk.png # Single child explanation
│   ├── cv_results.csv              # Cross-validation scores
│   └── feature_importance_shap.csv # Ranked feature importances
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## Setup & Usage

```bash
git clone https://github.com/0xafraidoftime/early-literacy-risk-classifier.git
cd early-literacy-risk-classifier
pip install -r requirements.txt

cd src
python run_classifier.py
```

---

## Key Features

| Feature | Description | Rationale |
|---|---|---|
| `vocab_baseline` | Vocabulary at kindergarten entry | Strong predictor of reading trajectory |
| `phonological` | Phonological awareness score | Core mechanism for reading acquisition |
| `ses_quintile` | Socioeconomic status composite | Well-established risk factor |
| `special_ed_flag` | IEP / special ed services | Signals existing identified need |
| `cognitive_composite` | Combined cognitive measure | Captures overall school readiness |
| `disadvantage_index` | Low SES + low parent edu | Compound disadvantage |
| `ses_x_vocab` | SES × vocabulary interaction | Does vocab protection vary by SES? |
| `phonological_x_sped` | Phonological × special ed | Compounding phonological risk |
| `class_risk` | Large class + inexperienced teacher | Environmental risk |

---

## Why SHAP?

Accuracy alone is insufficient for education research applications:

- **Global SHAP** → Which features matter most *across all children*?
- **Individual SHAP (waterfall)** → *Why* was this specific child flagged as at-risk?
- **SHAP interactions** → Does the effect of vocabulary differ by SES level?

This bridges the gap between ML performance and the *mechanistic understanding* that developmental researchers need.

---

## References

- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*.
- Logan, J. A. R. (research on individual differences in child development, Vanderbilt University)
- Catts, H. W., et al. (2001). Estimating the risk of future reading difficulties in kindergarten children. *Language, Speech, and Hearing Services in Schools*.

---

## Author

**0xafraidoftime** — [GitHub](https://github.com/0xafraidoftime)

Inspired by Dr. Jessica Logan's work at Vanderbilt University's Peabody College on applying advanced quantitative methods to understand individual differences in children's academic development.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
