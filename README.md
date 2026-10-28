# From Orientation to Finals: Can a Start-of-Term Survey Predict Student Depression?

McGill INSY 674 — Group Project (Fall 2026)

## Group 1
| Name | GitHub |
|---|---|
| Maria Bssadh | @mariabssadh1-droid |
| Diana Oliván | @dianaolig8 |
| Prethela Parvin Zolok | @prethela-parvin |
| Surya Jade Edakunni | @SuryaJadeEdakunni |
| Yan-Ling Lu | @yanling-lu |
| Natasha Sujitharaj | @natasha246 |
| Cathy Vu | @cathyvvu |

## Overview
Can a short survey at the start of term identify students who will experience significant
depressive symptoms by the end of term? Using the StudentLife dataset (Dartmouth, 2013), we
predict end-of-term PHQ-9 scores (regression) and an at-risk flag, PHQ-9 ≥ 10 (classification),
from start-of-term depression, loneliness, flourishing, stress, and neuroticism surveys.

## Key findings
- **Regression:** Ridge Regression using only start-of-term PHQ-9 scores performed best, achieving a test RMSE of **2.711**, MAE of **2.051**, and R² of **0.739**. This represents a **49% reduction in RMSE** compared with the baseline mean predictor.
- **Classification:** Logistic Regression correctly identified all 8 test students (1 at risk, 7 not at risk). However, a simple rule based on initial PHQ-9 scores achieved identical results.
- **Predictor importance:** Baseline depression was the strongest predictor of end-of-term depressive symptoms. Adding loneliness, flourishing, stress, and neuroticism did not demonstrate a consistent improvement in predictive performance.
- **Conclusion:** A short start-of-term depression survey shows promise for identifying students who may need additional support. However, the small sample size (38 students, including only 7 at risk) limits the reliability and generalizability of these findings.

---
 
## 1. Frame the Problem and Look at the Big Picture
 
**Objective:** Predict (1) end-of-term PHQ-9 score (regression, primary) and (2) an at-risk flag, PHQ-9 ≥ 10 (classification, secondary), from start-of-term surveys. Regression is primary because with only 7 at-risk students, a binary label throws away most of the information; a regression model can always be thresholded at 10 for the flag. We compare three nested predictor sets (A: PHQ-9 alone, B: + loneliness/flourishing, C: + stress/neuroticism) against a simple screening rule.
 
**Who uses it:** University wellness services, as a screening aid (not a diagnosis). Students take a short survey at the start of term; flagged students get low-intensity outreach before mid-term. All decisions stay with trained staff.
 
**Current workaround:** One-off PHQ-9 ≥ 10 screening, which misses students who worsen later (missed 2 of 6 at-risk students in training). Our model is only useful if it beats this baseline.
 
**Framing:** Supervised learning; regression (primary) + binary classification (secondary); offline/batch training and prediction (one survey wave per term). Performance measures: RMSE/MAE/R² for regression, recall-prioritized precision/F1/balanced accuracy for classification, since missing an at-risk student costs more than a false alarm.
 
## 2. Explore the Data
 
EDA was run on the **training split only** (80/20 stratified split, held out first to avoid leakage).
 
- **Attributes:** 5 bounded integer scores from validated questionnaires. Missingness is low: `flourishing_pre` (10%), `stress_pre` (6.7%), `neuroticism_pre` (3.3%), imputed with the training median.
- **Distribution:** PHQ-9 is right-skewed (skew ≈ 1.5); flourishing is left-skewed; the others are roughly symmetric. A few real, clinically meaningful outliers (e.g., PHQ-9 = 23–25) are kept, since those are exactly the students a screening tool should catch.
- **Quality:** Self-report noise only (no recording errors); Cronbach's α = 0.82–0.95 across all scales.
- **Target:** End-of-term PHQ-9 (and the derived at-risk flag); the at-risk class is rare (6 of 30 training students).
- **Correlations:** PHQ-9 is the strongest predictor (r = 0.77), then stress (0.69) and neuroticism (0.61), ahead of loneliness (0.54) and flourishing (−0.43). Predictors are also correlated with each other, so feature sets are compared with cross-validation rather than by individual correlation.
- **Learnings:** A log1p transform reduces PHQ-9's skew (tested as a hyperparameter); all predictors need standardizing (different scales); extra data that would help includes a larger sample, mid-term check-ins, and passive-sensing data (StudentLife/GLOBEM).
## 3. Prepare the Data
 
Predictors and targets are separated (the at-risk label is derived from `phq9_post`, never used as a predictor, to avoid leakage). All preprocessing is wrapped in a scikit-learn `Pipeline`: optional log1p on PHQ-9 (toggleable hyperparameter), median imputation for the three predictors with missing values, and standardization on all five, fit on training data only and re-fit inside every CV fold. No outliers are removed, since the extreme scores are real students the tool should catch.
 
## 4. Measure and Compare Performance
 
Models from several families (Dummy, Linear/Ridge/Lasso, Random Forest, SVM, Logistic Regression) were compared across feature sets A/B/C with **3-fold stratified CV** (mean ± std per fold).
 
- **Regression:** Random Forest + C performed best (RMSE 4.348, R² 0.434); the other shortlisted models trailed close behind, Random Forest + B (RMSE 4.405) and Ridge + A (RMSE 4.735); all three beat the dummy baseline (RMSE 5.803).
- **Classification:** SVC + C was best (recall 1.000, precision 0.778); the other shortlisted model, Logistic Regression + C, matched its recall (1.000, precision 0.611); both beat the screening-rule baseline (recall 0.667).
- **Feature importance** (permutation method): `neuroticism_pre` ranked most important for both tasks despite PHQ-9's stronger raw correlation (multicollinearity); `loneliness_pre` was consistently least important.
- **Error analysis:** Regression over-predicted for some low-baseline students (reading high stress/neuroticism as a depression signal) and missed some who worsened; classification errors were all false positives, not false negatives. A human would likely have caught both using mid-term check-ins.
- **Feature selection check:** Dropping `loneliness_pre` left regression unchanged but dropped classification recall sharply (1.000 → 0.833), so it was kept despite its low importance score.
- **Shortlist:** Random Forest + C, Random Forest + B, and Ridge + A for regression (different error structures); SVC + C and Logistic Regression + C for classification.
## 5. Fine-Tune the System
 
Each shortlisted model was tuned with **`RandomizedSearchCV`**, searching both model hyperparameters and the imputation strategy, using 3-fold CV. Because tuning-fold scores are optimistic, every candidate was then re-evaluated on **10 × 3-fold repeated CV (30 splits)**, alongside soft/hard voting ensembles, for a stable comparison.
 
- **Best regression model:** Ridge on PHQ-9 alone, no log transform (RMSE 4.02), beating tuned Random Forest (4.21) and ensembling (4.18).
- **Best classification model:** Logistic Regression + C (recall 0.97 ± 0.13); the tuned SVC's perfect recall did not hold up on more folds (dropped to 0.83). Adding stress/neuroticism (set C) clearly helped recall; loneliness/flourishing (set B) did not.
**Final models:** Ridge + Feature Set A (regression) and Logistic Regression + Feature Set C (classification), chosen for the best, most stable repeated-CV scores.
 
**Test set (8 held-out students, evaluated once):** Ridge achieved RMSE 2.711 / MAE 2.051 / R² 0.739, a 49% RMSE reduction vs. baseline. Logistic Regression classified all 8 correctly, but so did the simple screening rule, so the test set didn't show added classification value.
 
**Limitations:**
- Only 38 students (8 in test, 1 positive case)
- CV scores may be optimistic given how many models, features, and hyperparameters were compared
- Self-reported, single-cohort data limits generalizability
- This is a pilot proof of value, not a validated screening system

---

## Repository structure
- `INSY 674-Group Project.ipynb`: full analysis, organized by the ML project checklist:
  1. Frame the Problem · 2. Get the Data · 3. Explore the Data · 4. Prepare the Data 
  5. Measure and Compare Performance · 6. Fine-Tune the System

## How to run
1. `pip install pandas scikit-learn matplotlib seaborn "kagglehub[pandas-datasets]"`
2. Open the notebook and run all cells. Data downloads automatically from Kaggle.

## Data
StudentLife dataset: https://www.kaggle.com/datasets/dartweichen/student-life
(survey files: PHQ-9, UCLA Loneliness, Flourishing, Perceived Stress, Big Five)

## Limitations
Small pilot sample (38 students, 7 at risk); results require validation at larger scale.

## References
Cohen, S., Kamarck, T., & Mermelstein, R. (1983). A global measure of perceived stress. *Journal of Health and Social Behavior, 24*(4), 385–396. https://doi.org/10.2307/2136404

Dartmouth College StudentLife Project. (2014). *StudentLife dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/dartweichen/student-life

Diener, E., Wirtz, D., Tov, W., Kim-Prieto, C., Choi, D., Oishi, S., & Biswas-Diener, R. (2010). New well-being measures: Short scales to assess flourishing and positive and negative feelings. *Social Indicators Research, 97*(2), 143–156. https://doi.org/10.1007/s11205-009-9493-y

Healthy Minds Network. (2025). *The Healthy Minds Study: 2024–2025 data report*. University of Michigan, University of California Los Angeles, Boston University, & Wayne State University. https://healthymindsnetwork.org/wp-content/uploads/2025/09/2024-2025_HMS-National-Data-Report_Student.pdf

John, O. P., & Srivastava, S. (1999). The Big Five trait taxonomy: History, measurement, and theoretical perspectives. In L. A. Pervin & O. P. John (Eds.), *Handbook of personality: Theory and research* (2nd ed., pp. 102–138). Guilford Press.

Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). The PHQ-9: Validity of a brief depression severity measure. *Journal of General Internal Medicine, 16*(9), 606–613. https://doi.org/10.1046/j.1525-1497.2001.016009606.x

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Russell, D. W. (1996). UCLA Loneliness Scale (Version 3): Reliability, validity, and factor structure. *Journal of Personality Assessment, 66*(1), 20–40. https://doi.org/10.1207/s15327752jpa6601_2

Wang, R., Chen, F., Chen, Z., Li, T., Harari, G., Tignor, S., Zhou, X., Ben-Zeev, D., & Campbell, A. T. (2014). StudentLife: Assessing mental health, academic performance and behavioral trends of college students using smartphones. In *Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '14)* (pp. 3–14). ACM. https://doi.org/10.1145/2632048.2632054 (open-access PDF: https://www.cs.columbia.edu/~xia/publication/ubicomp14-studentlife/ubicomp14-studentlife.pdf)

Xu, X., et al. (2022). GLOBEM dataset: Multi-year datasets for longitudinal human behavior modeling generalization. In *Advances in Neural Information Processing Systems 35: Datasets and Benchmarks Track*. Dataset overview: https://the-globem.github.io/datasets/overview

Xu, X., Liu, X., Zhang, H., Wang, W., Nepal, S., Kuehn, K. S., Huckins, J. F., Morris, M. E., Nurius, P. S., Riskin, E. A., Patel, S., Althoff, T., Campbell, A., Dey, A. K., & Mankoff, J. (2023). GLOBEM: Cross-dataset generalization of longitudinal human behavior modeling. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 6*(4), Article 190. https://homes.cs.washington.edu/~althoff/docs/globem_imwut.pdf

## Mental health resources
If you or someone you know is struggling, support is available:

- **McGill Student Wellness Hub:** https://www.mcgill.ca/wellness-hub · 514-398-6017
- **McGill 24/7 resources** (including GuardMe / TELUS Health Student Support, 1-844-451-9700, and McGill Students' Nightline, 514-398-6246): https://www.mcgill.ca/wellness-hub/get-support/resources-available-247
- **Suicide Crisis Helpline:** call or text 9-8-8 (24/7, across Canada)
- **Suicide prevention, Québec:** 1-866-APPELLE (1-866-277-3553) · text 535353 · https://suicide.ca
- **Info-Social:** 811, option 2 (24/7, Québec)
- **In an emergency, call 911.**
