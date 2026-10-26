# From Orientation to Finals: Can a Start-of-Term Survey Predict Student Depression?

McGill INSY 674 — Group Project (Fall 2026)

## Group 1
| Name | GitHub |
|---|---|
| Maria Bssadh | @mariabssadh1-droid |
| Diana Oliván | @dianaolig8 |
| Prethela Parvin Zolok | @prethela-parvin |
| Surya Edakunni | @... |
| Yan-Ling Lu | @yanling-lu |
| Natasha Sujitharaj | @... |
| Cathy Vu | @cathyvvu |

## Overview
Can a short survey at the start of term identify students who will experience significant
depressive symptoms by the end of term? Using the StudentLife dataset (Dartmouth, 2013), we
predict end-of-term PHQ-9 scores (regression) and an at-risk flag, PHQ-9 ≥ 10 (classification),
from start-of-term depression, loneliness, flourishing, stress, and neuroticism surveys.

## Key findings
- [To be completed after modeling: best model, CV performance vs. baseline, test results]

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
