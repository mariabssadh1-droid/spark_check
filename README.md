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
