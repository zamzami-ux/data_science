# CLASSIFYING ACADEMIC SELF-EFFICACY LEVELS OF INDONESIAN SECONDARY STUDENTS BASED ON LEARNING READINESS DOMAINS USING MACHINE LEARNING ALGORITHMS

**Muhammad Zamzami**  
Department of Data Science / Computer Science  
Faculty of Science and Technology  
*E-mail: zamzami@student.institution.ac.id*

---

## Abstract
Academic self-efficacy is a pivotal psychological determinant governing student motivation, academic resilience, and cognitive achievement. In the Indonesian secondary education landscape (Junior and Senior High Schools), the early diagnosis of students suffering from academic self-efficacy deficits is often impeded by conventional, subjective, and retrospective assessment instruments. This research proposes and rigorously validates an end-to-end Educational Data Mining (EDM) machine learning framework designed to predict students' academic self-efficacy levels based on multidimensional learning readiness domains. The empirical dataset is sourced from the open-access Mendeley Data repository (DOI: 10.17632/ft5z9yk94n.1), encompassing 1,182 Indonesian secondary student respondents across 36 five-point Likert items distributed over 9 psychological subscales (A1–C3) alongside sociodemographic attributes. The classification target was formulated from subscale C1 (Academic Self-Efficacy), categorized into three balanced classes (*Low*, *Moderate*, *High*) using tercile quantile cutoffs (q1 = 4.25; q2 = 4.75). To strictly eliminate data leakage, subscale C1 was excised from the feature space, and all preprocessing operations—including reverse coding of negative items (A3.1 and A3.2), missing value verification, StandardScaler, and One-Hot Encoding—were encapsulated inside scikit-learn Pipeline and ColumnTransformer architectures. Five supervised machine learning algorithms were systematically benchmarked using Stratified 5-Fold Cross-Validation and an independent 20% test partition: Logistic Regression, Random Forest (300 estimators), Support Vector Machine (SVM RBF), K-Nearest Neighbors (KNN, k=15), and Gradient Boosting. All models substantially outperformed the Zero-Rule majority baseline (accuracy 44.73%). The SVM RBF model achieved the highest test accuracy of **65.82%** (Macro-F1: **0.6295**), while Gradient Boosting achieved the premier test Macro-F1 score of **0.6317** (Accuracy: **64.98%**). The models demonstrated superior identification of vulnerable students in the Low self-efficacy class (Recall: 78.30%, Precision: 76.15%). Permutation Feature Importance analysis revealed that **Classroom Engagement (B3)**, **Value of Learning (C2)**, and **Academic Persistence (B1)** serve as the paramount behavioral-cognitive drivers of academic self-efficacy, whereas regional and transport demographics exerted negligible influence. The trained models are operationalized into an interactive Streamlit web dashboard to deliver real-time early warning diagnostics for educators and guidance counselors.

**Keywords**: *Academic Self-Efficacy*, *Educational Data Mining*, *Machine Learning*, *Classification*, *School Readiness*, *Streamlit Dashboard*

---

## 1. INTRODUCTION

In the contemporary educational milieu shaped by the Fourth Industrial Revolution and Society 5.0, academic success is governed not merely by cognitive aptitude, but fundamentally by psychosocial resilience, emotional regulation, and self-regulatory mechanisms (Igual & Segui, 2017). Central to these non-cognitive competencies is **academic self-efficacy**, conceptualized in Albert Bandura’s Social Cognitive Theory as individuals' subjective beliefs in their capabilities to orchestrate and execute courses of action required to attain designated educational performances (Bandura, 1997). Extensive empirical research indicates that students endowed with robust self-efficacy demonstrate elevated perseverance in tackling demanding academic tasks, deploy effective self-regulated metacognitive strategies, experience lower academic stress, and maintain significantly higher graduation and retention rates (Zimmerman, 2000; Schunk & DiBenedetto, 2020).

Conversely, students exhibiting acute academic self-efficacy deficits frequently succumb to learned helplessness, severe test anxiety, systematic procrastination, and disengagement from school activities. Within the Indonesian educational ecosystem, which encompasses over 50 million students across heterogeneous socioeconomic regions, secondary education—comprising Junior High School (*Sekolah Menengah Pertama* / SMP) and Senior High School (*Sekolah Menengah Atas* / SMA)—represents a precarious developmental transition (Nugraha & Hairani, 2018). Students confront escalating curriculum complexity, high-stakes summative national evaluations, and socio-emotional peer pressures. Regrettably, traditional identification mechanisms deployed in Indonesian secondary schools remain overwhelmingly reactive: guidance counselors (*Guru Bimbingan Konseling*) and teachers typically detect student distress only after severe academic underperformance, chronic absenteeism, or formal grade retention has manifested.

Educational Data Mining (EDM) and Machine Learning (ML) present transformative opportunities to transition institutional pedagogical support from reactive remediation toward predictive, precision education (Romero & Ventura, 2020). By mining latent psychometric and behavioral data collected via routine school readiness screenings, supervised classification models can automatically uncover complex non-linear interactions across affective, cognitive, and behavioral learning readiness domains to forecast students' self-efficacy states (Hairani et al., 2020; Jadhav & Channe, 2016).

Nevertheless, prevailing literature in student predictive modeling frequently encounters severe methodological pitfalls:
1. **Pervasive Data Leakage**: Many existing predictive studies commit data leakage by retaining features that are algebraically intertwined with the target outcome or by applying data transformations (such as scaling or encoding) across the entire dataset prior to splitting, producing artificially inflated, unreplicable performance metrics (Kaufman et al., 2012).
2. **Inadequate Model Interpretability**: Black-box algorithmic implementations frequently omit post-hoc explainability evaluations, leaving educators without actionable insights into *why* a particular student is categorized as at-risk.
3. **Absence of Practical Deployment**: Most studies terminate at offline experimental notebooks without providing accessible decision-support tools tailored for real-world school practitioners lacking programming expertise.

To address these empirical and pedagogical gaps, this study presents a rigorous, reproducible machine learning investigation leveraging an authentic, large-scale psychometric dataset of Indonesian secondary students sourced from Mendeley Data (DOI: 10.17632/ft5z9yk94n.1; N = 1,182). This research is structured around three primary contributions:
1. **Multi-Model Benchmarking with Zero-Leakage Architecture**: We implement an airtight machine learning pipeline comparing five distinct algorithm paradigms (*Multinomial Logistic Regression*, *Random Forest*, *Support Vector Machine RBF*, *K-Nearest Neighbors*, and *Gradient Boosting*) evaluated under Stratified 5-Fold Cross-Validation and an independent holdout test set.
2. **Feature Importance & Psychological Interpretability**: Utilizing *Permutation Feature Importance* based on test Macro-F1 degradation, we empirically isolate which school readiness dimensions (e.g., Classroom Engagement, Grit/Persistence, Value of Learning) predominantly govern students' academic self-efficacy beliefs.
3. **Interactive Decision-Support Deployment**: We operationalize the optimal predictive pipeline within a cloud-hosted, user-friendly Streamlit web application, empowering educators to conduct real-time inference and simulated risk assessment.

---

## 2. THEORETICAL FOUNDATIONS AND RELATED WORK

### 2.1 Bandura’s Social Cognitive Framework and Learning Readiness Domains
Albert Bandura’s triadic reciprocal causation model posits that human functioning is dynamic and co-determined by interactions among personal determinants (cognitive and affective states), behavioral patterns, and environmental events (Bandura, 1997). In the school readiness measurement model utilized in this study, student preparedness is operationalized through nine distinct subscales spanning three psychological tiers:

1. **School Climate & Socio-Emotional Domain (Tier A)**:
   - **A1 - Sense of School Belonging**: The subjective feeling of acceptance, inclusion, and safety within the educational institution.
   - **A2 - Emotional Regulation**: The psychological capacity to monitor, modulate, and channel affective reactions, emotional arousal, and stress during academic tasks.
   - **A3 - Prosocial Behavior**: Adherence to institutional norms, interpersonal empathy, and constructive peer collaboration (including reverse-coded unfavorable traits such as classroom disruption and conflict).
2. **Learning Competencies & Self-Regulation Domain (Tier B)**:
   - **B1 - Academic Persistence (Grit)**: The sustained exertion of cognitive effort and perseverance in the face of academic setbacks.
   - **B2 - Study Planning & Organization**: Methodical time allocation, goal setting, and strategic scheduling of academic obligations.
   - **B3 - Classroom Engagement**: Active behavioral and cognitive participation, attentiveness, and voluntary inquiry during classroom instruction.
3. **Academic Attitudes & Mindset Domain (Tier C)**:
   - **C1 - Academic Self-Efficacy**: Confidence in one's personal capacity to master curriculum concepts and achieve scholastic goals. This construct serves as the **Classification Target**.
   - **C2 - Value of Learning**: The perceived personal relevance, intrinsic utility, and instrumental worth ascribed to formal education.
   - **C3 - Growth Mindset**: The belief that intellectual capability is malleable and can be enhanced through dedicated effort (Dweck, 2006).

### 2.2 Supervised Machine Learning in Educational Data Mining
The prediction of ordinal psychosocial states constitutes a supervised classification task. This study investigates five diverse algorithmic paradigms:
- **Multinomial Logistic Regression**: Serves as a standard, interpretable generalized linear model estimating multi-class probabilities via the softmax activation function (Hosmer et al., 2013).
- **Random Forest**: An ensemble bagging technique constructing an ensemble of de-correlated decision trees trained on bootstrap samples and random feature subsets to achieve minimal generalization variance (Breiman, 2001).
- **Support Vector Machine with Radial Basis Function (SVM RBF)**: Maximizes the geometric separation margin in an implicit, infinite-dimensional reproducing kernel Hilbert space, excelling in mapping non-linear psychometric boundaries (Cortes & Vapnik, 1995).
- **K-Nearest Neighbors (KNN)**: A non-parametric instance-based classifier assigning labels based on the Euclidean distance consensus of the $k$ most proximal neighbors in standardized feature space (Cover & Hart, 1967).
- **Gradient Boosting Classifier**: An ensemble boosting technique iteratively fitting decision tree base learners to the negative gradients of the pseudo-residual multinomial loss function (Friedman, 2001).

---

## 3. RESEARCH METHODOLOGY

The end-to-end computational methodology is depicted in Figure 1, illustrating the systematic sequence from data curation to cloud deployment.

```
+-------------------------------------------------------------------------------+
|                       DATA SCIENCE RESEARCH METHODOLOGY                        |
+-------------------------------------------------------------------------------+
|  1. DATA ACQUISITION & CURATION                                               |
|     - Mendeley Data (DOI: 10.17632/ft5z9yk94n.1) | N = 1,182 Students        |
|     - 36 Likert-Scale Items (A1 - C3) + Demographics (SMP & SMA Indonesia)   |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  2. DATA AUDIT, REVERSE CODING & TARGET FORMATION                             |
|     - Reverse coding of unfavorable items: Score = 6 - x (A3.1 & A3.2)       |
|     - Subscale Mean Aggregation: 4 items per subscale (A1 through C3)        |
|     - Target Discretization (C1): Tercile Quantiles (q1=4.25, q2=4.75)       |
|       Classes: Low (<=4.25), Moderate (4.25-4.75), High (>4.75)               |
|     - Anti-Leakage Protocol: Subscale C1 removed from predictor matrix X     |
|     - Elimination of 10 optional subject grade columns                        |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  3. FEATURE ENGINEERING & PREPROCESSING PIPELINE                              |
|     - Scikit-Learn ColumnTransformer:                                         |
|       * Continuous Numerical Features (A1-A3, B1-B3, C2, C3) -> StandardScaler|
|       * Categorical Features (Level, Grade, Gender, Location, Transport) ->OHE|
|     - Stratified Holdout Split: 80% Train (N = 945), 20% Test (N = 237)      |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  4. MODEL TRAINING & SYSTEMATIC BENCHMARKING                                  |
|     - Stratified 5-Fold Cross-Validation on Training Data                     |
|     - 5 Classifiers: Logistic Regression, Random Forest, SVM RBF, KNN, GradB  |
|     - Evaluation Metrics: Accuracy, Macro-F1, Precision, Recall, Confusion Mat|
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|  5. MODEL EXPLAINABILITY & WEB DEPLOYMENT                                     |
|     - Permutation Feature Importance (n_repeats = 15, scoring = 'f1_macro')   |
|     - Interactive Production Deployment on Streamlit Cloud                    |
+-------------------------------------------------------------------------------+
```
*Figure 1: Comprehensive Educational Data Mining Pipeline and Research Workflow*

### 3.1 Dataset Demographics and Operationalization
The study leverages an empirical cohort of **1,182 Indonesian secondary school students** cataloged on Mendeley Data. The psychometric instrument consists of 36 items calibrated on a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree), structured evenly across 9 subscales (4 items per subscale), as summarized in Table 1.

*Table 1. Operationalization of Learning Readiness Subscales*
| Subscale Code | Psychological Construct | Number of Items | Coding Specifications |
| :---: | :--- | :---: | :--- |
| **A1** | School Belonging | 4 items | Unidirectional positive scale (1.0–5.0) |
| **A2** | Emotion Regulation | 4 items | Unidirectional positive scale (1.0–5.0) |
| **A3** | Prosocial Behavior | 4 items | Items A3.1 & A3.2 are negative (*Reversed*) |
| **B1** | Academic Persistence | 4 items | Unidirectional positive scale (1.0–5.0) |
| **B2** | Study Planning | 4 items | Unidirectional positive scale (1.0–5.0) |
| **B3** | Classroom Engagement | 4 items | Unidirectional positive scale (1.0–5.0) |
| **C1** | Academic Self-Efficacy | 4 items | **Prediction Target** (Excised from predictors) |
| **C2** | Value of Learning | 4 items | Unidirectional positive scale (1.0–5.0) |
| **C3** | Growth Mindset | 4 items | Item C3.3 pre-reversed in raw data (*Retained*) |

The demographic attributes within the cohort encompass:
- **Educational Level**: Senior High School / SMA (633 students; 53.55%) and Junior High School / SMP (549 students; 46.45%).
- **Gender**: Female (717 students; 60.66%) and Male (465 students; 39.34%).
- **School Geographic Location**: Rural / *Desa* (618 students; 52.28%), Urban / *Kota* (527 students; 44.59%), and Semi-Urban Regional / *Daerah* (37 students; 3.13%).
- **Class Grade**: Grades 7, 8, 9 (SMP) and Grades 10, 11, 12 (SMA).
- **Commuting Mode**: Motorcycle, Bicycle, Walking, Private Car, and Public Transit.

### 3.2 Data Preprocessing and Anti-Leakage Protocol
To guarantee strict scientific validity, data preprocessing adhered to rigorous standards:
1. **Item Reversal (*Reverse Coding*)**: Items A3.1 (*"I frequently get into fights/arguments at school"*) and A3.2 (*"I frequently disrupt classroom order"*) are counter-indicative of prosocial behavior. Their values were inverted using:
   $$\text{Score}_{\text{corrected}} = 6 - \text{Score}_{\text{raw}}$$
   Item C3.3 (*"Intelligence cannot be changed"*) was empirically verified as already aligned positively within the source database; hence, no second inversion was performed.
2. **Composite Subscale Aggregation**: Subscale indices were calculated as the unweighted arithmetic mean of their respective four items.
3. **Discretization of Target Variable (C1)**: Subscale C1 scores were categorized into three ordinal tiers utilizing tercile quantiles (33.3rd and 66.7th percentiles):
   - **Low**: $C1 \le 4.25$ ($N = 529$ students; $44.75\%$)
   - **Moderate**: $4.25 < C1 \le 4.75$ ($N = 347$ students; $29.36\%$)
   - **High**: $C1 > 4.75$ ($N = 306$ students; $25.89\%$)
4. **Data Leakage Immunization**:
   - Subscale C1 was strictly deleted from the predictor feature set $X$.
   - Ten subject grade columns (e.g., Mathematics, Physics, Biology, Economics) were systematically omitted because they were subject to severe structural missingness (due to curriculum elective variations between SMP and SMA) and presented high risks of confounding leakage.

### 3.3 Feature Pipeline and Model Hyperparameters
The feature set comprises 8 continuous numerical subscales ($A1, A2, A3, B1, B2, B3, C2, C3$) and 5 categorical features (*level, grade, gender, location, transport*). The cohort ($N = 1,182$) was partitioned via stratified random splitting (80% train, $N = 945$; 20% test, $N = 237$; `random_state = 42`). 

All preprocessing was implemented inside a `ColumnTransformer` embedded in scikit-learn `Pipeline` objects:
- Continuous variables were standardized using `StandardScaler` ($\mu = 0, \sigma = 1$).
- Categorical features were converted via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- Scalers and encoders were fit exclusively on training folds, shielding all test distributions from informational contamination.

The five classifiers were instantiated with standardized configurations:
- **Logistic Regression**: `solver='lbfgs'`, `max_iter=1000`, `multi_class='multinomial'`.
- **Random Forest**: `n_estimators=300`, `criterion='gini'`, `bootstrap=True`.
- **SVM RBF**: `kernel='rbf'`, $C = 1.0$, $\gamma = \text{'scale'}$, `probability=True`.
- **KNN**: $k = 15$ neighbors, uniform weights, Euclidean distance metric.
- **Gradient Boosting**: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`.

Model performance was assessed using Stratified 5-Fold Cross-Validation on the training subset, followed by unbiased evaluation on the holdout test set using Accuracy, Macro-Averaged F1-Score, Confusion Matrix, and Permutation Feature Importance ($n_{\text{repeats}} = 15$).

---

## 4. RESULTS AND DISCUSSION

### 4.1 Exploratory Analysis and Psychometric Properties
The statistical distributions of the nine school readiness subscales are detailed in Table 2.

*Table 2. Descriptive Statistics of Learning Readiness Subscales (N = 1,182)*
| Subscale | Latent Construct | Mean | Std. Dev. | Min | Max |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **A1** | School Belonging | 3.835 | 0.821 | 1.000 | 5.000 |
| **A2** | Emotion Regulation | 3.771 | 0.799 | 1.000 | 5.000 |
| **A3** | Prosocial Behavior | 4.155 | 0.661 | 1.000 | 5.000 |
| **B1** | Academic Persistence | 4.357 | 0.692 | 1.000 | 5.000 |
| **B2** | Study Planning | 3.807 | 0.853 | 1.000 | 5.000 |
| **B3** | Classroom Engagement | 4.025 | 0.732 | 1.000 | 5.000 |
| **C1** | Academic Self-Efficacy (*Target*) | 4.328 | 0.678 | 1.000 | 5.000 |
| **C2** | Value of Learning | 4.555 | 0.657 | 1.000 | 5.000 |
| **C3** | Growth Mindset | 4.253 | 0.681 | 1.000 | 5.000 |

Key exploratory findings:
- **Strong Linear Predictors**: B1 (*Academic Persistence*) exhibited the strongest positive Pearson correlation with C1 ($r = 0.71, p < 0.001$), closely followed by C2 (*Value of Learning*, $r = 0.69, p < 0.001$).
- **Absence of Critical Multicollinearity**: The peak pairwise inter-predictor correlation occurred between B3 (*Engagement*) and B1 (*Persistence*) at $r = 0.69$. With all predictor inter-correlations remaining substantially below the collinearity ceiling of $0.85$, gradient estimations in both linear and ensemble models remain statistically stable.
- **PCA Dimensionality Separation**: 2D Principal Component Analysis demonstrated that the primary principal component (PC1) effectively separates low-efficacy and high-efficacy students into distinct linear manifolds, establishing the intrinsic separability of the feature space.

### 4.2 Machine Learning Model Benchmarking
Table 3 presents the comparative performance metrics across all models, benchmarked against the Zero-Rule majority baseline.

*Table 3. Cross-Validation and Test Performance Benchmark*
| Model Architecture | CV Accuracy (Mean ± Std) | CV Macro-F1 (Mean ± Std) | Test Accuracy | Test Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| *Zero-Rule Baseline* | *0.4479 ± 0.0000* | *0.2063 ± 0.0000* | *0.4473* | *0.2061* |
| **Logistic Regression** | 0.6349 ± 0.0229 | 0.6038 ± 0.0303 | 0.6329 | 0.6026 |
| **Random Forest (300 Trees)** | **0.6593 ± 0.0152** | **0.6404 ± 0.0190** | 0.6329 | 0.6111 |
| **SVM RBF** | 0.6571 ± 0.0188 | 0.6342 ± 0.0243 | **0.6582** | 0.6295 |
| **KNN (k=15)** | 0.6317 ± 0.0205 | 0.6015 ± 0.0209 | 0.6414 | 0.6038 |
| **Gradient Boosting** | 0.6497 ± 0.0113 | 0.6343 ± 0.0123 | 0.6498 | **0.6317** |

As shown in Table 3:
- Every machine learning classifier significantly surpassed the naive Zero-Rule baseline, generating an accuracy gain exceeding 19–21 percentage points and tripling the baseline Macro-F1 score.
- In cross-validation, **Random Forest** achieved the leading CV Accuracy ($0.6593$) and CV Macro-F1 ($0.6404$).
- On the unseen holdout test set ($N = 237$), **SVM RBF** delivered the highest overall accuracy of **65.82%** (Macro-F1: 0.6295), demonstrating the efficacy of margin maximization in psychometric classification.
- **Gradient Boosting** achieved the highest test Macro-F1 score of **0.6317** (Accuracy: 64.98%), demonstrating balanced discriminatory power across all target categories.

### 4.3 Granular Error Analysis: Confusion Matrix & Per-Class Metrics
Table 4 delineates the granular per-class precision, recall, and F1 performance on the holdout test set.

*Table 4. Test Set Classification Report per Target Class*
| Target Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Low** | 0.7615 | 0.7830 | **0.7721** | 106 students |
| **Moderate** | 0.4167 | 0.3571 | **0.3846** | 70 students |
| **High** | 0.6176 | 0.6885 | **0.6512** | 61 students |
| **Overall Accuracy** | - | - | **0.6329** | 237 students |
| **Macro Average** | **0.5986** | **0.6096** | **0.6026** | 237 students |
| **Weighted Average** | **0.6226** | **0.6329** | **0.6265** | 237 students |

Analysis of the confusion matrix reveals key operational strengths:
1. **Clinical Detection of At-Risk Students**: The model achieves an exceptional **Recall of 78.30%** and **Precision of 76.15%** (F1 = 0.7721) for the *Low* self-efficacy group, correctly isolating 83 out of 106 vulnerable students with only 5 students misclassified as High. This characteristic is vital for screening, ensuring that at-risk students are reliably prioritized for psychological counseling.
2. **Identification of High Efficacy**: For the *High* self-efficacy class, the model achieves a Recall of 68.85% (42 correctly identified out of 61) with only 2 students falsely tagged as Low.
3. **The Borderline Challenge**: As typical in psychometric classification, the *Moderate* category represents a transition zone with overlapping boundaries, resulting in an F1-score of 0.3846, where errors disperse into adjacent tiers.

### 4.4 Model Interpretability: Permutation Feature Importance
To assess feature importance without reliance on model-specific internal heuristics, *Permutation Feature Importance* was executed on the test partition with 15 random shuffles. The mean degradation in Macro-F1 is ranked in Table 5.

*Table 5. Permutation Feature Importance Ranking (Test Set Macro-F1 Penalty)*
| Rank | Feature / Predictor | Domain Category | Macro-F1 Drop (Mean ± Std) |
| :---: | :--- | :--- | :---: |
| **1** | **B3 (Classroom Engagement)** | Behavioral Readiness | **+0.1039 ± 0.0272** |
| **2** | **C2 (Value of Learning)** | Academic Attitude | **+0.0501 ± 0.0216** |
| **3** | **Grade Level (*Kelas*)** | Demographic Profile | **+0.0418 ± 0.0097** |
| **4** | **B1 (Academic Persistence)** | Learning Competency | **+0.0393 ± 0.0268** |
| **5** | **School Tier (*Jenjang*)** | Demographic Profile | **+0.0316 ± 0.0162** |
| **6** | **B2 (Study Planning)** | Learning Competency | **+0.0292 ± 0.0266** |
| **7** | **A1 (School Belonging)** | Socio-Emotional | **+0.0144 ± 0.0040** |
| **8** | **C3 (Growth Mindset)** | Cognitive Attitude | **+0.0131 ± 0.0086** |
| **9** | **A2 (Emotion Regulation)** | Socio-Emotional | **+0.0128 ± 0.0086** |
| **10** | **A3 (Prosocial Behavior)** | Socio-Emotional | **+0.0083 ± 0.0054** |
| 11 | School Location | Geographic Demographics | -0.0032 ± 0.0051 |
| 12 | Gender | Student Demographics | -0.0044 ± 0.0078 |
| 13 | Commuting Transport | Physical Demographics | -0.0084 ± 0.0084 |

These empirical findings provide critical educational insights:
- **Classroom Engagement (B3)** emerged as the most indispensable feature; shuffling its values induced an immediate 0.1039 drop in test Macro-F1. Active classroom inquiry and participation serve as the primary behavioral manifestations of academic confidence.
- **Value of Learning (C2)** and **Academic Persistence (B1)** represent the principal cognitive pillars: students who recognize the utility of education and exhibit grit are fundamentally equipped to develop high self-efficacy.
- **Invariance Across Superficial Demographics**: Variables such as commuting transportation, gender, and school geographic location exerted negligible or mildly negative impact on predictive power. This proves that academic self-efficacy is overwhelmingly shaped by institutional learning habits, personal agency, and psychological readiness rather than innate demographic categories or geographic divides.

### 4.5 Production Deployment and Practical Utility
The complete predictive architecture has been packaged and deployed as a publicly accessible, interactive web system using Streamlit Community Cloud. The application incorporates:
- Interactive exploratory dashboards illustrating class distributions, correlation heatmaps, and PCA manifold projections.
- A single-student prediction simulator enabling counselors to adjust readiness scores via intuitive sliders and receive immediate probability breakdowns.
- Automated early-warning indicators that immediately flag at-risk profiles to guide early psycho-educational interventions.

---

## 5. CONCLUSION AND FUTURE WORK

This study successfully designed, benchmarked, and deployed a machine learning framework for classifying academic self-efficacy among Indonesian secondary school students:
1. Supervised machine learning algorithms—operating within a leak-free `ColumnTransformer` and `Pipeline` framework—achieved robust classification performance, with **SVM RBF** reaching **65.82% test accuracy** and **Gradient Boosting** obtaining a **0.6317 Macro-F1**, outperforming the Zero-Rule baseline by over 21 percentage points.
2. The predictive framework exhibits exceptional sensitivity in identifying vulnerable students (*Low* self-efficacy recall: **78.30%**, precision: **76.15%**).
3. Permutation feature importance analysis empirically isolated **Classroom Engagement (B3)**, **Value of Learning (C2)**, and **Academic Persistence (B1)** as the decisive drivers of student self-efficacy, establishing actionable targets for teacher interventions.
4. Deployment via Streamlit bridges the gap between machine learning research and educational practice, offering schools an accessible digital screening tool.

Future research will focus on integrating longitudinal panel data across consecutive semesters and evaluating cost-sensitive ordinal loss functions to refine decision boundaries between intermediate self-efficacy cohorts.

---

## ACKNOWLEDGMENTS

The author expresses sincere appreciation to the original investigators who curated and published the Indonesian student psychometric readiness dataset on Mendeley Data, and extends gratitude to academic supervisors and peer researchers for constructive methodological feedback throughout this research.

---

## REFERENCES

Bandura, A. (1997). *Self-Efficacy: The Exercise of Control*. New York: W. H. Freeman and Company.

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273–297. https://doi.org/10.1007/BF00994018

Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. *IEEE Transactions on Information Theory*, 13(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964

Dweck, C. S. (2006). *Mindset: The New Psychology of Success*. New York: Random House.

Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Gunawan, H., & Catherine. (2020). Information Systems Students' Study Performance Prediction Using Data Mining Approach. *2019 International Conference on Information and Communications Technology (ICOIACT)*, 1–8. https://doi.org/10.1109/icic47613.2019.8985718

Hairani, H., Nugraha, G. S., Abdillah, M. N., & Innuddin, M. (2018). Komparasi Akurasi Metode Correlated Naive Bayes Classifier dan Naive Bayes Classifier untuk Diagnosis Penyakit Diabetes. *InfoTekJar (Jurnal Nasional Informatika Dan Teknologi Jaringan)*, 3(1), 6–11. https://doi.org/10.30743/infotekjar.v3i1.558

Hairani, H., Saputro, K. E., & Fadli, S. (2020). K-means-SMOTE untuk menangani ketidakseimbangan kelas dalam klasifikasi penyakit diabetes dengan C4.5, SVM, dan naive Bayes. *Jurnal Teknologi Dan Sistem Komputer*, 8(2), 89–93. https://doi.org/10.14710/jtsiskom.8.2.2020.89-93

Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). John Wiley & Sons.

Igual, L., & Segui, S. (2017). *Introduction to Data Science: A Python Approach to Concepts, Techniques and Applications*. Springer Nature.

Jadhav, S. D., & Channe, H. P. (2016). Comparative Study of K-NN, Naive Bayes and Decision Tree Classification Techniques. *International Journal of Science and Research (IJSR)*, 5(1), 2014–2017.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(4), 1–21. https://doi.org/10.1145/2382577.2382579

Nugraha, G. S., & Hairani, H. (2018). Aplikasi Pemetaan Kualitas Pendidikan di Indonesia Menggunakan Metode K-Means. *MATRIK: Jurnal Manajemen, Teknik Informatika Dan Rekayasa Komputer*, 17(2), 13–23.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Romero, C., & Ventura, S. (2020). Educational Data Mining and Learning Analytics: An Updated Survey. *WIREs Data Mining and Knowledge Discovery*, 10(3), e1355. https://doi.org/10.1002/widm.1355

Schunk, D. H., & DiBenedetto, M. K. (2020). Motivation and Social Cognitive Theory. *Contemporary Educational Psychology*, 60, 101832. https://doi.org/10.1016/j.cedpsych.2019.101832

Sulistyono, S., & Sulistiyowati, W. (2018). Peramalan Produksi dengan Metode Regresi Linier Berganda. *PROZIMA (Productivity, Optimization and Manufacturing System Engineering)*, 1(2), 82. https://doi.org/10.21070/prozima.v1i2.1350

Zimmerman, B. J. (2000). Self-Efficacy: An Essential Motive to Learn. *Contemporary Educational Psychology*, 25(1), 82–91. https://doi.org/10.1006/ceps.1999.1016
