# SaaS Customer Churn: From Data to Retention Strategy

**An end-to-end data science project that analyzes 7,043 SaaS customer records to predict churn, quantify revenue impact, segment customers into actionable personas, and deliver a retention strategy with projected $1.85M annual impact.**

This project demonstrates how data science translates into business decisions — not just model accuracy scores, but dollar-denominated recommendations a leadership team can act on.

---

## The business problem

A subscription-based SaaS company is losing customers at an alarming rate. Leadership knows churn is high but lacks visibility into who is leaving, why they are leaving, and what it would cost to retain them versus letting them go.

**Key question:** Can we predict which customers will churn, understand the drivers, and build a retention strategy with clear ROI?

## Key findings

| Metric | Value | Business implication |
|--------|-------|---------------------|
| Overall churn rate | 31.7% | ~1 in 3 customers leaves — well above the 5-7% SaaS benchmark |
| Monthly revenue at risk | $143,866 | $1.73M annually walking out the door |
| Month-to-month churn | 43.1% | 3.3x higher than two-year contracts — biggest single lever |
| First-year churn | Highest concentration | Onboarding is failing to deliver value fast enough |
| Fiber optic churn | 37.6% | Price-to-value mismatch — paying premium, not getting premium experience |

## Project deliverables

### 1. Business intelligence dashboard
Improved EDA dashboard with revenue impact quantification, not just pattern visualization. Includes KPI summary, churn drivers by contract/internet/payment type, tenure survival analysis, and monthly revenue waterfall.

![EDA Dashboard](images/01_eda_dashboard.png)

### 2. Predictive model evaluation with business context
Three classification models (Logistic Regression, Random Forest, Gradient Boosting) evaluated not just on statistical metrics but on dollar impact. Confusion matrix shows the cost of each error type: a missed churner costs $1,500 in lost LTV versus $50 for a false alarm.

![Model Performance](images/02_model_performance.png)

**Best model:** Gradient Boosting (AUC: 0.722)
**Recommendation:** Lower threshold from 0.50 to 0.35 to prioritize recall — the asymmetric cost structure makes catching more churners worth the additional false positives.

### 3. Customer segmentation personas
K-means clustering identified 4 distinct customer segments, each with a named persona, behavioral profile, and tailored retention strategy with budget estimates.

![Customer Segments](images/03_customer_segments.png)

### 4. Executive memo
A 2-page decision document structured for C-suite consumption: problem quantification, key findings, model results, and 5 ranked initiatives with investment requirements, projected impact, and implementation timeline.

[Download Executive Memo (DOCX)](04_executive_memo.docx)

### 5. Complete analysis code
Fully documented Python script covering the entire pipeline from data loading through strategic recommendations.

---

## Recommended strategy (ranked by ROI)

| # | Initiative | Investment | Annual impact | Timeline |
|---|-----------|-----------|---------------|----------|
| 1 | Deploy ML churn early-warning system | $120K | $850K saved | Q2 2026 |
| 2 | Annual plan migration campaign | $60K | $400K saved | Q2 2026 |
| 3 | Redesign first-90-day onboarding | $80K | $300K saved | Q3 2026 |
| 4 | Fiber optic service quality audit | $150K | $200K saved | Q3-Q4 2026 |
| 5 | Auto-pay migration campaign | $25K | $100K saved | Q2 2026 |

**Total investment: $435K | Projected annual impact: $1.85M | ROI: 4.3x**

---

## Technical approach

### Data pipeline
- Loaded and profiled 7,043 customer records with 21 features
- Identified and resolved data quality issues (TotalCharges stored as text, 11 blank values for new customers)
- Engineered derived features and encoded categorical variables

### Exploratory data analysis
- Univariate and bivariate analysis across all features
- Correlation analysis identifying top churn drivers
- Revenue impact quantification (not just statistical patterns)

### Classification models
- Logistic Regression, Random Forest (200 trees), Gradient Boosting (200 estimators)
- 5-fold cross-validation for robust performance estimation
- Precision-recall threshold optimization based on business cost asymmetry
- Feature importance analysis translating model internals into product priorities

### Customer segmentation
- K-means clustering (K=4) on behavioral features
- Elbow method for optimal cluster selection
- Persona naming and strategy mapping based on segment profiles

---

## Tools and libraries

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

---

## About the author

Sarat Bhattiprolu — Product and strategy leader with 8+ years across consulting, SaaS, and healthcare. Georgetown MBA. Combining domain expertise with data science to drive business decisions.

[LinkedIn](https://linkedin.com/in/saratbhattiprolu) | [Email](mailto:sarat.bhattiprolu1@gmail.com)
