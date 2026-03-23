"""
============================================================================
END-TO-END DATA SCIENCE PROJECT: SaaS Customer Churn Analysis
============================================================================
Dataset: Telco Customer Churn (IBM/Kaggle)
Author:  Sarat Bhattiprolu (learning project)
Goal:    Build a complete DS pipeline covering EDA, Classification, 
         Regression, and Clustering

WHAT YOU'LL LEARN:
  - How to load, clean, and explore data (pandas)
  - How to visualize patterns (matplotlib, seaborn)
  - How to build classification models (predict churn)
  - How to build regression models (predict lifetime value)
  - How to segment customers with clustering (K-Means)
  - How to translate technical results into business strategy

HOW TO RUN:
  pip install pandas numpy matplotlib seaborn scikit-learn
  python saas_churn_project.py

  Or copy sections into a Jupyter Notebook for interactive exploration.
============================================================================
"""

# ============================================================
# IMPORTS — these are the core data science libraries
# ============================================================
import pandas as pd               # Data manipulation (think Excel on steroids)
import numpy as np                 # Math operations on arrays
import matplotlib.pyplot as plt    # Plotting and charts
import seaborn as sns              # Pretty statistical plots (built on matplotlib)
import warnings
warnings.filterwarnings('ignore')

# scikit-learn — the ML library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score,
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# PHASE 1: LOAD & EXPLORE THE DATA
# ============================================================
# CONCEPT: Every project starts with understanding your data.
# You NEVER jump straight to modeling. That's a beginner mistake.
# 
# Key questions to answer:
#   - How big is the dataset?
#   - What columns do we have?
#   - Are there missing values or data quality issues?
#   - What does the target variable look like?
# ============================================================

print("=" * 70)
print("PHASE 1: LOADING AND EXPLORING THE DATA")
print("=" * 70)

# Load the CSV file into a DataFrame
# A DataFrame is like a spreadsheet — rows and columns
df = pd.read_csv('telco_churn.csv')

# .shape returns (rows, columns)
print(f"\nDataset: {df.shape[0]} customers, {df.shape[1]} features")

# .head() shows the first 5 rows — always check this first
print(f"\nFirst 5 rows:")
print(df.head().to_string())

# .dtypes shows the data type of each column
# IMPORTANT: Notice TotalCharges is 'object' (string), not a number!
# This is a common real-world data quality issue.
print(f"\nColumn data types:")
print(df.dtypes)

# .describe() gives summary statistics for numeric columns
# Check: do min/max values make sense? Any obvious outliers?
print(f"\nSummary statistics:")
print(df.describe().to_string())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum().to_string())

# Check for hidden blank strings (a sneaky data quality issue)
print(f"\nBlank TotalCharges: {(df['TotalCharges'].str.strip() == '').sum()}")


# ============================================================
# PHASE 2: DATA CLEANING
# ============================================================
# CONCEPT: Real-world data is ALWAYS messy. Cleaning it properly 
# is 80% of a data scientist's job. Skip this step and your 
# models will give you garbage results.
#
# Common issues:
#   - Wrong data types (numbers stored as text)
#   - Missing values
#   - Duplicates
#   - Inconsistent categories
# ============================================================

print(f"\n{'=' * 70}")
print("PHASE 2: DATA CLEANING")
print("=" * 70)

# Fix 1: Convert TotalCharges from string to number
# errors='coerce' converts non-numeric values to NaN instead of crashing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fix 2: Fill missing TotalCharges
# These are new customers (tenure=1) who haven't been billed yet
print(f"Customers with missing TotalCharges: {df['TotalCharges'].isna().sum()}")
print(f"Their tenure: {df[df['TotalCharges'].isna()]['tenure'].unique()}")
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])

# Fix 3: Create useful derived features
# Feature engineering = creating new columns from existing ones
# This is where domain knowledge (your SaaS experience) adds value
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)  # 0/1 version

print(f"\nCleaned: {df.shape[0]} rows, {df.shape[1]} columns, 0 missing values")


# ============================================================
# PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# CONCEPT: EDA is about finding patterns BEFORE you model.
# Good EDA often reveals more business value than complex models.
#
# Think like a product manager:
#   - Who is churning? (demographics)
#   - When do they churn? (tenure patterns)
#   - What are they paying? (revenue impact)
#   - What services do they use? (product engagement)
# ============================================================

print(f"\n{'=' * 70}")
print("PHASE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Key business metrics
churn_rate = df['Churn_Binary'].mean()
churned_revenue = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
total_revenue = df['MonthlyCharges'].sum()

print(f"\n--- Key Business Metrics ---")
print(f"Overall Churn Rate: {churn_rate:.1%}")
print(f"Monthly Revenue at Risk: ${churned_revenue:,.0f} ({churned_revenue/total_revenue:.1%})")
print(f"Avg Tenure (churned): {df[df['Churn']=='Yes']['tenure'].mean():.0f} months")
print(f"Avg Tenure (retained): {df[df['Churn']=='No']['tenure'].mean():.0f} months")

# Churn by key dimensions
print(f"\n--- Churn by Contract Type ---")
print(df.groupby('Contract')['Churn_Binary'].mean().sort_values(ascending=False).to_string())

print(f"\n--- Churn by Internet Service ---")
print(df.groupby('InternetService')['Churn_Binary'].mean().sort_values(ascending=False).to_string())

print(f"\n--- Churn by Payment Method ---")
print(df.groupby('PaymentMethod')['Churn_Binary'].mean().sort_values(ascending=False).to_string())

# Create EDA visualization dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('SaaS Customer Churn — Exploratory Data Analysis', 
             fontsize=16, fontweight='bold', y=1.02)
colors = ['#2ecc71', '#e74c3c']

# Plot 1: Churn distribution
ax = axes[0, 0]
counts = df['Churn'].value_counts()
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{val} ({val/len(df):.1%})', ha='center', fontweight='bold')
ax.set_title('Churn Distribution', fontweight='bold')
ax.set_ylabel('Customers')

# Plot 2: Churn by Contract
ax = axes[0, 1]
contract_churn = df.groupby('Contract')['Churn_Binary'].mean().sort_values(ascending=False)
bars = ax.barh(contract_churn.index, contract_churn.values, 
               color=['#e74c3c', '#f39c12', '#2ecc71'])
for bar, val in zip(bars, contract_churn.values):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.1%}', va='center', fontweight='bold')
ax.set_title('Churn by Contract Type', fontweight='bold')
ax.set_xlim(0, 0.6)

# Plot 3: Tenure distribution by churn
ax = axes[0, 2]
df[df['Churn']=='No']['tenure'].hist(ax=ax, bins=30, alpha=0.7, color=colors[0], 
                                      label='Stayed', density=True)
df[df['Churn']=='Yes']['tenure'].hist(ax=ax, bins=30, alpha=0.7, color=colors[1], 
                                       label='Churned', density=True)
ax.set_title('Tenure by Churn Status', fontweight='bold')
ax.set_xlabel('Tenure (months)')
ax.legend()

# Plot 4: Monthly charges by churn
ax = axes[1, 0]
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=ax, palette=colors)
ax.set_title('Monthly Charges by Churn', fontweight='bold')

# Plot 5: Internet service & churn
ax = axes[1, 1]
internet_churn = df.groupby('InternetService')['Churn_Binary'].mean().sort_values(ascending=False)
ax.barh(internet_churn.index, internet_churn.values, color=['#e74c3c', '#3498db', '#95a5a6'])
ax.set_title('Churn by Internet Service', fontweight='bold')

# Plot 6: Correlation heatmap (numeric features)
ax = axes[1, 2]
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn_Binary']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, center=0)
ax.set_title('Correlation Matrix', fontweight='bold')

plt.tight_layout()
plt.savefig('01_eda_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n[Saved] 01_eda_dashboard.png")


# ============================================================
# PHASE 4: CLASSIFICATION — PREDICTING CHURN
# ============================================================
# CONCEPT: Classification predicts a CATEGORY (churn: yes/no).
# This is the most valuable model for a SaaS business because
# it lets you intervene BEFORE customers leave.
#
# We'll train 3 different algorithms and compare them:
#   1. Logistic Regression — simple, interpretable, fast
#   2. Random Forest — handles complex patterns, feature importance
#   3. Gradient Boosting — often the most accurate
#
# KEY TERMS:
#   - Training set: Data the model learns from (80%)
#   - Test set: Data we evaluate on (20%) — model never sees this
#   - Accuracy: % of predictions that are correct
#   - AUC-ROC: How well the model separates churners from non-churners
#     (0.5 = random guess, 1.0 = perfect)
#   - Precision: When model says "will churn", how often is it right?
#   - Recall: Of all actual churners, how many did we catch?
# ============================================================

print(f"\n{'=' * 70}")
print("PHASE 4: CLASSIFICATION — PREDICTING CHURN")
print("=" * 70)

# Prepare features for ML
df_model = df.drop(['customerID', 'Churn'], axis=1)

# Encode categorical variables (ML needs numbers, not text)
# LabelEncoder: 'Yes'->1, 'No'->0, 'Month-to-month'->0, etc.
cat_cols = df_model.select_dtypes(include=['object']).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Split features (X) from target (y)
X = df_model.drop('Churn_Binary', axis=1)
y = df_model['Churn_Binary']

# 80/20 train-test split
# stratify=y ensures both sets have the same churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Train and evaluate 3 models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {results[name]['accuracy']:.3f}")
    print(f"  AUC-ROC:  {results[name]['auc']:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))

# Feature Importance (from Random Forest)
rf = results['Random Forest']['model']
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 Churn Drivers:")
for _, row in importance.head(10).iterrows():
    print(f"  {row['Feature']:25s} {row['Importance']:.4f}")

# Classification visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Churn Prediction — Model Performance', fontsize=14, fontweight='bold')

# ROC Curves
ax = axes[0]
for (name, res), color in zip(results.items(), ['#3498db', '#2ecc71', '#e74c3c']):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} ({res['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontweight='bold')
ax.legend(fontsize=8)

# Confusion Matrix
best = max(results, key=lambda k: results[k]['auc'])
ax = axes[1]
cm = confusion_matrix(y_test, results[best]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred: Stay', 'Pred: Churn'],
            yticklabels=['True: Stay', 'True: Churn'])
ax.set_title(f'Confusion Matrix ({best})', fontweight='bold')

# Feature Importance
ax = axes[2]
top = importance.head(10)
ax.barh(top['Feature'][::-1], top['Importance'][::-1], color='#3498db')
ax.set_title('Top 10 Churn Drivers', fontweight='bold')

plt.tight_layout()
plt.savefig('02_classification.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n[Saved] 02_classification.png")


# ============================================================
# PHASE 5: REGRESSION — PREDICTING LIFETIME VALUE
# ============================================================
# CONCEPT: Regression predicts a NUMBER (how much $ will this 
# customer generate?). This is your LTV model.
#
# KEY TERMS:
#   - R² Score: How much variance the model explains (0-1)
#     0.80 = model explains 80% of the variation in LTV
#   - RMSE: Root Mean Squared Error — avg prediction error in $
#   - MAE: Mean Absolute Error — typical dollar amount we're off by
# ============================================================

print(f"\n{'=' * 70}")
print("PHASE 5: REGRESSION — PREDICTING LIFETIME VALUE")
print("=" * 70)

X_reg = df_model.drop(['TotalCharges', 'Churn_Binary'], axis=1)
y_reg = df_model['TotalCharges']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train_r, y_train_r)
    y_pred_r = model.predict(X_test_r)
    r2 = r2_score(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    mae = mean_absolute_error(y_test_r, y_pred_r)
    reg_results[name] = {'y_pred': y_pred_r, 'r2': r2, 'rmse': rmse, 'mae': mae}
    print(f"\n{name}: R²={r2:.3f}, RMSE=${rmse:,.0f}, MAE=${mae:,.0f}")

# Regression visualization
best_reg = max(reg_results, key=lambda k: reg_results[k]['r2'])
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(y_test_r, reg_results[best_reg]['y_pred'], alpha=0.3, s=10, color='#3498db')
ax.plot([0, y_test_r.max()], [0, y_test_r.max()], 'r--', linewidth=2, label='Perfect')
ax.set_xlabel('Actual LTV ($)')
ax.set_ylabel('Predicted LTV ($)')
ax.set_title(f'LTV Prediction — {best_reg} (R²={reg_results[best_reg]["r2"]:.3f})', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('03_regression.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n[Saved] 03_regression.png")


# ============================================================
# PHASE 6: CLUSTERING — CUSTOMER SEGMENTATION
# ============================================================
# CONCEPT: Clustering finds NATURAL GROUPS in your data without
# being told what to look for. Unlike classification/regression
# (supervised learning), clustering is UNSUPERVISED — no labels.
#
# K-Means algorithm:
#   1. Pick K cluster centers randomly
#   2. Assign each customer to nearest center
#   3. Move centers to middle of their assigned customers
#   4. Repeat until stable
#
# The result: customer personas you can act on.
# ============================================================

print(f"\n{'=' * 70}")
print("PHASE 6: CLUSTERING — CUSTOMER SEGMENTATION")
print("=" * 70)

cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Binary']
X_cluster = StandardScaler().fit_transform(df[cluster_features])

# Elbow method: try different K values
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

# Fit with K=4 (a good balance for actionable segments)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(X_cluster)

# Profile each segment
print("\nSegment Profiles:")
print("-" * 70)
for seg in sorted(df['Segment'].unique()):
    subset = df[df['Segment'] == seg]
    print(f"\n  Segment {seg} ({len(subset)} customers, {len(subset)/len(df):.1%})")
    print(f"    Avg Tenure:    {subset['tenure'].mean():.0f} months")
    print(f"    Avg Monthly:   ${subset['MonthlyCharges'].mean():.0f}")
    print(f"    Avg LTV:       ${subset['TotalCharges'].mean():,.0f}")
    print(f"    Churn Rate:    {subset['Churn_Binary'].mean():.1%}")
    print(f"    Top Contract:  {subset['Contract'].mode().values[0]}")

# Clustering visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Customer Segmentation (K-Means, K=4)', fontsize=14, fontweight='bold')
seg_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

ax = axes[0]
for seg in sorted(df['Segment'].unique()):
    mask = df['Segment'] == seg
    ax.scatter(df[mask]['tenure'], df[mask]['MonthlyCharges'],
              alpha=0.4, s=15, color=seg_colors[seg], label=f'Segment {seg}')
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Monthly Charges ($)')
ax.set_title('Segments: Tenure vs Monthly Charges', fontweight='bold')
ax.legend()

ax = axes[1]
ax.plot(range(2, 9), inertias, 'bo-', linewidth=2, markersize=8)
ax.axvline(4, color='red', linestyle='--', alpha=0.7, label='K=4')
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('04_clustering.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n[Saved] 04_clustering.png")


# ============================================================
# PHASE 7: BUSINESS RECOMMENDATIONS
# ============================================================
print(f"\n{'=' * 70}")
print("PHASE 7: STRATEGIC RECOMMENDATIONS")
print("=" * 70)

best_auc = results[best]['auc']
best_acc = results[best]['accuracy']

print(f"""
PROJECT SUMMARY
---------------
We analyzed {len(df):,} SaaS customers and built 3 types of ML models.

CLASSIFICATION (Churn Prediction):
  Best model: {best} (AUC: {best_auc:.3f}, Accuracy: {best_acc:.1%})
  Key drivers: TotalCharges, MonthlyCharges, tenure, Contract type

REGRESSION (LTV Prediction):
  Best model: {best_reg} (R²: {reg_results[best_reg]['r2']:.3f})
  Can predict customer lifetime value within ${reg_results[best_reg]['mae']:,.0f}

CLUSTERING (Segmentation):
  Identified 4 distinct customer segments with different churn profiles.

RECOMMENDED ACTIONS:
  1. Deploy churn scoring model to flag at-risk accounts weekly
  2. Build segment-specific retention playbooks
  3. Incentivize month-to-month customers to switch to annual plans
  4. Invest in first-year onboarding (highest churn risk window)
  5. Investigate fiber optic service quality (highest churn segment)
""")

print("Project complete! Check the saved PNG files for all visualizations.")
