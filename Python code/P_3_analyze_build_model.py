import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,StratifiedGroupKFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics    import(
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score   
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.makedirs("plots",exist_ok=True)

# UPI Fraud Analyzer - Phase 3 model bultding and Evaluation
# Input: transaction_features.csv Output: fraud_predictions.csv,model_xgb.pkl(saved model),plots(all ecaluation charts)
# Install: pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn joblib 
"""
color pallete
"""
C_FRAUD   = "#D85A30"   # coral  – fraud / high risk
C_LEGIT   = "#1D9E75"   # teal   – legitimate / low risk
C_MODEL   = ["#534AB7", "#D85A30", "#1D9E75"]   # LR, XGB, RF
 
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False,
                      "axes.spines.right": False})

"""Step 1 Load data
"""

print("\n Loading the transaction_features.csv file....")
df=pd.read_csv("transaction_features.csv")
print(f"  Rows    : {len(df):,}")
print(f"  Columns : {df.shape[1]}")  # in two dimension to find number of  column
print(f"  Fraud   : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.2f}%)")

"""
Select features
"""
print("\n[2/8] Selecting feature columns ...")
 
FEATURE_COLS = [
    # Velocity
    "txn_count_last_15min", "txn_count_last_1hr",
    "amount_sum_last_1hr",  "is_velocity_burst",
    # Time
    "hour_of_day", "day_of_week", "is_weekend",
    "is_night_txn", "time_risk_score",
    # Recipient
    "is_new_recipient", "recipient_unique_senders",
    "recipient_txn_frequency", "days_since_first_txn_to_vpa",
    "new_recipient_large_amount",
    # Amount
    "amount_zscore", "amount_vs_avg_ratio",
    "is_round_amount", "is_high_amount", "zscore_new_recip_combo",
    # Balance
    "balance_discrepancy", "has_balance_error",
    "balance_drop_ratio", "near_zero_balance_flag",
    "recipient_balance_unchanged",
    # Device & Identity
    "is_new_device", "txn_type_shift_flag",
    "txn_type_encoded", "device_type_encoded",
    "new_device_night_combo",
    # Raw amount (useful signal)
    "amount",
]
 
TARGET = "is_fraud"

#  keep only columns that exist in the file
FEATURE_COLS=[c for c in FEATURE_COLS if c in df.columns]
print(f"  Features used : {len(FEATURE_COLS)}")
 
X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()
 
# Fill any remaining NaNs with column median
X = X.fillna(X.median(numeric_only=True))
# test split + SMOTE
# Synthetic Minority Oversampling Technique (SMOTE) is a statistical 
# technique for increasing the number of cases in your dataset in a balanced way

print("\n Splitting the data applyinng SMOTE")

X_train,X_test,y_train,y_test= train_test_split(
    X,y,test_size=0.20,random_state=42,stratify=y)
print(f"Train size : {len(X_train):,} | Test size : {len(X_test):,}")


# scale features (required for logistics regression : harmless for tree models)
scaler=StandardScaler()
X_train_sc=scaler.fit_transform(X_train)
X_test_sc=scaler.transform(X_test)

# SMOTE: oversample minority class in training get only
smote=SMOTE(random_state=42, k_neighbors=5) # using as seed(42),"5" - selects minority lcass member and its 5 neighbouring
X_train_sm, y_train_sm   = smote.fit_resample(X_train_sc,y_train)
print(f"  After SMOTE — Train size: {len(X_train_sm):,}  "
      f"Fraud: {y_train_sm.sum():,} ({y_train_sm.mean()*100:.1f}%)")

# Step 4 train three model
# 1. Training Logistic Regression 2. Random Forest 3. XGBoost — primary model
print("\n Training models.....")
# 4a. Logistic regression baseline
print(f"Training logistic regression......")
lr=LogisticRegression(max_iter=1000,C=0.5,class_weight="balanced",
                      random_state=42)
lr.fit(X_train_sm,y_train_sm)
# 4b. Random forest
print("training random forest")
rf=RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=5,
    class_weight="balanced",n_jobs=-1, random_state=42
)
rf.fit(X_train_sm,y_train_sm)

# 4c. XGBoost
# scale _pos_weight balances classes natiely
fraud_ratio=(y_train_sm==0).sum()/(y_train_sm==1).sum()
print(f"  Training XGBoost  (scale_pos_weight={fraud_ratio:.1f}) ...")
xgb=XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=fraud_ratio,
    eval_metric="aucpr",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)

xgb.fit(
    X_train_sc, y_train,           # XGBoost uses unsmoted but scaled data
    eval_set=[(X_test_sc, y_test)],
    verbose=False,
)

print("all three models are trained.\n")
# evaluate the models
print("\n Evaluating models....")
models={
    "Logistic Regression":(lr,X_test_sc),
    "Random Forest":(rf,X_test_sc),
    "XGBoost":(xgb,X_test_sc),
}
results={}
for name, (model,X_eval) in models.items():
    y_pred=model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:,1]
    auc_roc=roc_auc_score(y_test,y_prob)
    avg_prec=average_precision_score(y_test,y_prob)
    f1=f1_score(y_test,y_pred)
    report = classification_report(y_test,y_pred,output_dict=True)
    precision_f=report["1"]["precision"]
    recall_f=report["1"]["recall"]
    results[name]={
        "model":model,"y_pred":y_pred,"y_prob":y_prob,
        "auc_roc":auc_roc,"avg_prec":avg_prec,"f1":f1,
        "precision":precision_f,"recall":recall_f,
    }

    print(f"\n  ── {name} ──")
    print(f"     AUC-ROC   : {auc_roc:.4f}")
    print(f"     Avg Prec  : {avg_prec:.4f}")
    print(f"     F1 (fraud): {f1:.4f}")
    print(f"     Precision : {precision_f:.4f}")
    print(f"     Recall    : {recall_f:.4f}")

# plots
print("\n Generating evaluation plots")

# plot A : ROC curves (all 3 models)
fig,ax=plt.subplots(figsize=(7,5))
for ( name, res), color in zip(results.items(),C_MODEL):
    fpr,tpr,_=roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr,color=color,lw=2,
            label=f"{name}  (AUC={res['auc_roc']:.3f})")

ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve — UPI fraud detection")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plots/roc_curves.png")
plt.close()
print("  Saved plots/roc_curves.png")

# plot b precision recall curves
fig, ax = plt.subplots(figsize=(7, 5))
for (name,res),color in zip(results.items(),C_MODEL):
    prec,rec,_=  precision_recall_curve(y_test,res["y_prob"])
    ap=res["avg_prec"]
    ax.plot(rec, prec, color=color, lw=2, label=f"{name}    (AP={ap:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curve — UPI fraud detection")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("plots/precision_recall_curves.png")
plt.close()
print("  Saved plots/precision_recall_curves.png")


# ── Plot C: Confusion matrix — XGBoost ────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, results["XGBoost"]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion matrix — XGBoost")
plt.tight_layout()
plt.savefig("plots/confusion_matrix_xgb.png")
plt.close()
print("  Saved plots/confusion_matrix_xgb.png")


# plot D: model comparison bar chart
metrics_df=pd.DataFrame({
    name:{
        "AUC-ROC"   :   res["auc_roc"],
        "F1-Score"  :   res["f1"],
        "Precision" :   res["precision"] ,
        "Recall"    :   res["recall"],
    }
    for name, res in results.items()
}).T
fig,ax=plt.subplots(figsize=(8,4))
metrics_df.plot(kind="bar",ax=ax,color=["#534AB7","#D85A30","#1D9E75","#FAC775"],
                edgecolor="white", width=0.7  )
ax.set_ylim(0,1.1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.set_title("Model comparison — key metrics")
ax.legend(loc="lower right", fontsize=9)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
                ha="center", va="bottom", fontsize=8)
    
plt.tight_layout()
plt.savefig("plots/model_comparison.png")
plt.close()
print("  Saved plots/model_comparison.png")

# plot E feature importance -XGboost
feat_imp=pd.Series(xgb.feature_importances_, index=FEATURE_COLS)
feat_imp = feat_imp.sort_values(ascending=True).tail(20)
 
fig, ax = plt.subplots(figsize=(7, 7))
bars = ax.barh(feat_imp.index, feat_imp.values, color=C_FRAUD, edgecolor="white")
ax.set_xlabel("Feature importance (XGBoost)")
ax.set_title("Top 20 features — XGBoost")
plt.tight_layout()
plt.savefig("plots/feature_importance_xgb.png")
plt.close()
print("  Saved plots/feature_importance_xgb.png")
 
# ── Plot F: Fraud rate by hour ─────────────────────────────
fraud_by_hour = df.groupby("hour_of_day")["is_fraud"].mean() * 100
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(fraud_by_hour.index, fraud_by_hour.values,
              color=[C_FRAUD if h in range(1, 5) else C_LEGIT
                     for h in fraud_by_hour.index],
              edgecolor="white")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Fraud rate (%)")
ax.set_title("Fraud rate by hour of day  (red = highest-risk window 1am–4am)")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig("plots/fraud_rate_by_hour.png")
plt.close()
print("  Saved plots/fraud_rate_by_hour.png")
 
 
# ════════════════════════════════════════════════════════════
# STEP 7 — SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════
print("\n[7/8] Generating SHAP explainability plots ...")
 
# Use a sample of 2000 rows for SHAP (faster; representative)
sample_idx  = np.random.choice(len(X_test_sc), size=min(2000, len(X_test_sc)),
                                replace=False)
X_shap      = pd.DataFrame(X_test_sc, columns=FEATURE_COLS)
 
explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_shap)
 
# ── SHAP A: Summary bar plot ───────────────────────────────
plt.figure(figsize=(8, 7))
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False,
                  max_display=15)
plt.title("SHAP feature importance — XGBoost")
plt.tight_layout()
plt.savefig("plots/shap_summary_bar.png", bbox_inches="tight")
plt.close()
print("  Saved plots/shap_summary_bar.png")
 
# ── SHAP B: Beeswarm plot (impact direction) ───────────────
plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
plt.title("SHAP beeswarm — feature impact direction")
plt.tight_layout()
plt.savefig("plots/shap_beeswarm.png", bbox_inches="tight")
plt.close()
print("  Saved plots/shap_beeswarm.png")
 
# ── SHAP C: Single transaction explanation ─────────────────
# Pick the highest-scoring fraud transaction as the example
fraud_probs = results["XGBoost"]["y_prob"]
top_fraud_idx = np.argsort(fraud_probs)[-1]   # most confident fraud in test set
 
shap_exp = shap.Explanation(
    values    = shap_values[top_fraud_idx],
    base_values = explainer.expected_value,
    data      = X_shap.iloc[0].values,         # representative row
    feature_names = FEATURE_COLS,
)
plt.figure(figsize=(9, 5))
shap.waterfall_plot(shap_exp, show=False, max_display=12)
plt.title("SHAP waterfall — why this transaction was flagged as HIGH RISK")
plt.tight_layout()
plt.savefig("plots/shap_waterfall_example.png", bbox_inches="tight")
plt.close()
print("  Saved plots/shap_waterfall_example.png")
 
 
# ════════════════════════════════════════════════════════════
# STEP 8 — RISK SCORE & SAVE PREDICTIONS
# ════════════════════════════════════════════════════════════
print("\n[8/8] Assigning risk scores and saving predictions ...")
 
# Score on the FULL dataset (not just test set) for the dashboard
X_full_sc = scaler.transform(X.fillna(X.median(numeric_only=True)))
fraud_prob = xgb.predict_proba(X_full_sc)[:, 1]
 
# Convert probability → 0–100 risk score
df["fraud_probability"] = np.round(fraud_prob * 100, 2)
df["risk_score"]        = df["fraud_probability"].astype(int).clip(0, 100)
 
# Risk tier labels
def assign_tier(score):
    if score >= 61:  return "High"
    if score >= 31:  return "Medium"
    return "Low"
 
df["risk_tier"] = df["risk_score"].apply(assign_tier)
 
# Readable flag
df["model_flagged"] = (df["risk_score"] >= 61).astype(int)
 
# Risk tier distribution
print(f"\n  Risk tier breakdown:")
tier_counts = df["risk_tier"].value_counts()
for tier, cnt in tier_counts.items():
    pct = cnt / len(df) * 100
    print(f"    {tier:<8} : {cnt:>7,}  ({pct:.1f}%)")
 
# Fraud capture rate at High tier
high_tier    = df[df["risk_tier"] == "High"]
capture_rate = high_tier["is_fraud"].sum() / df["is_fraud"].sum() * 100
print(f"\n  Fraud capture rate at 'High' tier : {capture_rate:.1f}%")
print(f"  (Model flags {len(high_tier):,} txns as High; captures "
      f"{high_tier['is_fraud'].sum():,} of {df['is_fraud'].sum():,} actual frauds)")
 
# ── Save predictions CSV ────────────────────────────────────
KEEP_COLS = [
    "transaction_id", "created_at", "transaction_type",
    "sender_vpa", "receiver_vpa", "amount",
    "is_fraud", "fraud_probability", "risk_score", "risk_tier", "model_flagged",
    # Key features for Power BI drill-through
    "is_new_recipient", "is_night_txn", "is_velocity_burst",
    "has_balance_error", "amount_zscore", "balance_drop_ratio",
    "is_new_device", "txn_type_shift_flag", "composite_risk_score",
    "hour_of_day", "day_of_week",
]
KEEP_COLS = [c for c in KEEP_COLS if c in df.columns]
df[KEEP_COLS].to_csv("fraud_predictions.csv", index=False)
print(f"\n  Saved fraud_predictions.csv  ({len(df):,} rows, {len(KEEP_COLS)} columns)")
 
# ── Save model & scaler ─────────────────────────────────────
joblib.dump(xgb,    "model_xgb.pkl")
joblib.dump(scaler, "scaler.pkl")
print("  Saved model_xgb.pkl  and  scaler.pkl")
 
# ── Final summary ────────────────────────────────────────────
print("\n" + "=" * 62)
print("Phase 3 complete!  Files generated:")
print("  fraud_predictions.csv  →  use in Phase 5 Power BI dashboard")
print("  model_xgb.pkl          →  saved XGBoost model")
print("  scaler.pkl             →  saved StandardScaler")
print("  plots/                 →  8 evaluation & explainability charts")
print()
print("  Best model (XGBoost) results:")
xgb_res = results["XGBoost"]
print(f"    AUC-ROC   : {xgb_res['auc_roc']:.4f}")
print(f"    F1-Score  : {xgb_res['f1']:.4f}")
print(f"    Precision : {xgb_res['precision']:.4f}")
print(f"    Recall    : {xgb_res['recall']:.4f}")

 










