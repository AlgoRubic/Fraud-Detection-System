import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/fraudTrain.csv")

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "trans_num"], errors="ignore")

# =========================
# PREPROCESSING
# =========================

# Encode categorical columns
encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Split features & target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# HANDLE IMBALANCE
# =========================
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# =========================
# TRAIN MODELS
# =========================

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)

#LightGBM
lgb_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    class_weight="balanced"
)

lgb_model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
lgb_pred = lgb_model.predict_proba(X_val)[:, 1]

xgb_auc = roc_auc_score(y_val, xgb_pred)
lgb_auc = roc_auc_score(y_val, lgb_pred)

print(f"XGBoost ROC-AUC: {xgb_auc}")
print(f"LightGBM ROC-AUC: {lgb_auc}")

# Choose best model
best_model = xgb_model if xgb_auc > lgb_auc else lgb_model

print("\nClassification Report:")
y_pred = best_model.predict(X_val)
print(classification_report(y_val, y_pred))

# =========================
# SAVE MODEL + ENCODERS
# =========================

joblib.dump(best_model, "models/model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("\nTraining complete. Best model saved.")