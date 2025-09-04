import os, json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from scripts.generate_synthetic_data import generate

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def ensure_data():
    if not os.path.exists(DATA_PATH):
        df = generate()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df

def train():
    df = ensure_data()

    target = "target"
    features = [c for c in df.columns if c != target]

    # Tipos
    categorical = ["channel"]
    numeric = [c for c in features if c not in categorical]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric)
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_split=4,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    # Métricas
    y_proba = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Persistir
    joblib.dump(pipe, os.path.join(MODEL_DIR, "model.pkl"))
    # nomes das features após transformação
    feature_names = list(pipe.named_steps["pre"].get_feature_names_out())
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Salvar pequena amostra de treino (para LIME)
    X_train.sample(1000, random_state=42).to_csv(os.path.join(MODEL_DIR, "train_sample.csv"), index=False)

    metrics = {
        "roc_auc": auc,
        "report": report
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model trained. AUC:", round(auc, 4))
    return metrics

if __name__ == "__main__":
    train()
