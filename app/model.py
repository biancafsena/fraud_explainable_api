import os, joblib, numpy as np, pandas as pd
from typing import List, Dict, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TRAIN_SAMPLE_PATH = os.path.join(MODEL_DIR, "train_sample.csv")

# Carrega o pipeline treinado
PIPE = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))

# Nomes das colunas transformadas diretamente do pré-processador
PRE = PIPE.named_steps["pre"]
try:
    TRANSFORMED_FEATURES = list(PRE.get_feature_names_out())
except Exception:
    # Fallback: tenta inferir qtd. de features do estimador
    n_feats = getattr(PIPE.named_steps["clf"], "n_features_in_", None)
    if n_feats is None:
        n_feats = 64
    TRANSFORMED_FEATURES = [f"feature_{i}" for i in range(int(n_feats))]

def _to_df(payload: Dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    df["channel"] = df["channel"].astype(str)
    return df

def predict(payload: Dict) -> Tuple[int, float]:
    df = _to_df(payload)
    proba = float(PIPE.predict_proba(df)[:, 1][0])
    label = int(proba >= 0.5)
    return label, proba

def shap_explain(payload: Dict, top_k: int = 5) -> List[Dict]:
    import shap
    df = _to_df(payload)

    # transforma no espaço do modelo (pré-processado)
    X_trans = PIPE.named_steps["pre"].transform(df)
    model = PIPE.named_steps["clf"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # binário: pode vir list [class0, class1] ou diretamente um array
    if isinstance(shap_values, list):
        sv = shap_values[1]  # classe positiva
    else:
        sv = shap_values

    # 1 amostra
    sv_row = np.asarray(sv[0], dtype=float).ravel()

    # garante alinhamento entre tamanhos (SHAP vs nomes)
    n = min(len(sv_row), len(TRANSFORMED_FEATURES))
    sv_row = sv_row[:n]
    feature_names = TRANSFORMED_FEATURES[:n]

    # ordena por impacto absoluto e pega top_k
    idx = np.argsort(np.abs(sv_row))[::-1][:top_k]

    out = []
    for i in idx:
        i = int(i)  # índice escalar
        feat_name = feature_names[i]
        shap_val = float(sv_row[i])
        impact = "increase" if shap_val > 0 else "decrease"

        # tenta recuperar o valor original do payload (heurística simples)
        if "channel" in feat_name:
            val = payload.get("channel")
        else:
            base_key = feat_name.split("__")[-1]
            val = payload.get(base_key, None)

        out.append({
            "feature": feat_name,
            "value": val,
            "shap_value": shap_val,
            "impact": impact
        })
    return out

def lime_explain(payload: Dict, top_k: int = 5) -> List[Dict]:
    from lime.lime_tabular import LimeTabularExplainer
    df = _to_df(payload)

    # dados de treino no espaço original (antes do preprocessor)
    if os.path.exists(TRAIN_SAMPLE_PATH):
        train_df = pd.read_csv(TRAIN_SAMPLE_PATH)
    else:
        # usa a própria instância como fallback de treino (melhor que None)
        train_df = df.copy()

    feature_names = list(df.columns)
    categorical_features = [feature_names.index("channel")]

    def predict_proba_fn(X: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X, columns=feature_names)
        X_df["channel"] = X_df["channel"].astype(str)
        return PIPE.predict_proba(X_df)

    explainer = LimeTabularExplainer(
        training_data=train_df[feature_names].values,
        feature_names=feature_names,
        class_names=["legit", "fraud"],
        categorical_features=categorical_features,
        discretize_continuous=True,
        mode="classification"
    )

    exp = explainer.explain_instance(
        data_row=df.iloc[0].values,
        predict_fn=predict_proba_fn,
        num_features=top_k
    )

    items = []
    for feat, weight in exp.as_list():
        items.append({
            "feature": feat,
            "weight": float(weight),
            "impact": "increase" if weight > 0 else "decrease"
        })
    return items
