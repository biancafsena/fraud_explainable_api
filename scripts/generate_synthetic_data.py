import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

FEATURES = [
    "amount",
    "channel",  # 'PIX' ou 'CARD'
    "hour",
    "is_new_device",
    "device_trust_score",
    "days_since_last_tx",
    "tx_velocity_1h",
    "merchant_risk_score",
    "customer_age",
    "has_chargeback_history",
    "country_risk_score",
]

def generate(n_rows: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = np.exp(rng.normal(4.8, 0.9, n_rows))  # ~ 121 USD mediana
    channel = rng.choice(["PIX", "CARD"], size=n_rows, p=[0.55, 0.45])
    hour = rng.integers(0, 24, n_rows)
    is_new_device = rng.integers(0, 2, n_rows)
    device_trust_score = np.clip(rng.beta(2, 3, n_rows), 0, 1)
    days_since_last_tx = np.clip(rng.exponential(3.0, n_rows), 0, None)
    tx_velocity_1h = rng.poisson(1.5, n_rows)
    merchant_risk_score = np.clip(rng.beta(2, 2, n_rows), 0, 1)
    customer_age = np.clip(rng.normal(37, 10, n_rows), 18, 85).astype(int)
    has_chargeback_history = rng.integers(0, 2, n_rows)
    country_risk_score = np.clip(rng.beta(1.6, 2.4, n_rows), 0, 1)

    # Score de risco com interações (regra latente)
    base = (
        0.002 * amount
        + 0.7 * (channel == "PIX").astype(float)
        + 0.25 * (hour <= 5)  # madrugada
        + 0.8 * is_new_device
        - 1.2 * device_trust_score
        - 0.02 * days_since_last_tx
        + 0.35 * tx_velocity_1h
        + 1.1 * merchant_risk_score
        - 0.01 * (customer_age - 35)
        + 0.9 * has_chargeback_history
        + 0.9 * country_risk_score
    )

    # Interações non-linear
    base += 0.6 * ((amount > 1500) & (hour <= 5))
    base += 0.5 * ((tx_velocity_1h >= 5) & (is_new_device == 1))
    base += 0.4 * ((merchant_risk_score > 0.8) & (country_risk_score > 0.7))

    # Ruído
    noise = rng.normal(0, 0.8, n_rows)
    risk = base + noise

    # Probabilidade via sigmoide
    prob = 1 / (1 + np.exp(-risk + 5.0))  # shift para ~classes desbalanceadas
    y = (prob > rng.uniform(0.0, 1.0, n_rows)).astype(int)

    df = pd.DataFrame({
        "amount": amount.round(2),
        "channel": channel,
        "hour": hour,
        "is_new_device": is_new_device,
        "device_trust_score": device_trust_score.round(3),
        "days_since_last_tx": days_since_last_tx.round(2),
        "tx_velocity_1h": tx_velocity_1h,
        "merchant_risk_score": merchant_risk_score.round(3),
        "customer_age": customer_age,
        "has_chargeback_history": has_chargeback_history,
        "country_risk_score": country_risk_score.round(3),
        "target": y,
    })

    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv("data/transactions.csv", index=False)
    print("Saved data/transactions.csv", df.shape)
