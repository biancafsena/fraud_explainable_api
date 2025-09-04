# 🚀 Fraud Scoring API — Explainable AI (SHAP + LIME)

API em **FastAPI** para detecção de fraude em tempo real com foco em **explicabilidade**.  
O projeto simula um fluxo usado por bancos e fintechs: treina um modelo em dados **sintéticos** de transações e expõe endpoints REST para:

- **Prever** probabilidade de fraude.  
- **Explicar** cada decisão usando **SHAP** e **LIME** (explicabilidade exigida por times de **Risco, Compliance e Bacen**).

---

## 🧱 Tech Stack
- Python 3.11+
- FastAPI + Uvicorn
- scikit-learn (RandomForest + Pipeline)
- SHAP (TreeExplainer)
- LIME (LimeTabularExplainer)
- Docker (deploy pronto)

---

## 📦 Como rodar localmente
```bash
# 1) Clone e crie ambiente virtual
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) (Opcional) Regerar dados e re-treinar o modelo
python -m model.train

# 3) Subir a API
uvicorn app.main:app --reload
# Swagger: http://127.0.0.1:8000/docs


#Docker

docker build -t fraud-explainable-api .
docker run -p 8000:8000 fraud-explainable-api


#Endpoints

POST /predict → Probabilidade de fraude + rótulo (0/1).
POST /explain/shap → Top features que mais contribuíram (SHAP).
POST /explain/lime → Explicação local por LIME.

#Exemplo de payload

{
  "amount": 1299.90,
  "channel": "PIX",
  "hour": 1,
  "is_new_device": 1,
  "device_trust_score": 0.23,
  "days_since_last_tx": 0.5,
  "tx_velocity_1h": 7,
  "merchant_risk_score": 0.88,
  "customer_age": 28,
  "has_chargeback_history": 1,
  "country_risk_score": 0.67
}


#Curl rápido

curl -X POST http://127.0.0.1:8000/predict \
 -H "Content-Type: application/json" \
 -d '{"amount":1299.9,"channel":"PIX","hour":1,"is_new_device":1,"device_trust_score":0.23,"days_since_last_tx":0.5,"tx_velocity_1h":7,"merchant_risk_score":0.88,"customer_age":28,"has_chargeback_history":1,"country_risk_score":0.67}'


curl -X POST http://127.0.0.1:8000/predict \
 -H "Content-Type: application/json" \
 -d '{"amount":1299.9,"channel":"PIX","hour":1,"is_new_device":1,"device_trust_score":0.23,"days_since_last_tx":0.5,"tx_velocity_1h":7,"merchant_risk_score":0.88,"customer_age":28,"has_chargeback_history":1,"country_risk_score":0.67}'


#Dados sintéticos

Os dados são gerados artificialmente (scripts/generate_synthetic_data.py) e reproduzem padrões comuns em cenários de fraude:
transações de alto valor
horários de madrugada
dispositivos novos
estabelecimentos de alto risco
histórico de chargeback
alta velocidade de transações

#Explicabilidade
SHAP → mostra a contribuição individual de cada feature para a predição.
LIME → gera explicações locais aproximando o modelo ao redor de cada instância.
⚠️ Este projeto é educacional. Explicabilidade não substitui governança de modelos, fairness ou monitoramento em produção (drift, estabilidade, etc.).


✅ #Testes automatizados

pytest -q
# saída esperada: 2 passed in Xs


#Por que importa para o setor financeiro?
Transparência: decisões de fraude auditáveis para Bacen e auditorias.
Suporte a analistas de fraude: entender por que uma transação foi marcada como suspeita.
Compliance & Riscos: comunicação clara com áreas regulatórias.

✨ Projeto criado para demonstrar boas práticas de ML + MLOps + Explainable AI.
Made with ❤️ por Bianca Sena

---

👉 Esse README já equilibra: parte técnica (pra devs) + impacto de negócio

Quer que eu prepare também uma **seção visual com badges (shields.io)** no topo? Assim você teria ícones tipo “Python”, “FastAPI”, “Docker”, “SHAP” logo abaixo do título. Isso chama muito a atenção quando alguém abre o GitHub.
