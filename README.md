# SetkaPredict

Hosted, leakage-safe probability-model demonstration for table-tennis match prediction.

> **Demo limitation:** The included deployment initializes from synthetic players and synthetic match history. It verifies the application and modeling workflow; it does not claim real Setka predictive performance. Genuine historical Setka data and walk-forward evaluation are required before real-player use.

## Streamlit deployment

- Entrypoint: `streamlit_app.py`
- Python: 3.12
- Dependencies: `requirements.txt`

The model combines online Elo and set Elo, Bayesian-smoothed rolling form, head-to-head, rest/fatigue, trend and volatility features, logistic regression, gradient boosting, validation-selected blending, Platt probability calibration, and an abstention policy.
