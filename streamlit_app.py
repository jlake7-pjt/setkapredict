import streamlit as st
from setkapredict import SetkaPredictor
from setkapredict.real_data import load_public_archive

st.set_page_config(page_title="SetkaPredict", page_icon="🏓")

@st.cache_resource(show_spinner="Downloading, validating, training, and calibrating real Setka history…")
def load_model():
    history, audit = load_public_archive()
    model = SetkaPredictor(seed=5212).fit(history)
    return model, audit

try:
    model, audit = load_model()
except Exception as exc:
    st.error("The historical-data source could not be loaded. Please retry shortly.")
    st.exception(exc)
    st.stop()
names = sorted(model.names.values())

st.title("🏓 SetkaPredict")
st.caption("Calibrated table-tennis match probabilities with an evidence-based abstention option")
st.warning("HISTORICAL PROTOTYPE: This model uses 7,846 genuine Setka matches from June 10–July 8, 2022. The names are abbreviated and the archive has no official player IDs. Do not treat it as a current 2026 forecast until recent history is added.")

with st.expander("Verified historical-data and holdout results"):
    test = model.report["test"]
    st.write(f"Clean matches: **{audit['clean_rows']:,}** · Players: **{audit['players']:,}** · Untouched test matches: **{test['n']:,}**")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Overall accuracy", f"{test['accuracy']:.1%}")
    m2.metric("Brier score", f"{test['brier']:.3f}")
    m3.metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    m4.metric("Calibration error", f"{test['ece_10']:.1%}")
    st.write("At ≥70% model confidence: "
             f"**{test['selective']['0.7']['accuracy']:.1%} accuracy** on "
             f"{test['selective']['0.7']['n']} test matches "
             f"({test['selective']['0.7']['coverage']:.1%} coverage).")
    st.caption("Source audit: 4 duplicates and 1 invalid self-match removed; all retained point-level scores agree with set totals.")

left, right = st.columns(2)
with left:
    player_a = st.selectbox("Player A", names, help="Click and begin typing to search")
with right:
    player_b = st.selectbox("Player B", names, index=1, help="Click and begin typing to search")

if st.button("Predict winner", type="primary", use_container_width=True):
    if player_a == player_b:
        st.error("Choose two different players.")
    else:
        result = model.predict(player_a, player_b)
        a_col, b_col = st.columns(2)
        a_col.metric(player_a, f"{result['p_a']:.1%}")
        b_col.metric(player_b, f"{result['p_b']:.1%}")
        if result["status"] == "ABSTAIN":
            st.warning("No confident prediction: " + result["reason"])
        else:
            st.success(f"Predicted winner: {result['winner']} ({result['confidence']} confidence)")
        st.write(f"Empirical stability range for {player_a}: {result['interval_95'][0]:.1%}–{result['interval_95'][1]:.1%}")
        with st.expander("Explanation and model checks"):
            st.write("Most influential signals: " + ", ".join(result["top_factors"]))
            st.json({"history": result["history"], "checks": result["checks"]})

st.divider()
st.caption("Probabilities are estimates, not guarantees. Historical holdout performance does not guarantee current or future results.")
