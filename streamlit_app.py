from pathlib import Path
import tempfile
import streamlit as st
from setkapredict import SetkaPredictor
from setkapredict.synthetic import generate

st.set_page_config(page_title="SetkaPredict", page_icon="🏓")

@st.cache_resource(show_spinner="Initializing demonstration model…")
def load_model():
    demo_path = Path(tempfile.gettempdir()) / "setkapredict_demo.csv"
    history = generate(demo_path, n_matches=2500, n_players=48, seed=5212)
    return SetkaPredictor(seed=5212).fit(history)

model = load_model()
names = sorted(model.names.values())

st.title("🏓 SetkaPredict")
st.caption("Calibrated table-tennis match probabilities with an evidence-based abstention option")
st.warning("DEMONSTRATION MODE: The players and results are synthetic. This page proves the hosted application works, but it must be trained on genuine Setka history before it can predict real players.")

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
st.caption("Probabilities are estimates, not guarantees. Synthetic validation scores do not measure real Setka performance.")

