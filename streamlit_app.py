from pathlib import Path
from datetime import timedelta
import base64, tempfile
import joblib
import streamlit as st

st.set_page_config(page_title="SetkaPredict 2026", page_icon="🏓")

@st.cache_resource(show_spinner="Loading the calibrated 2026 model…")
def load_model():
    parts = sorted((Path(__file__).parent / "model_v2_parts").glob("model.b64.part*"))
    if not parts: raise FileNotFoundError("Model files are missing")
    encoded = "".join(p.read_text(encoding="ascii") for p in parts)
    target = Path(tempfile.gettempdir()) / "setkapredict_2026.joblib"
    target.write_bytes(base64.b64decode(encoded))
    return joblib.load(target)

try:
    model = load_model()
except Exception as exc:
    st.error("The 2026 model could not be loaded. Please retry shortly."); st.exception(exc); st.stop()

latest = max(s.last_time for s in model.states.values() if s.last_time is not None)
active_cutoff = latest - timedelta(days=30)
active_keys = {k for k,s in model.states.items() if s.last_time is not None and s.last_time >= active_cutoff}
active_names = sorted(model.names[k] for k in active_keys)
test = model.report["test"]

st.title("🏓 SetkaPredict 2026")
st.caption("Calibrated probabilities from official Setka Cup men’s results")
st.info(f"CURRENT MODEL: 92,946 completed 2026 matches through {latest:%B %d, %Y}; {len(active_names)} players active in the latest 30 days.")

with st.expander("Data audit and unseen-match test results"):
    st.write("Official feed: January 1–September 12, 2026 · 96,556 records inspected · 529 technical results and 3,081 non-men’s records excluded · all retained set scores verified.")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Overall accuracy", f"{test['accuracy']:.1%}"); m2.metric("Brier score", f"{test['brier']:.3f}")
    m3.metric("ROC-AUC", f"{test['roc_auc']:.3f}"); m4.metric("Calibration error", f"{test['ece_10']:.1%}")
    st.write(f"Untouched test period: {model.report['date_ranges']['calibration_end'][:10]} through {model.report['date_ranges']['test_end'][:10]} ({test['n']:,} matches).")
    st.write(f"At ≥70% confidence: **{test['selective']['0.7']['accuracy']:.1%} accuracy** on {test['selective']['0.7']['n']:,} matches. At ≥75%: **{test['selective']['0.75']['accuracy']:.1%}** on {test['selective']['0.75']['n']:,} matches.")

left,right = st.columns(2)
with left: player_a = st.selectbox("Player A", active_names, help="Search players active during the latest 30 days")
with right: player_b = st.selectbox("Player B", active_names, index=1, help="Search players active during the latest 30 days")

if st.button("Predict winner", type="primary", use_container_width=True):
    if player_a == player_b: st.error("Choose two different players.")
    else:
        result=model.predict(player_a,player_b,when=latest+timedelta(minutes=30))
        a_col,b_col=st.columns(2); a_col.metric(player_a,f"{result['p_a']:.1%}"); b_col.metric(player_b,f"{result['p_b']:.1%}")
        if result["status"]=="ABSTAIN": st.warning("No confident prediction: "+result["reason"])
        else: st.success(f"Predicted winner: {result['winner']} ({result['confidence']} confidence)")
        st.write(f"Empirical stability range for {player_a}: {result['interval_95'][0]:.1%}–{result['interval_95'][1]:.1%}")
        with st.expander("Explanation and model checks"):
            st.write("Most influential signals: "+", ".join(result["top_factors"])); st.json({"history":result["history"],"checks":result["checks"]})

st.divider(); st.caption("Probabilities are estimates, not guarantees. The model abstains when evidence or calibrated edge is insufficient.")
