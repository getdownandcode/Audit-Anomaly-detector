"""Streamlit interface for transaction anomaly review."""

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from detector import benford_test, prepare, risk_score, evaluate
from load_data import load_data


@st.cache_data(show_spinner=False)
def cached_kaggle_data() -> pd.DataFrame:
    """Load and cache the public Kaggle dataset."""
    return load_data()


@st.cache_data(show_spinner="Scoring transactions...")
def score_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare and score a transaction table."""
    prepared = prepare(data)
    benford_test(prepared)
    return risk_score(prepared)


st.set_page_config(page_title="Audit Anomaly Detector", layout="wide")
st.title("Audit Anomaly Detector")
st.write("Rank transactions for review using Benford's Law, anomaly detection, and simple audit rules.")

with st.sidebar:
    upload = st.file_uploader("Upload transactions CSV", type="csv")
    load_button = st.button("Load Kaggle dataset", type="primary")
    top_n = st.slider("Top transactions to show", min_value=10, max_value=1000, value=100, step=10)
    sample_size = st.number_input("Sample rows (0 = full dataset)", min_value=0, value=0, step=10000)

if upload is not None:
    source = pd.read_csv(upload)
elif load_button or "transactions" not in st.session_state:
    try:
        source = cached_kaggle_data()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()
else:
    source = st.session_state["transactions"]

st.session_state["transactions"] = source
if sample_size:
    source = source.sample(min(int(sample_size), len(source)), random_state=42)

try:
    scored = score_data(source)
except (ValueError, KeyError) as error:
    st.error(str(error))
    st.stop()

st.header("1. Dataset summary")
summary = st.columns(4)
summary[0].metric("Transactions", f"{len(scored):,}")
summary[1].metric("Total amount", f"${scored['Amount'].sum():,.2f}")
summary[2].metric("Time span", f"{(scored['Time'].max() - scored['Time'].min()) / 3600:.1f} hours")
if "Class" in scored:
    summary[3].metric("Labeled frauds", f"{int(scored['Class'].sum()):,}")

st.header("2. Benford's Law")
benford_data = scored.copy()
expected, actual, p_value = benford_test(benford_data)
mad = float((actual - expected).abs().mean())
verdict = "close conformity" if mad < .006 else "acceptable" if mad < .012 else "marginal" if mad < .015 else "nonconformity"
fig, ax = plt.subplots()
ax.bar(actual.index, actual.values, label="Actual")
ax.plot(expected.index, expected.values, color="darkorange", marker="o", label="Benford expected")
ax.set(xlabel="Leading digit", ylabel="Frequency", xticks=range(1, 10))
ax.legend()
st.pyplot(fig)
plt.close(fig)
st.caption(f"MAD: {mad:.4f} ({verdict}) · Chi-square p-value: {p_value:.3g}")

st.header("3. Highest risk transactions")
columns = ["txn_id", "Time", "Amount", "risk", "label", "reasons"] + (["Class"] if "Class" in scored else [])
flagged = scored.loc[scored["reasons"].ne("")].head(top_n)
st.dataframe(flagged[columns], use_container_width=True, hide_index=True)

if "Class" in scored:
    st.header("4. Evaluation")
    metrics = evaluate(scored)
    st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)

st.header("5. Download")
csv = flagged[columns].to_csv(index=False).encode("utf-8")
st.download_button("Download flagged transactions", csv, "flagged_transactions.csv", "text/csv")

with st.expander("Limitations"):
    st.markdown(
        """The dataset is PCA-anonymized, so flags cannot be explained in business terms such as vendor, category, or approver. Fraud is about 0.17% of rows, making accuracy meaningless; precision@k and PR-AUC are more useful. Benford's Law is not proof of fraud, can fail on fixed-range or assigned numbers, and can flag ordinary variation at large sample sizes. Unsupervised detection creates false positives that need human review. The weights and thresholds are heuristic."""
    )
