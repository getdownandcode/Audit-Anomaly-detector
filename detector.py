"""Explainable transaction anomaly checks and evaluation."""

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid positive transactions, preserve their index, and derive hour."""
    result = df.copy()
    required = {"Amount", "Time"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    result = result.loc[result["Amount"].notna() & (result["Amount"] > 0)].copy()
    result["txn_id"] = result.index
    result["hour"] = (result["Time"] // 3600) % 24
    return result


def benford_test(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    """Add leading digits and return Benford expected/observed rates and chi-square p-value."""
    expected = pd.Series({d: np.log10(1 + 1 / d) for d in range(1, 10)}, name="expected")
    amounts = df["Amount"].where(df["Amount"] >= 1)
    digits = amounts.dropna().astype(str).str.replace(r"^0+\.?0*", "", regex=True).str[0]
    df["leading_digit"] = pd.to_numeric(digits.reindex(df.index), errors="coerce").astype("Int64")
    observed = df["leading_digit"].value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    observed.name = "actual"
    counts = df["leading_digit"].dropna().value_counts().reindex(range(1, 10), fill_value=0)
    p_value = float(chisquare(counts.to_numpy(), f_exp=expected.to_numpy() * counts.sum()).pvalue) if counts.sum() else 1.0
    return expected, observed, p_value


def benford_flag(df: pd.DataFrame, actual: pd.Series, expected: pd.Series, threshold: float = 0.02) -> pd.Series:
    """Flag amounts whose leading digit is overrepresented beyond threshold."""
    overrepresented = actual.index[(actual - expected.reindex(actual.index)).gt(threshold)]
    digit = df.get("leading_digit", pd.Series(index=df.index, dtype="Int64"))
    return digit.isin(overrepresented).fillna(False).astype(bool)


def rule_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Return simple amount/time audit rules and readable reasons."""
    time = df["Time"]
    order = np.argsort(time.to_numpy(), kind="stable")
    amounts = df["Amount"].to_numpy()
    sorted_amounts = amounts[order]
    sorted_time = time.to_numpy()[order]
    repeat = np.zeros(len(df), dtype=bool)
    for amount in np.unique(amounts):
        positions = order[sorted_amounts == amount]
        times = time.to_numpy()[positions]
        for start in range(len(positions)):
            end = np.searchsorted(times, times[start] + 3600, side="right")
            if end - start >= 3:
                repeat[positions[start:end]] = True
    checks = pd.DataFrame(index=df.index)
    checks["round_amount"] = (df["Amount"] >= 100) & (df["Amount"] % 100 == 0)
    checks["repeat_amount"] = repeat
    checks["night_txn"] = df["hour"].between(0, 5)
    checks["high_amount"] = df["Amount"] > df["Amount"].quantile(0.99)
    labels = {"round_amount": "Round amount", "repeat_amount": "Repeated amount", "night_txn": "Night transaction", "high_amount": "High amount"}
    checks["reasons"] = checks[list(labels)].apply(lambda row: ", ".join(labels[c] for c, yes in row.items() if yes), axis=1)
    return checks


def isolation_forest_score(df: pd.DataFrame) -> pd.Series:
    """Score transactions from zero to one, with larger values more anomalous."""
    features = pd.DataFrame({"log_amount": np.log1p(df["Amount"]), "hour": df["hour"]}, index=df.index)
    features = pd.concat([features, df[[f"V{i}" for i in range(1, 29) if f"V{i}" in df]]], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(features.median()).fillna(0)
    raw = -IsolationForest(n_estimators=100, contamination=0.002, random_state=42, n_jobs=-1).fit(features).score_samples(features)
    span = np.ptp(raw)
    return pd.Series((raw - raw.min()) / span if span else np.zeros(len(raw)), index=df.index, name="iso_score")


def risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine anomaly scores and rules into a ranked transaction table."""
    result = df.copy()
    checks = rule_checks(result)
    result[checks.columns] = checks
    result["iso_score"] = isolation_forest_score(result)
    if "leading_digit" not in result:
        amounts = result["Amount"].where(result["Amount"] >= 1).dropna().astype(str).str.replace(r"^0+\.?0*", "", regex=True).str[0]
        result["leading_digit"] = pd.to_numeric(amounts.reindex(result.index), errors="coerce").astype("Int64")
    digit_counts = result["leading_digit"].value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    expected = pd.Series({d: np.log10(1 + 1 / d) for d in range(1, 10)})
    result["benford_flag"] = benford_flag(result, digit_counts, expected)
    rule_columns = ["round_amount", "repeat_amount", "night_txn", "high_amount"]
    result["risk"] = (0.6 * result["iso_score"] + 0.1 * checks[rule_columns].sum(axis=1).clip(upper=3) + 0.1 * result["benford_flag"]).clip(0, 1)
    result["label"] = np.select([result["risk"] >= 0.7, result["risk"] >= 0.4], ["High", "Medium"], default="Low")
    result.loc[result["iso_score"] > 0.7, "reasons"] = result.loc[result["iso_score"] > 0.7, "reasons"].map(lambda s: ", ".join(filter(None, [s, "Isolation Forest outlier"])))
    result.loc[result["benford_flag"], "reasons"] = result.loc[result["benford_flag"], "reasons"].map(lambda s: ", ".join(filter(None, [s, "Benford digit overrepresented"])))
    return result.sort_values("risk", ascending=False)


def evaluate(df: pd.DataFrame, k: int = 500) -> dict[str, dict[str, float]]:
    """Compare combined, rules-only, and Isolation Forest ranking metrics when labels exist."""
    if "Class" not in df:
        return {}
    y = df["Class"].astype(int)
    rule_score = rule_checks(df).drop(columns="reasons").sum(axis=1).astype(float)
    scores = {"Combined": df["risk"], "Rules only": rule_score, "Isolation Forest only": df["iso_score"]}
    results = {}
    for name, score in scores.items():
        top = y.loc[score.sort_values(ascending=False).head(min(k, len(df))).index]
        results[name] = {
            f"precision@{k}": float(top.mean()) if len(top) else 0.0,
            f"recall@{k}": float(top.sum() / y.sum()) if y.sum() else 0.0,
            "ROC-AUC": float(roc_auc_score(y, score)) if y.nunique() > 1 else float("nan"),
            "PR-AUC": float(average_precision_score(y, score)) if y.nunique() > 1 else float("nan"),
        }
    return results
