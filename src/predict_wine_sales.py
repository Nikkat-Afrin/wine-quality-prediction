"""
Wine sales prediction — regression on top of the data-cleaning notebook.

The course notebook cleans this 12,795-row wine dataset but stops before
modeling. This script adds the predictive layer: it predicts TARGET (the
number of cases of wine purchased, 0-8) from chemical properties and review
signals, comparing three regressors.

Feature engineering note: a missing STARS rating is itself highly predictive
(unrated wines sell poorly), so we add a `STARS_missing` indicator before
imputation.

Writes reports/model_comparison.md and reports/figures/*.png.
Run from repo root:  python src/predict_wine_sales.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load():
    df = pd.read_csv(ROOT / "data" / "wine_data.csv").drop(columns=["INDEX"], errors="ignore")
    y = df["TARGET"].astype(float)
    X = df.drop(columns=["TARGET"])
    if "STARS" in X.columns:
        X["STARS_missing"] = X["STARS"].isna().astype(int)   # predictive: unrated -> low sales
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    return X, y


def main():
    X, y = load()
    print(f"rows={len(y)}  features={X.shape[1]}  TARGET mean={y.mean():.2f} (cases sold, 0-8)")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RNG)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RNG, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RNG),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        rows.append({"Model": name,
                     "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
                     "MAE": mean_absolute_error(y_te, pred),
                     "R2": r2_score(y_te, pred)})
        fitted[name] = m

    res = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    (ROOT / "reports").mkdir(exist_ok=True)
    cols = ["Model", "RMSE", "MAE", "R2"]
    fmt = lambda v: v if isinstance(v, str) else f"{v:.3f}"
    lines = ["# Test-set regression comparison (predicting wine cases sold)", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
    for _, r in res.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    (ROOT / "reports" / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = res.iloc[0]["Model"]
    # Feature importance (best tree model, fall back to RF)
    tree = fitted.get(best if best != "Linear Regression" else "Random Forest")
    imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values().tail(12)
    plt.figure(figsize=(8, 6)); imp.plot.barh(color="#7b1fa2")
    plt.title("Feature importance — predicting wine cases sold"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=120); plt.close()

    pred = fitted[best].predict(X_te)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_te, pred, s=8, alpha=0.3, color="#6a1b9a")
    plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "k--", alpha=0.6)
    plt.xlabel("Actual cases sold"); plt.ylabel("Predicted"); plt.title(f"Predicted vs actual — {best}")
    plt.tight_layout(); plt.savefig(FIG / "predicted_vs_actual.png", dpi=120); plt.close()
    print(f"Best by R2: {best}\nWritten to {ROOT/'reports'}")


if __name__ == "__main__":
    main()
