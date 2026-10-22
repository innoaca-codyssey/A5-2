"""같은 분류 모델의 연체 확률을 Global/Local SHAP으로 설명한다."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from finance import FEATURES
from prepare import digest


def explain(pipeline, raw):
    """내부 표준화 좌표를 설명한 뒤 원본 단위와 기여도를 연결한다."""
    transformed = pipeline[:-1].transform(raw)
    forest = pipeline.named_steps["model"]
    class_index = int(np.flatnonzero(forest.classes_ == 1)[0])
    result = shap.TreeExplainer(forest, feature_perturbation="tree_path_dependent")(
        transformed
    )
    values = result.values[:, :, class_index]
    base = result.base_values[:, class_index]
    explanation = shap.Explanation(
        values=values,
        base_values=base,
        data=raw[FEATURES].to_numpy(),
        feature_names=FEATURES,
    )
    probability = pipeline.predict_proba(raw)[:, class_index]
    residual = probability - (base + values.sum(axis=1))
    if np.max(np.abs(residual)) > 1e-6:
        raise ValueError("SHAP 가산성 검증 실패")
    return explanation, probability, residual


def save_plot(path):
    """현재 SHAP 그림을 경계 잘림 없이 저장한다."""
    plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("finance_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path("models/metadata.json").read_text())
    if (
        digest(args.data) != meta["data_sha256"]
        or digest("models/classifier.joblib") != meta["model_sha256"]
    ):
        raise ValueError("선행 데이터/모델 해시 불일치")
    df = pd.read_csv(args.data)
    raw = df[FEATURES]
    classifier = joblib.load("models/classifier.joblib")
    test_indices = np.load("models/split.npz")["test"]
    indices = np.random.default_rng(42).choice(test_indices, size=1000, replace=False)
    explanation, probability, residual = explain(classifier, raw.iloc[indices])
    shap.plots.beeswarm(explanation, max_display=6, show=False)
    save_plot(args.output / "shap-summary.png")
    importance = np.abs(explanation.values).mean(axis=0)
    rank = np.argsort(importance)[::-1]
    directions = {}
    for j, feature in enumerate(FEATURES):
        x = explanation.data[:, j]
        lo, hi = np.quantile(x, [0.25, 0.75])
        directions[feature] = {
            "mean_absolute_shap": float(importance[j]),
            "lower_quartile_shap_mean": float(explanation.values[x <= lo, j].mean()),
            "upper_quartile_shap_mean": float(explanation.values[x >= hi, j].mean()),
        }
    dependence = list(
        dict.fromkeys([FEATURES[rank[0]], FEATURES[rank[1]], "debt_ratio"])
    )
    for feature in dependence:
        shap.dependence_plot(
            feature,
            explanation.values,
            raw.iloc[indices],
            feature_names=FEATURES,
            interaction_index=None,
            show=False,
        )
        plt.ylabel("SHAP contribution to P(overdue=1)")
        save_plot(args.output / f"dependence-{feature}.png")
    full_prob = classifier.predict_proba(raw.iloc[test_indices])[:, 1]
    selected = {
        "approval": int(test_indices[np.argmin(full_prob)]),
        "rejection": int(test_indices[np.argmax(full_prob)]),
    }
    local, local_p, local_r = explain(classifier, raw.iloc[list(selected.values())])
    cases = []
    for pos, (decision, row) in enumerate(selected.items()):
        shap.plots.waterfall(local[pos], max_display=6, show=False)
        save_plot(args.output / f"waterfall-{decision}.png")
        cases.append(
            {
                "decision": decision,
                "row_index": row,
                "features": raw.iloc[row].to_dict(),
                "true_is_overdue": int(df.is_overdue.iloc[row]),
                "probability": float(local_p[pos]),
                "base_value": float(local.base_values[pos]),
                "contributions": dict(zip(FEATURES, local.values[pos].tolist())),
                "additivity_residual": float(local_r[pos]),
            }
        )
    if not cases[0]["probability"] < 0.5 or not cases[1]["probability"] >= 0.5:
        raise ValueError("승인/거절 두 범주를 찾지 못했습니다")
    clusters = joblib.load("models/clustering.joblib")
    scaled = clusters["scaler"].transform(raw)
    labels = np.load(args.output / "cluster-labels.npz")["labels"]
    representatives = []
    for cluster in range(clusters["kmeans"].n_clusters):
        members = np.flatnonzero(labels == cluster)
        distances = np.linalg.norm(
            scaled[members] - clusters["kmeans"].cluster_centers_[cluster], axis=1
        )
        row = int(members[np.argmin(distances)])
        exp, prob, error = explain(classifier, raw.iloc[[row]])
        shap.plots.waterfall(exp[0], max_display=6, show=False)
        save_plot(args.output / f"waterfall-cluster-{cluster}.png")
        representatives.append(
            {
                "cluster": cluster,
                "row_index": row,
                "centroid_distance": float(distances.min()),
                "features": raw.iloc[row].to_dict(),
                "probability": float(prob[0]),
                "contributions": dict(zip(FEATURES, exp.values[0].tolist())),
            }
        )
    result = {
        "data_sha256": meta["data_sha256"],
        "model_sha256": meta["model_sha256"],
        "sample": "seed42, 1000 distinct Test rows",
        "maximum_additivity_error": float(np.abs(residual).max()),
        "ranking": [FEATURES[i] for i in rank],
        "cluster_representatives": representatives,
        "global": directions,
        "dependence": dependence,
        "cases": cases,
        "threshold": 0.5,
        "original_threshold_note": "원문 확률5 이상은 불가능하므로0.5로 해석한 실행 기준",
    }
    np.savez(
        args.output / "shap-values.npz",
        indices=indices,
        values=explanation.values,
        base_values=explanation.base_values,
    )
    (args.output / "shap-analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
