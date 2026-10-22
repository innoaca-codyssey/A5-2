"""원본6변수로 K-Means/PCA와 DBSCAN을 비교한다."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from finance import FEATURES
from prepare import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("finance_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    metadata = json.loads(Path("models/metadata.json").read_text())
    if digest(args.data) != metadata["data_sha256"]:
        raise ValueError("원본 데이터 해시 불일치")
    df = pd.read_csv(args.data)
    raw = df[FEATURES]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    args.output.mkdir(parents=True, exist_ok=True)
    selection, models = [], []
    with threadpool_limits(limits=2):
        for k in range(2, 9):
            model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
            score = silhouette_score(
                scaled, model.labels_, sample_size=2000, random_state=42
            )
            selection.append(
                {"k": k, "inertia": float(model.inertia_), "silhouette": float(score)}
            )
            models.append(model)
        best_index = int(np.argmax([r["silhouette"] for r in selection]))
        best = models[best_index]
        labels = best.labels_
        alternatives = []
        for eps in (0.5, 0.8, 1.1):
            db = DBSCAN(eps=eps, min_samples=15, n_jobs=2).fit(scaled)
            kept = db.labels_ >= 0
            clusters = len(np.unique(db.labels_[kept]))
            score = None
            if clusters >= 2 and kept.sum() > clusters:
                score = float(
                    silhouette_score(
                        scaled[kept],
                        db.labels_[kept],
                        sample_size=min(2000, int(kept.sum())),
                        random_state=42,
                    )
                )
            alternatives.append(
                {
                    "eps": eps,
                    "min_samples": 15,
                    "clusters": clusters,
                    "noise_fraction": float((~kept).mean()),
                    "silhouette_non_noise": score,
                }
            )
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(scaled)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        [r["k"] for r in selection], [r["inertia"] for r in selection], marker="o"
    )
    axes[0].set(xlabel="K", ylabel="inertia")
    axes[1].plot(
        [r["k"] for r in selection], [r["silhouette"] for r in selection], marker="o"
    )
    axes[1].set(xlabel="K", ylabel="silhouette (same seeded sample)")
    fig.tight_layout()
    fig.savefig(args.output / "k-selection.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(best.n_clusters):
        mask = labels == k
        ax.scatter(
            points[mask, 0], points[mask, 1], s=5, alpha=0.35, label=f"Cluster {k}"
        )
    ax.set(
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        title=f"PCA total explained variance {pca.explained_variance_ratio_.sum():.1%}",
    )
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(args.output / "pca.png", dpi=150)
    plt.close(fig)
    classifier = joblib.load("models/classifier.joblib")
    risk = classifier.predict_proba(raw)[:, 1]
    personas = []
    for k in range(best.n_clusters):
        mask = labels == k
        personas.append(
            {
                "cluster": k,
                "count": int(mask.sum()),
                "mean": raw.loc[mask].mean().to_dict(),
                "median": raw.loc[mask].median().to_dict(),
                "predicted_overdue_mean": float(risk[mask].mean()),
                "observed_overdue_fraction_posthoc": float(
                    df.is_overdue.loc[mask].mean()
                ),
            }
        )
    np.savez(args.output / "cluster-labels.npz", labels=labels)
    joblib.dump(
        {"scaler": scaler, "kmeans": best, "pca": pca}, "models/clustering.joblib"
    )
    result = {
        "features": FEATURES,
        "data_sha256": digest(args.data),
        "selection": selection,
        "selected_k": int(best.n_clusters),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "personas": personas,
        "dbscan": alternatives,
    }
    (args.output / "clustering.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
