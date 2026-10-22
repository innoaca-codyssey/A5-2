"""금융 입력 검증, 전처리와 규칙 기반 비교 모델."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "age",
    "annual_income",
    "spending_score",
    "debt_ratio",
    "credit_card_count",
    "overdue_count_6m",
]


class SchemaValidator(TransformerMixin, BaseEstimator):
    """입력6열만 허용하고 타겟의 우발적 혼입을 거절한다."""

    def fit(self, X, y=None):
        self.transform(X)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or set(X.columns) != set(FEATURES):
            raise ValueError("고정된 입력6개 열이 필요합니다")
        result = X.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
        if np.isinf(result.to_numpy()).any():
            raise ValueError("입력에 무한대가 있습니다")
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)


def preprocessor(encode_card=False):
    """수치 대치/스케일링과 선택적인 카드 수 인코딩을 구성한다."""
    numeric = [f for f in FEATURES if not (encode_card and f == "credit_card_count")]
    steps = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if encode_card:
        steps.append(
            (
                "card_category",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                ["credit_card_count"],
            )
        )
    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)


def pipeline(model, encode_card=False):
    """검증/전처리/모델 전체를 교차 검증의 단위로 묶는다."""
    return Pipeline(
        [
            ("schema", SchemaValidator()),
            ("preprocess", preprocessor(encode_card)),
            ("model", model),
        ]
    )


def rule_scores(X):
    """사전에 정한 if/elif 규칙의 순위 점수를 반환한다."""
    checked = SchemaValidator().transform(X)
    scores = []
    for r in checked.itertuples(index=False):
        if r.overdue_count_6m >= 2:
            score = 0.9
        elif r.debt_ratio >= 0.8 and r.annual_income < 5000:
            score = 0.8
        elif r.annual_income <= 2500:
            score = 0.7
        elif r.overdue_count_6m >= 1 and r.debt_ratio >= 0.5:
            score = 0.65
        elif r.age < 25 and r.debt_ratio > 0.7:
            score = 0.6
        elif r.annual_income >= 8000 and r.overdue_count_6m == 0:
            score = 0.05
        else:
            score = 0.15
        scores.append(score)
    return np.asarray(scores)
