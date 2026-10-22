"""선행 미션의 데이터와 학습 산출물을 바이트 단위로 연결한다."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.source / "models/metadata.json").read_text())
    if digest(args.data) != metadata["data_sha256"]:
        raise ValueError("A5-1 학습 데이터와 해시가 다릅니다")
    if digest(args.source / "models/classifier.joblib") != metadata["model_sha256"]:
        raise ValueError("A5-1 분류 모델과 해시가 다릅니다")
    Path("models").mkdir(exist_ok=True)
    for name in ("metadata.json", "classifier.joblib", "split.npz"):
        shutil.copyfile(args.source / "models" / name, Path("models") / name)
    shutil.copyfile(args.source / "finance.py", "finance.py")
    shutil.copyfile(args.data, "finance_data.csv")
    print(
        json.dumps(
            {
                "data_sha256": metadata["data_sha256"],
                "model_sha256": metadata["model_sha256"],
                "exact_copy": True,
            }
        )
    )


if __name__ == "__main__":
    main()
