import os
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 분자 특징 계산 함수
def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 6
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
    ]

def train(train_data_path: str, save_path: str):
    # 데이터 로드
    df = pd.read_csv(train_data_path)
    if "Drug" not in df.columns or "Y" not in df.columns:
        raise ValueError("CSV 파일에 'Drug' 또는 'Y' 열이 필요합니다.")
    
    # SMILES 문자열에서 특징 추출
    features = df["Drug"].apply(featurize_smiles).tolist()
    x = np.array(features, dtype=float)
    y = df["Y"].values

    # nan 값 처리
    col_means = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_means, inds[1])

    # 학습, 검증 데이터 분리
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # 모델 정의
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # 모델 학습
    model.fit(x_train, y_train)

    # 검증 성능 출력
    y_pred = model.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))  # squared 제거
    mae = mean_absolute_error(y_val, y_pred)
    print(f"[Validation] RMSE={rmse:.4f}, MAE={mae:.4f}")

    # 전체 데이터로 재학습 후 저장
    model.fit(x, y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"✅ 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="학습 CSV 파일 경로")
    parser.add_argument("--save_path", type=str, default="model2/model.pt", help="저장할 모델 경로")
    args = parser.parse_args()
    train(args.train_csv, args.save_path)
