from __future__ import annotations

import os
import json
import math
import joblib
import warnings
import numpy as np
import pandas as pd

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

# Spearman 상관계수 계산
try:
    from scipy.stats import spearmanr
    _has_scipy = True
except Exception:
    _has_scipy = False

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# 사용할 분자 특징 정의
DESCRIPTOR_NAMES = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHBA",
    "NumHBD",
    "NumRotatableBonds",
    "RingCount",
]

# 특징(feature) 계산 함수
def _compute_descriptors(mol: Chem.Mol) -> np.ndarray:
    """RDKit Mol 객체에서 기본적인 물리·화학 descriptor 계산"""
    if mol is None:
        # 분자 생성 실패 시 NaN으로 채움
        return np.array([np.nan] * len(DESCRIPTOR_NAMES), dtype=float)
    return np.array([
        Descriptors.MolWt(mol),  # 분자량
        Descriptors.MolLogP(mol),  # 소수성
        Descriptors.TPSA(mol),  # 극성 표면적
        rdMolDescriptors.CalcNumHBA(mol),  # 수소 결합 수용자 개수
        rdMolDescriptors.CalcNumHBD(mol),  # 수소 결합 공여자 개수
        rdMolDescriptors.CalcNumRotatableBonds(mol),  # 회전 가능한 결합 수
        rdMolDescriptors.CalcNumSaturatedRings(mol) + rdMolDescriptors.CalcNumAromaticRings(mol),  # 고리 개수
    ], dtype=float)


def _morgan_bits(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan fingerprint를 numpy array(0/1 비트 벡터)로 변환"""
    if mol is None:
        return np.zeros((n_bits,), dtype=np.uint8)
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def _smiles_to_mol(smiles: str) -> Chem.Mol | None:
    """SMILES 문자열을 RDKit Mol 객체로 변환"""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 분자 구조 정제
    Chem.SanitizeMol(mol, catchErrors=True)
    return mol


def featurize_smiles(smiles_list, radius=2, n_bits=2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    여러 개의 SMILES 문자열을 입력받아
    (Morgan fingerprint, descriptor 배열) 반환
    """
    fps = []
    descs = []
    for smi in smiles_list:
        mol = _smiles_to_mol(smi)
        fps.append(_morgan_bits(mol, radius=radius, n_bits=n_bits))
        descs.append(_compute_descriptors(mol))
    return np.asarray(fps, dtype=np.uint8), np.asarray(descs, dtype=float)

# 구성 클래스
@dataclass
class FeatureConfig:
    """특징(feature) 관련 설정"""
    radius: int = 2
    n_bits: int = 2048
    descriptor_names: Tuple[str, ...] = tuple(DESCRIPTOR_NAMES)
    y_transform: str = "log1p"  # 타겟값 변환 방식: "none", "log1p"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["descriptor_names"] = list(self.descriptor_names)
        return d

# 타겟값 변환 함수
def _transform_y(y: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """타겟값 Y 변환 (로그 변환 등)"""
    if mode == "log1p":
        return np.log1p(y.clip(min=0)), {"mode": mode}
    return y, {"mode": "none"}


def _inverse_transform_y(yhat: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    """예측값 역변환 (로그 변환 복원 등)"""
    mode = info.get("mode", "none")
    if mode == "log1p":
        return np.expm1(yhat)
    return yhat

# 데이터 로드
def _load_training_csv(path: str) -> pd.DataFrame:
    """CSV 파일 로드 및 열 이름 표준화"""
    df = pd.read_csv(path)
    # 다양한 이름을 표준 이름으로 변경
    rename_map = {
        "Drug_ID": "Drug_ID",
        "Drug": "Drug",
        "SMILES": "Drug",
        "Y": "Y",
        "Target": "Y", 
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "Drug" not in df.columns or "Y" not in df.columns:
        raise ValueError("CSV에 'Drug'(SMILES)와 'Y' 열이 반드시 필요합니다.")
    return df[["Drug_ID"] + ["Drug", "Y"]] if "Drug_ID" in df.columns else df[["Drug", "Y"]]

# 학습 함수
def train(train_data_path: str, save_path: str):
    """
    주어진 CSV로부터 모델 학습 후 save_path에 저장
    :param train_data_path: 학습용 CSV 경로
    :param save_path: 저장할 모델 경로 (joblib 형식)
    """
    # 데이터 로드
    df = _load_training_csv(train_data_path).copy()
    # NaN 값 제거
    df = df.dropna(subset=["Drug", "Y"]).reset_index(drop=True)

    cfg = FeatureConfig()

    # 특징 추출
    fps, descs = featurize_smiles(df["Drug"].tolist(), radius=cfg.radius, n_bits=cfg.n_bits)

    # descriptor에 NaN이 있으면 컬럼 중앙값으로 대체
    if np.isnan(descs).any():
        col_medians = np.nanmedian(descs, axis=0)
        inds = np.where(np.isnan(descs))
        descs[inds] = np.take(col_medians, inds[1])

    # 타겟값 변환
    y_raw = df["Y"].astype(float).values
    y_tr, y_info = _transform_y(y_raw, cfg.y_transform)

    # 학습/검증 데이터 분리
    X_desc_train, X_desc_val, X_fp_train, X_fp_val, y_train, y_val = train_test_split(
        descs, fps, y_tr, test_size=0.15, random_state=42
    )

    # descriptor는 scaling 적용 (fingerprint는 0/1이므로 scaling 불필요)
    scaler = StandardScaler()
    X_desc_train_scaled = scaler.fit_transform(X_desc_train)
    X_desc_val_scaled = scaler.transform(X_desc_val)

    # fingerprint + descriptor 결합
    X_train = np.hstack([X_fp_train.astype(np.float32), X_desc_train_scaled.astype(np.float32)])
    X_val = np.hstack([X_fp_val.astype(np.float32), X_desc_val_scaled.astype(np.float32)])

    # 회귀 모델 정의 (Gradient Boosting)
    model = HistGradientBoostingRegressor(
        max_depth=None,
        learning_rate=0.06,
        max_iter=600,
        l2_regularization=0.0,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 검증 데이터 예측
    y_val_pred_tr = model.predict(X_val)
    y_val_pred = _inverse_transform_y(y_val_pred_tr, y_info)
    y_val_true = _inverse_transform_y(y_val, y_info)

    # 검증 지표 계산
    rmse = math.sqrt(mean_squared_error(y_val_true, y_val_pred))
    mae = mean_absolute_error(y_val_true, y_val_pred)
    r2 = r2_score(y_val_true, y_val_pred)
    if _has_scipy:
        sp, _ = spearmanr(y_val_true, y_val_pred)
    else:
        sp = pd.Series(y_val_true).rank().corr(pd.Series(y_val_pred).rank(), method="pearson")

    print(f"[Validation] RMSE={rmse:.4f}  MAE={mae:.4f}  Spearman={sp:.4f}  R2={r2:.4f}")

    # 전체 데이터로 최종 학습
    scaler_full = StandardScaler()
    descs_scaled_full = scaler_full.fit_transform(descs.astype(np.float32))
    X_full = np.hstack([fps.astype(np.float32), descs_scaled_full.astype(np.float32)])

    model_full = HistGradientBoostingRegressor(
        max_depth=None,
        learning_rate=0.06,
        max_iter=600,
        l2_regularization=0.0,
        min_samples_leaf=20,
        random_state=42
    ).fit(X_full, y_tr)

    # 체크포인트 저장
    ckpt = {
        "feature_config": cfg.to_dict(),
        "descriptor_scaler": scaler_full,
        "model": model_full,
        "y_info": y_info,
        "columns": {
            "smiles": "Drug",
            "target": "Y",
        }
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(ckpt, save_path)
    print(f"✅ 모델 저장 완료: {save_path}")

# 스크립트 실행부
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="학습용 CSV 경로")
    parser.add_argument("--save_path", type=str, default="model/model.pkl", help="저장할 모델 경로")
    args = parser.parse_args()
    train(args.train_csv, args.save_path)
