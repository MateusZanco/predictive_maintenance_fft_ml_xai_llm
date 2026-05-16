from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier


MAPA_CLASSES = {
    0: "Normal",
    1: "Desgaste Superficial",
    2: "Dente Trincado",
    3: "Dente Lascado",
    4: "Dente Ausente",
}
AXES = ("x", "y", "z")
FS_PADRAO = 10_000.0
PONTOS_POR_LINHA_PADRAO = 200

ROTACAO_ENTRADA_RPM = 1500.0
ZR1 = 100.0
ZS1 = 20.0
REDUCAO_PRIMEIRO_ESTAGIO = 6.0
ZR2 = 100.0
ZS2 = 28.0

FSH1 = ROTACAO_ENTRADA_RPM / 60.0
FM1 = ((ZR1 * ZS1) / (ZR1 + ZS1)) * FSH1
FCSd1 = FM1 / ZS1
FCSL1 = 3 * FCSd1

FSH2 = (ROTACAO_ENTRADA_RPM / REDUCAO_PRIMEIRO_ESTAGIO) / 60.0
FM2 = ((ZR2 * ZS2) / (ZR2 + ZS2)) * FSH2
FCSd2 = FM2 / ZS2
FCSL2 = 4 * FCSd2


@dataclass(slots=True)
class AnalysisParams:
    dataset_key: str
    fs: float = FS_PADRAO
    pontos_por_linha: int = PONTOS_POR_LINHA_PADRAO
    duracao_intervalo_s: float = 1.0
    largura_busca_fm_real_hz: float = 10.0
    largura_banda_harmonica_hz: float = 10.0
    ordens_harmonicas_fm: int = 5
    ordens_harmonicas_fm1: int = 5
    ressonancia_min_hz: float = 3000.0
    ressonancia_max_hz: float = 5000.0
    test_size: float = 0.3
    explanation_samples_per_class: int = 3
    shap_top_k: int = 15
    random_state: int = 42


@dataclass(slots=True)
class AnalysisArtifacts:
    params: AnalysisParams
    features_modelo: pd.DataFrame
    segmentos_por_eixo_classe: dict[str, dict[int, np.ndarray]]
    X_features: pd.DataFrame
    y_features: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: list[int]
    test_indices: list[int]
    X_explicacao: pd.DataFrame
    y_explicacao: pd.Series
    amostras_explicacao_info: pd.DataFrame
    random_forest: RandomForestClassifier
    xgboost_model: XGBClassifier
    svm_model: Pipeline
    y_pred_rf: np.ndarray
    y_pred_xgb: np.ndarray
    y_pred_svm: np.ndarray
    y_proba_rf: np.ndarray
    y_proba_xgb: np.ndarray
    y_proba_svm: np.ndarray
    prediction_catalog: dict[str, list[dict[str, Any]]]
    artifact_dir: Path


def _dataset_dirs(repo_root: Path) -> list[Path]:
    return [repo_root / "data" / "raw", repo_root / "data"]


def list_datasets(repo_root: Path) -> list[dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    for base_dir in _dataset_dirs(repo_root):
        if not base_dir.exists():
            continue
        for gt_file in base_dir.glob("gt_*.npy"):
            key = gt_file.stem.replace("gt_", "")
            axes_exist = all((base_dir / f"{axis}_{key}.npy").exists() for axis in AXES)
            if not axes_exist:
                continue
            datasets[key] = {
                "key": key,
                "label": key.replace("_", " | "),
                "path": str(base_dir),
            }
    return sorted(datasets.values(), key=lambda item: item["key"])


def _find_dataset_base(repo_root: Path, dataset_key: str) -> Path:
    for base_dir in _dataset_dirs(repo_root):
        if (base_dir / f"gt_{dataset_key}.npy").exists():
            return base_dir
    raise FileNotFoundError(f"Dataset '{dataset_key}' nao encontrado em data/raw nem em data.")


def _load_dataset(repo_root: Path, dataset_key: str) -> dict[str, np.ndarray]:
    base_dir = _find_dataset_base(repo_root, dataset_key)
    return {
        "x": np.load(base_dir / f"x_{dataset_key}.npy"),
        "y": np.load(base_dir / f"y_{dataset_key}.npy"),
        "z": np.load(base_dir / f"z_{dataset_key}.npy"),
        "gt": np.load(base_dir / f"gt_{dataset_key}.npy"),
    }


def _segment_by_class(arrays: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    segmented: dict[str, dict[int, np.ndarray]] = {axis: {} for axis in AXES}
    for axis in AXES:
        df_axis = pd.DataFrame(arrays[axis])
        df_axis["classe"] = labels
        for classe in sorted(MAPA_CLASSES):
            segmented[axis][classe] = df_axis[df_axis["classe"] == classe].drop(columns="classe").to_numpy()
    return segmented


def _reshape_segments(
    segmented_rows: dict[str, dict[int, np.ndarray]],
    amostras_por_intervalo: int,
) -> dict[str, dict[int, np.ndarray]]:
    output: dict[str, dict[int, np.ndarray]] = {axis: {} for axis in AXES}
    for axis in AXES:
        for classe, rows in segmented_rows[axis].items():
            output[axis][classe] = rows.reshape(-1, amostras_por_intervalo)
    return output


def _calcular_espectro_amplitude(sinal_segmento: np.ndarray, janela: np.ndarray) -> np.ndarray:
    sinal_centrado = sinal_segmento - np.mean(sinal_segmento)
    sinal_janelado = sinal_centrado * janela
    espectro = np.fft.rfft(sinal_janelado)
    amplitude = np.abs(espectro) * 2.0 / janela.sum()
    amplitude[0] = 0.0
    return amplitude


def _extrair_pico_na_banda(
    amplitude_espectral: np.ndarray,
    frequencias: np.ndarray,
    mascara_banda: np.ndarray,
) -> tuple[float, float]:
    if not np.any(mascara_banda):
        return np.nan, np.nan
    amplitudes_banda = amplitude_espectral[mascara_banda]
    frequencias_banda = frequencias[mascara_banda]
    indice_pico = int(np.argmax(amplitudes_banda))
    return float(amplitudes_banda[indice_pico]), float(frequencias_banda[indice_pico])


def _extrair_soma_amplitudes_na_banda(amplitude_espectral: np.ndarray, mascara_banda: np.ndarray) -> float:
    if not np.any(mascara_banda):
        return np.nan
    return float(np.sum(np.square(amplitude_espectral[mascara_banda])))


def _extrair_amplitude_na_frequencia_alvo(
    amplitude_espectral: np.ndarray,
    frequencias: np.ndarray,
    frequencia_alvo: float,
) -> float:
    if np.isnan(frequencia_alvo):
        return np.nan
    indice_mais_proximo = int(np.argmin(np.abs(frequencias - frequencia_alvo)))
    return float(amplitude_espectral[indice_mais_proximo])


def _calcular_rms(sinal_segmento: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(sinal_segmento))))


def _calcular_peak_value(sinal_segmento: np.ndarray) -> float:
    return float(np.max(np.abs(sinal_segmento)))


def _calcular_kurtosis(sinal_segmento: np.ndarray) -> float:
    sinal_centrado = sinal_segmento - np.mean(sinal_segmento)
    desvio_padrao = np.std(sinal_centrado)
    if desvio_padrao == 0:
        return np.nan
    quarto_momento = np.mean(np.power(sinal_centrado, 4))
    return float(quarto_momento / np.power(desvio_padrao, 4))


def _calcular_crest_factor(peak_value: float, rms: float) -> float:
    if rms == 0:
        return np.nan
    return float(peak_value / rms)


def _build_features(segmentos_por_eixo_classe: dict[str, dict[int, np.ndarray]], params: AnalysisParams) -> pd.DataFrame:
    linhas_por_intervalo = int((params.duracao_intervalo_s / (params.pontos_por_linha / params.fs)))
    amostras_por_intervalo = linhas_por_intervalo * params.pontos_por_linha
    janela_hann = np.hanning(amostras_por_intervalo)
    frequencias_fft = np.fft.rfftfreq(amostras_por_intervalo, d=1 / params.fs)

    mascara_fm2 = (frequencias_fft >= FM2 - params.largura_busca_fm_real_hz) & (
        frequencias_fft <= FM2 + params.largura_busca_fm_real_hz
    )
    mascara_fm1 = (frequencias_fft >= FM1 - params.largura_busca_fm_real_hz) & (
        frequencias_fft <= FM1 + params.largura_busca_fm_real_hz
    )
    mascara_ressonancia = (frequencias_fft > params.ressonancia_min_hz) & (
        frequencias_fft < params.ressonancia_max_hz
    )

    ordens_harmonicas_fm = list(range(1, params.ordens_harmonicas_fm + 1))
    ordens_harmonicas_fm1 = list(range(1, params.ordens_harmonicas_fm1 + 1))

    features_modelo: list[dict[str, Any]] = []
    for classe in sorted(MAPA_CLASSES):
        quantidade_segmentos = segmentos_por_eixo_classe["x"][classe].shape[0]
        for segmento_id in range(quantidade_segmentos):
            linha: dict[str, Any] = {
                "classe": classe,
                "classe_nome": MAPA_CLASSES[classe],
                "segmento_id": segmento_id,
                "tempo_inicial_s": segmento_id * params.duracao_intervalo_s,
                "tempo_final_s": (segmento_id + 1) * params.duracao_intervalo_s,
            }
            for eixo in AXES:
                sinal_segmento = segmentos_por_eixo_classe[eixo][classe][segmento_id]
                rms = _calcular_rms(sinal_segmento)
                peak_value = _calcular_peak_value(sinal_segmento)
                kurtosis = _calcular_kurtosis(sinal_segmento)

                linha[f"rms_{eixo}"] = rms
                linha[f"peak_value_{eixo}"] = peak_value
                linha[f"kurtosis_{eixo}"] = kurtosis
                linha[f"crest_factor_{eixo}"] = _calcular_crest_factor(peak_value, rms)

                amplitude_espectral = _calcular_espectro_amplitude(sinal_segmento, janela_hann)
                _, fm_real = _extrair_pico_na_banda(amplitude_espectral, frequencias_fft, mascara_fm2)
                _, fm1_real = _extrair_pico_na_banda(amplitude_espectral, frequencias_fft, mascara_fm1)

                linha[f"fm_real_{eixo}"] = fm_real
                linha[f"fm1_real_{eixo}"] = fm1_real
                linha[f"amp_fcsd_esq_{eixo}"] = _extrair_amplitude_na_frequencia_alvo(
                    amplitude_espectral,
                    frequencias_fft,
                    fm_real - FCSd2,
                )
                linha[f"amp_fcsd_dir_{eixo}"] = _extrair_amplitude_na_frequencia_alvo(
                    amplitude_espectral,
                    frequencias_fft,
                    fm_real + FCSd2,
                )
                linha[f"amp_fcsl_esq_{eixo}"] = _extrair_amplitude_na_frequencia_alvo(
                    amplitude_espectral,
                    frequencias_fft,
                    fm_real - FCSL2,
                )
                linha[f"amp_fcsl_dir_{eixo}"] = _extrair_amplitude_na_frequencia_alvo(
                    amplitude_espectral,
                    frequencias_fft,
                    fm_real + FCSL2,
                )
                linha[f"energia_ressonancia_{eixo}"] = _extrair_soma_amplitudes_na_banda(
                    amplitude_espectral,
                    mascara_ressonancia,
                )
                linha[f"amp_fcsd1_dir_{eixo}"] = _extrair_amplitude_na_frequencia_alvo(
                    amplitude_espectral,
                    frequencias_fft,
                    fm1_real + FCSd1,
                )
                linha[f"amp_fcsl1_dir_{eixo}"] = _extrair_amplitude_na_frequencia_alvo(
                    amplitude_espectral,
                    frequencias_fft,
                    fm1_real + FCSL1,
                )

                for ordem_harmonica in ordens_harmonicas_fm:
                    frequencia_alvo = ordem_harmonica * fm_real
                    mascara = (frequencias_fft >= frequencia_alvo - params.largura_banda_harmonica_hz) & (
                        frequencias_fft <= frequencia_alvo + params.largura_banda_harmonica_hz
                    )
                    linha[f"amp_fm_h{ordem_harmonica}_{eixo}"] = _extrair_soma_amplitudes_na_banda(
                        amplitude_espectral,
                        mascara,
                    )

                for ordem_harmonica in ordens_harmonicas_fm1:
                    frequencia_alvo = ordem_harmonica * fm1_real
                    mascara = (frequencias_fft >= frequencia_alvo - params.largura_banda_harmonica_hz) & (
                        frequencias_fft <= frequencia_alvo + params.largura_banda_harmonica_hz
                    )
                    linha[f"amp_fm1_h{ordem_harmonica}_{eixo}"] = _extrair_soma_amplitudes_na_banda(
                        amplitude_espectral,
                        mascara,
                    )
            features_modelo.append(linha)
    return pd.DataFrame(features_modelo)


def _metric_payload(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    y_bin = label_binarize(y_true, classes=list(sorted(MAPA_CLASSES)))
    try:
        roc_auc_macro = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
    except ValueError:
        roc_auc_macro = float("nan")

    report = classification_report(
        y_true,
        y_pred,
        labels=sorted(MAPA_CLASSES),
        target_names=[MAPA_CLASSES[i] for i in sorted(MAPA_CLASSES)],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_macro_ovr": roc_auc_macro,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=sorted(MAPA_CLASSES)).tolist(),
        "classification_report": report,
    }


def _prediction_catalog_for_split(
    modelo_nome: str,
    split_nome: str,
    indices: list[int],
    features_modelo: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> list[dict[str, Any]]:
    catalogo: list[dict[str, Any]] = []
    for pos, indice_original in enumerate(indices):
        row_meta = features_modelo.loc[indice_original]
        classe_real = int(y_true.iloc[pos])
        classe_predita = int(y_pred[pos])
        probabilidade_predita = float(y_proba[pos, classe_predita])
        catalogo.append(
            {
                "source_model": modelo_nome,
                "split": split_nome,
                "indice_original": int(indice_original),
                "classe_real": classe_real,
                "classe_real_nome": MAPA_CLASSES[classe_real],
                "classe_predita": classe_predita,
                "classe_predita_nome": MAPA_CLASSES[classe_predita],
                "acertou": classe_real == classe_predita,
                "probabilidade_predita": probabilidade_predita,
                "segmento_id": int(row_meta["segmento_id"]),
                "tempo_inicial_s": float(row_meta["tempo_inicial_s"]),
                "tempo_final_s": float(row_meta["tempo_final_s"]),
            }
        )
    return catalogo


def _save_model_artifacts(
    repo_root: Path,
    analysis_id: str,
    params: AnalysisParams,
    features_modelo: pd.DataFrame,
    random_forest: RandomForestClassifier,
    xgboost_model: XGBClassifier,
    svm_model: Pipeline,
) -> Path:
    analysis_hash = hashlib.sha1(analysis_id.encode("utf-8")).hexdigest()[:12]
    dataset_slug = params.dataset_key.replace("/", "_").replace("\\", "_")
    safe_id = f"{dataset_slug}__{analysis_hash}"
    artifact_dir = repo_root / "outputs" / "model_artifacts" / safe_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dump(random_forest, artifact_dir / "random_forest.joblib")
    dump(xgboost_model, artifact_dir / "xgboost.joblib")
    dump(svm_model, artifact_dir / "svm.joblib")

    metadata = {
        "analysis_id": analysis_id,
        "analysis_parameters": asdict(params),
        "n_samples": int(features_modelo.shape[0]),
        "n_features": int(features_modelo.shape[1]),
        "feature_columns": features_modelo.columns.tolist(),
    }
    pd.Series(metadata).to_json(artifact_dir / "metadata.json", force_ascii=False, indent=2)
    return artifact_dir


def _organizar_shap_multiclasse(shap_values: Any, n_classes: int) -> np.ndarray:
    if hasattr(shap_values, "values"):
        shap_array = shap_values.values
    else:
        shap_array = shap_values
    if isinstance(shap_array, list):
        return np.stack(shap_array, axis=0)
    shap_array = np.asarray(shap_array)
    if shap_array.ndim == 3 and shap_array.shape[2] == n_classes:
        return np.moveaxis(shap_array, 2, 0)
    if shap_array.ndim == 3 and shap_array.shape[0] == n_classes:
        return shap_array
    raise ValueError(f"Formato de SHAP nao suportado: {shap_array.shape}")


def _amostras_explicacao(features_modelo: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    grupo = features_modelo.groupby("classe", group_keys=False)
    amostras = grupo.sample(n=min(params.explanation_samples_per_class, grupo.size().min()), random_state=params.random_state)
    amostras = amostras.reset_index().rename(columns={"index": "indice_original"})
    amostras = amostras.sort_values(["classe", "segmento_id"]).reset_index(drop=True)
    return amostras[["indice_original", "classe", "classe_nome", "segmento_id", "tempo_inicial_s", "tempo_final_s"]]


def run_analysis(repo_root: Path, params: AnalysisParams, analysis_id: str) -> tuple[AnalysisArtifacts, dict[str, Any]]:
    arrays = _load_dataset(repo_root, params.dataset_key)
    labels = arrays["gt"]

    duracao_linha_s = params.pontos_por_linha / params.fs
    linhas_por_intervalo = int(params.duracao_intervalo_s / duracao_linha_s)
    amostras_por_intervalo = linhas_por_intervalo * params.pontos_por_linha

    segmented_rows = _segment_by_class(arrays, labels)
    segmentos_por_eixo_classe = _reshape_segments(segmented_rows, amostras_por_intervalo)
    features_modelo = _build_features(segmentos_por_eixo_classe, params)

    colunas_identificacao = ["classe", "classe_nome", "segmento_id", "tempo_inicial_s", "tempo_final_s"]
    colunas_features = [col for col in features_modelo.columns if col not in colunas_identificacao]

    X_features = features_modelo[colunas_features].copy()
    y_features = features_modelo["classe"].copy()
    amostras_explicacao_info = _amostras_explicacao(features_modelo, params)

    indices_explicacao = amostras_explicacao_info["indice_original"].tolist()
    X_explicacao = X_features.loc[indices_explicacao].copy().reset_index(drop=True)
    y_explicacao = y_features.loc[indices_explicacao].copy().reset_index(drop=True)
    X_restante = X_features.drop(index=indices_explicacao)
    y_restante = y_features.drop(index=indices_explicacao)

    X_train, X_test, y_train, y_test = train_test_split(
        X_restante,
        y_restante,
        test_size=params.test_size,
        random_state=params.random_state,
        stratify=y_restante,
    )
    train_indices = X_train.index.to_list()
    test_indices = X_test.index.to_list()

    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=params.random_state,
        n_jobs=1,
    )
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    y_proba_rf = random_forest.predict_proba(X_test)

    xgboost_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(MAPA_CLASSES),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=params.random_state,
        n_jobs=1,
    )
    xgboost_model.fit(X_train, y_train)
    y_pred_xgb = xgboost_model.predict(X_test)
    y_proba_xgb = xgboost_model.predict_proba(X_test)

    svm_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=10, kernel="rbf", gamma="scale", probability=True, random_state=params.random_state)),
        ]
    )
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    y_proba_svm = svm_model.predict_proba(X_test)

    y_pred_rf_train = random_forest.predict(X_train)
    y_proba_rf_train = random_forest.predict_proba(X_train)
    y_pred_xgb_train = xgboost_model.predict(X_train)
    y_proba_xgb_train = xgboost_model.predict_proba(X_train)
    y_pred_svm_train = svm_model.predict(X_train)
    y_proba_svm_train = svm_model.predict_proba(X_train)

    prediction_catalog = {
        "RandomForest": _prediction_catalog_for_split(
            "RandomForest",
            "teste",
            test_indices,
            features_modelo,
            y_test.reset_index(drop=True),
            y_pred_rf,
            y_proba_rf,
        )
        + _prediction_catalog_for_split(
            "RandomForest",
            "treino",
            train_indices,
            features_modelo,
            y_train.reset_index(drop=True),
            y_pred_rf_train,
            y_proba_rf_train,
        ),
        "XGBoost": _prediction_catalog_for_split(
            "XGBoost",
            "teste",
            test_indices,
            features_modelo,
            y_test.reset_index(drop=True),
            y_pred_xgb,
            y_proba_xgb,
        )
        + _prediction_catalog_for_split(
            "XGBoost",
            "treino",
            train_indices,
            features_modelo,
            y_train.reset_index(drop=True),
            y_pred_xgb_train,
            y_proba_xgb_train,
        ),
        "SVM": _prediction_catalog_for_split(
            "SVM",
            "teste",
            test_indices,
            features_modelo,
            y_test.reset_index(drop=True),
            y_pred_svm,
            y_proba_svm,
        )
        + _prediction_catalog_for_split(
            "SVM",
            "treino",
            train_indices,
            features_modelo,
            y_train.reset_index(drop=True),
            y_pred_svm_train,
            y_proba_svm_train,
        ),
    }

    artifact_dir = _save_model_artifacts(
        repo_root,
        analysis_id,
        params,
        features_modelo,
        random_forest,
        xgboost_model,
        svm_model,
    )

    artifacts = AnalysisArtifacts(
        params=params,
        features_modelo=features_modelo,
        segmentos_por_eixo_classe=segmentos_por_eixo_classe,
        X_features=X_features,
        y_features=y_features,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_indices=train_indices,
        test_indices=test_indices,
        X_explicacao=X_explicacao,
        y_explicacao=y_explicacao,
        amostras_explicacao_info=amostras_explicacao_info,
        random_forest=random_forest,
        xgboost_model=xgboost_model,
        svm_model=svm_model,
        y_pred_rf=y_pred_rf,
        y_pred_xgb=y_pred_xgb,
        y_pred_svm=y_pred_svm,
        y_proba_rf=y_proba_rf,
        y_proba_xgb=y_proba_xgb,
        y_proba_svm=y_proba_svm,
        prediction_catalog=prediction_catalog,
        artifact_dir=artifact_dir,
    )

    resumo = {
        "dataset_key": params.dataset_key,
        "dataset_shape": {
            "linhas": int(labels.shape[0]),
            "pontos_por_linha": int(params.pontos_por_linha),
            "classes": len(MAPA_CLASSES),
        },
        "segmentacao": {
            "duracao_intervalo_s": params.duracao_intervalo_s,
            "linhas_por_intervalo": linhas_por_intervalo,
            "amostras_por_intervalo": amostras_por_intervalo,
        },
        "feature_space": {
            "n_features": int(X_features.shape[1]),
            "feature_names": X_features.columns.tolist(),
        },
        "artifact": {
            "directory": str(artifact_dir),
            "files": [
                "random_forest.joblib",
                "xgboost.joblib",
                "svm.joblib",
                "metadata.json",
            ],
        },
        "metricas_modelos": [
            {
                "modelo": "RandomForest",
                "treino": _metric_payload(y_train, y_pred_rf_train, y_proba_rf_train),
                "teste": _metric_payload(y_test, y_pred_rf, y_proba_rf),
            },
            {
                "modelo": "XGBoost",
                "treino": _metric_payload(y_train, y_pred_xgb_train, y_proba_xgb_train),
                "teste": _metric_payload(y_test, y_pred_xgb, y_proba_xgb),
            },
            {
                "modelo": "SVM",
                "treino": _metric_payload(y_train, y_pred_svm_train, y_proba_svm_train),
                "teste": _metric_payload(y_test, y_pred_svm, y_proba_svm),
            },
        ],
        "prediction_catalog": prediction_catalog,
        "explanation_pool": amostras_explicacao_info.to_dict(orient="records"),
        "analysis_parameters": asdict(params),
    }
    return artifacts, resumo


def sample_fft_payload(
    artifacts: AnalysisArtifacts,
    sample_index: int,
    frequencia_min_hz: float = 0.0,
    frequencia_max_hz: float | None = None,
) -> dict[str, Any]:
    if sample_index not in artifacts.features_modelo.index:
        raise KeyError(f"Amostra '{sample_index}' nao encontrada.")

    row_meta = artifacts.features_modelo.loc[sample_index]
    classe = int(row_meta["classe"])
    segmento_id = int(row_meta["segmento_id"])
    fs = artifacts.params.fs

    if frequencia_max_hz is None:
        frequencia_max_hz = fs / 2.0

    payload_axes: dict[str, Any] = {}
    for eixo in AXES:
        sinal_segmento = artifacts.segmentos_por_eixo_classe[eixo][classe][segmento_id]
        janela_hann = np.hanning(len(sinal_segmento))
        amplitude_espectral = _calcular_espectro_amplitude(sinal_segmento, janela_hann)
        frequencias_fft = np.fft.rfftfreq(len(sinal_segmento), d=1 / fs)
        mascara = (frequencias_fft >= frequencia_min_hz) & (frequencias_fft <= frequencia_max_hz)
        payload_axes[eixo] = {
            "frequencias_hz": frequencias_fft[mascara].tolist(),
            "amplitudes": amplitude_espectral[mascara].tolist(),
        }

    return {
        "sample_index": int(sample_index),
        "classe_real": classe,
        "classe_real_nome": MAPA_CLASSES[classe],
        "segmento_id": segmento_id,
        "tempo_inicial_s": float(row_meta["tempo_inicial_s"]),
        "tempo_final_s": float(row_meta["tempo_final_s"]),
        "fft_axes": payload_axes,
    }


def explain_sample(
    artifacts: AnalysisArtifacts,
    sample_index: int,
    shap_source_model: str,
    top_k: int,
) -> dict[str, Any]:
    if sample_index not in artifacts.features_modelo.index:
        raise KeyError(f"Amostra '{sample_index}' nao encontrada.")

    row_meta = artifacts.features_modelo.loc[sample_index]
    X_row = artifacts.X_features.loc[[sample_index]].copy()
    y_real = int(artifacts.y_features.loc[sample_index])

    if shap_source_model == "RandomForest":
        modelo = artifacts.random_forest
        nome_modelo = "RandomForest"
    elif shap_source_model == "XGBoost":
        modelo = artifacts.xgboost_model
        nome_modelo = "XGBoost"
    else:
        raise ValueError("shap_source_model deve ser 'RandomForest' ou 'XGBoost'.")

    explainer = shap.TreeExplainer(modelo)
    shap_raw = explainer.shap_values(X_row)
    shap_classes = _organizar_shap_multiclasse(shap_raw, n_classes=len(MAPA_CLASSES))
    pred = modelo.predict(X_row)
    proba = modelo.predict_proba(X_row)
    classe_predita = int(pred[0])
    probabilidade_predita = float(proba[0, classe_predita])
    valores_shap = shap_classes[classe_predita, 0, :]
    ordem = np.argsort(np.abs(valores_shap))[::-1][:top_k]

    top_features = []
    for posicao, indice_feature in enumerate(ordem, start=1):
        top_features.append(
            {
                "rank": posicao,
                "feature": X_row.columns[indice_feature],
                "valor_feature": float(X_row.iloc[0, indice_feature]),
                "shap_value": float(valores_shap[indice_feature]),
                "impacto_absoluto": float(abs(valores_shap[indice_feature])),
            }
        )

    return {
        "sample_index": int(sample_index),
        "sample_metadata": {
            "classe_real": y_real,
            "classe_real_nome": MAPA_CLASSES[y_real],
            "classe_predita": classe_predita,
            "classe_predita_nome": MAPA_CLASSES[classe_predita],
            "probabilidade_predita": probabilidade_predita,
            "segmento_id": int(row_meta["segmento_id"]),
            "tempo_inicial_s": float(row_meta["tempo_inicial_s"]),
            "tempo_final_s": float(row_meta["tempo_final_s"]),
        },
        "source_model": nome_modelo,
        "top_features": top_features,
    }
