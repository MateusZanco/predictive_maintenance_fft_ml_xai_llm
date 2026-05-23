from __future__ import annotations

import json
from pathlib import Path


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


NOTEBOOK_CELLS = [
    md_cell(
        """# Treinamento Simplificado do XGBoost para a Rock Pi

Notebook enxuto para:
- reservar janelas de teste da Rock Pi por classe e condicao operacional;
- garantir que essas janelas nao participem do treino;
- extrair 4 features no tempo e bandas harmonicas de `Fm1` e `Fm2`;
- avaliar o `XGBoost` com holdout por blocos e early stopping;
- treinar o modelo final com todas as amostras disponiveis fora da reserva da Rock Pi;
- exportar os artefatos necessarios para a aplicacao embarcada.
"""
    ),
    md_cell("## 1. Configuracao do Ambiente"),
    code_cell(
        """from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ROCKPI_SAMPLES_DIR = OUTPUTS_DIR / "rockpi_test_samples"
MODEL_ARTIFACTS_DIR = OUTPUTS_DIR / "model_artifacts_rockpi_simplificado"

ROCKPI_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
np.random.seed(42)
"""
    ),
    md_cell("## 2. Definicao dos Datasets e Parametros"),
    code_cell(
        """mapa_classes = {
    0: "Normal",
    1: "Desgaste Superficial",
    2: "Dente Trincado",
    3: "Dente Lascado",
    4: "Dente Ausente",
}

config_datasets = {
    "1500_10": {
        "rpm": 1500.0,
        "torque_nm": 10.0,
        "condicao_operacao": "1500 rpm / 10 Nm",
    },
    "2700_25": {
        "rpm": 2700.0,
        "torque_nm": 25.0,
        "condicao_operacao": "2700 rpm / 25 Nm",
    },
}

fs = 10_000
pontos_por_linha = 200
duracao_linha_s = pontos_por_linha / fs
duracao_janela_s = 1.0
linhas_por_janela = int(duracao_janela_s / duracao_linha_s)
amostras_por_janela = linhas_por_janela * pontos_por_linha

duracao_reserva_rockpi_s = 5.0
linhas_reserva_rockpi = int(duracao_reserva_rockpi_s / duracao_linha_s)
janelas_por_bloco = 5
test_size_blocos = 0.3
val_size_blocos = 0.2

zr1 = 100
zs1 = 20
zr2 = 100
zs2 = 28
reducao_primeiro_estagio = 6.0
ordens_harmonicas = [1, 2, 3, 4, 5]
largura_banda_harmonica_hz = 10.0

assert linhas_reserva_rockpi % linhas_por_janela == 0

print(f"Fs = {fs} Hz")
print(f"Linhas por janela de 1 s = {linhas_por_janela}")
print(f"Linhas reservadas para a Rock Pi por classe = {linhas_reserva_rockpi}")
print(f"Janelas reservadas por classe = {linhas_reserva_rockpi // linhas_por_janela}")
"""
    ),
    md_cell("## 3. Carregamento dos Dados Brutos"),
    code_cell(
        """datasets_brutos = {}
resumo_datasets = []

for nome_dataset, meta in config_datasets.items():
    x = np.load(DATA_DIR / f"x_{nome_dataset}.npy")
    y = np.load(DATA_DIR / f"y_{nome_dataset}.npy")
    z = np.load(DATA_DIR / f"z_{nome_dataset}.npy")
    gt = np.load(DATA_DIR / f"gt_{nome_dataset}.npy")

    datasets_brutos[nome_dataset] = {
        "x": x,
        "y": y,
        "z": z,
        "gt": gt,
        **meta,
    }

    resumo_datasets.append(
        {
            "dataset_operacao": nome_dataset,
            "condicao_operacao": meta["condicao_operacao"],
            "rpm": meta["rpm"],
            "torque_nm": meta["torque_nm"],
            "linhas": int(gt.shape[0]),
            "classes_unicas": sorted(np.unique(gt).tolist()),
        }
    )

display(pd.DataFrame(resumo_datasets))
"""
    ),
    md_cell("## 4. Reserva e Exportacao das Amostras de Teste da Rock Pi"),
    code_cell(
        """def reservar_e_exportar_amostras_rockpi(datasets_info):
    datasets_para_treino = {}
    registros_exportacao = []

    for nome_dataset, info in datasets_info.items():
        gt = info["gt"]
        indices_treino_por_classe = {}

        for classe in sorted(mapa_classes):
            indices_classe = np.flatnonzero(gt == classe)
            if indices_classe.shape[0] < linhas_reserva_rockpi:
                raise ValueError(
                    f"{nome_dataset} nao possui linhas suficientes para reservar {duracao_reserva_rockpi_s:.1f} s da classe {classe}."
                )

            indices_reservados = indices_classe[:linhas_reserva_rockpi]
            indices_treino = indices_classe[linhas_reserva_rockpi:]
            indices_treino_por_classe[classe] = indices_treino

            x_rows = info["x"][indices_reservados]
            y_rows = info["y"][indices_reservados]
            z_rows = info["z"][indices_reservados]
            gt_rows = gt[indices_reservados]

            caminho_npz = ROCKPI_SAMPLES_DIR / f"rockpi_raw_{nome_dataset}_classe_{classe}_{int(duracao_reserva_rockpi_s)}s.npz"

            np.savez(
                caminho_npz,
                dataset_operacao=nome_dataset,
                condicao_operacao=info["condicao_operacao"],
                rpm=info["rpm"],
                torque_nm=info["torque_nm"],
                classe=classe,
                classe_nome=mapa_classes[classe],
                fs=fs,
                pontos_por_linha=pontos_por_linha,
                duracao_linha_s=duracao_linha_s,
                duracao_total_s=duracao_reserva_rockpi_s,
                linha_inicial_classe_reconstruida=0,
                linha_final_classe_reconstruida=linhas_reserva_rockpi,
                indices_linhas_origem=indices_reservados,
                x_rows=x_rows,
                y_rows=y_rows,
                z_rows=z_rows,
                gt_rows=gt_rows,
                x_flat=x_rows.reshape(-1),
                y_flat=y_rows.reshape(-1),
                z_flat=z_rows.reshape(-1),
            )

            registros_exportacao.append(
                {
                    "dataset_operacao": nome_dataset,
                    "condicao_operacao": info["condicao_operacao"],
                    "classe": classe,
                    "classe_nome": mapa_classes[classe],
                    "linhas_reservadas": int(indices_reservados.shape[0]),
                    "janelas_reservadas": int(indices_reservados.shape[0] // linhas_por_janela),
                    "arquivo": caminho_npz.name,
                }
            )

        datasets_para_treino[nome_dataset] = {
            **info,
            "indices_treino_por_classe": indices_treino_por_classe,
        }

    return datasets_para_treino, pd.DataFrame(registros_exportacao)


datasets_para_treino, resumo_exportacao_rockpi = reservar_e_exportar_amostras_rockpi(datasets_brutos)
display(resumo_exportacao_rockpi)
"""
    ),
    md_cell("## 5. Segmentacao das Amostras de Treino"),
    code_cell(
        """def segmentar_dataset_para_treino(info):
    segmentos_por_eixo_classe = {eixo: {} for eixo in ["x", "y", "z"]}

    for classe in sorted(mapa_classes):
        indices_treino = info["indices_treino_por_classe"][classe]

        for eixo in ["x", "y", "z"]:
            matriz_classe = info[eixo][indices_treino]
            if matriz_classe.shape[0] % linhas_por_janela != 0:
                raise ValueError(
                    f"O total de linhas da classe {classe} no eixo {eixo} nao e multiplo de {linhas_por_janela}."
                )
            segmentos_por_eixo_classe[eixo][classe] = matriz_classe.reshape(-1, amostras_por_janela)

    return segmentos_por_eixo_classe


datasets_segmentados = {}
resumo_segmentacao = []

for nome_dataset, info in datasets_para_treino.items():
    segmentos_por_eixo_classe = segmentar_dataset_para_treino(info)
    datasets_segmentados[nome_dataset] = {
        **info,
        "segmentos_por_eixo_classe": segmentos_por_eixo_classe,
    }

    for classe in sorted(mapa_classes):
        resumo_segmentacao.append(
            {
                "dataset_operacao": nome_dataset,
                "condicao_operacao": info["condicao_operacao"],
                "classe": classe,
                "classe_nome": mapa_classes[classe],
                "segmentos_por_classe": int(segmentos_por_eixo_classe["x"][classe].shape[0]),
            }
        )

display(pd.DataFrame(resumo_segmentacao))
"""
    ),
    md_cell("## 6. Extracao das Features no Tempo e Harmonicas de Fm1/Fm2"),
    code_cell(
        """frequencias_fft = np.fft.rfftfreq(amostras_por_janela, d=1 / fs)
janela_hann = np.hanning(amostras_por_janela)


def calcular_rms(sinal_segmento):
    return float(np.sqrt(np.mean(np.square(sinal_segmento))))


def calcular_kurtosis(sinal_segmento):
    sinal_centrado = sinal_segmento - np.mean(sinal_segmento)
    desvio_padrao = np.std(sinal_centrado)
    if desvio_padrao == 0:
        return 0.0
    quarto_momento = np.mean(np.power(sinal_centrado, 4))
    return float(quarto_momento / (desvio_padrao ** 4))


def calcular_peak_value(sinal_segmento):
    return float(np.max(np.abs(sinal_segmento)))


def calcular_crest_factor(sinal_segmento):
    rms = calcular_rms(sinal_segmento)
    pico = calcular_peak_value(sinal_segmento)
    return float(pico / (rms + 1e-12))


def calcular_frequencias_engrenamento(rpm_entrada):
    fsh1 = rpm_entrada / 60.0
    fm1_hz = ((zr1 * zs1) / (zr1 + zs1)) * fsh1
    fsh2 = (rpm_entrada / reducao_primeiro_estagio) / 60.0
    fm2_hz = ((zr2 * zs2) / (zr2 + zs2)) * fsh2
    return float(fm1_hz), float(fm2_hz)


def calcular_espectro_amplitude(sinal_segmento):
    sinal_centrado = sinal_segmento - np.mean(sinal_segmento)
    espectro = np.fft.rfft(sinal_centrado * janela_hann)
    return (2.0 / amostras_por_janela) * np.abs(espectro)


def extrair_metricas_banda(espectro_amplitude, frequencia_central_hz, largura_hz):
    mascara = (
        (frequencias_fft >= (frequencia_central_hz - largura_hz))
        & (frequencias_fft <= (frequencia_central_hz + largura_hz))
    )
    amplitudes_banda = espectro_amplitude[mascara]
    energia_banda = float(np.sum(np.square(amplitudes_banda)))
    amplitude_maxima = float(np.max(amplitudes_banda))
    return energia_banda, amplitude_maxima


def extrair_features_harmonicas(datasets_info):
    linhas_features = []

    for nome_dataset, info in datasets_info.items():
        segmentos_por_eixo_classe = info["segmentos_por_eixo_classe"]

        for classe in sorted(mapa_classes):
            quantidade_segmentos = segmentos_por_eixo_classe["x"][classe].shape[0]

            for segmento_id in range(quantidade_segmentos):
                bloco_id = segmento_id // janelas_por_bloco
                fm1_hz, fm2_hz = calcular_frequencias_engrenamento(info["rpm"])
                linha = {
                    "dataset_operacao": nome_dataset,
                    "condicao_operacao": info["condicao_operacao"],
                    "rpm": info["rpm"],
                    "torque_nm": info["torque_nm"],
                    "classe": classe,
                    "classe_nome": mapa_classes[classe],
                    "segmento_id": segmento_id,
                    "bloco_id": bloco_id,
                    "grupo_bloco": f"{nome_dataset}_classe_{classe}_bloco_{bloco_id}",
                    "estrato": f"{nome_dataset}_classe_{classe}",
                    "tempo_inicial_s_reconstruido": segmento_id * duracao_janela_s,
                    "tempo_final_s_reconstruido": (segmento_id + 1) * duracao_janela_s,
                    "fm1_hz": fm1_hz,
                    "fm2_hz": fm2_hz,
                }

                for eixo in ["x", "y", "z"]:
                    sinal_segmento = segmentos_por_eixo_classe[eixo][classe][segmento_id]
                    linha[f"rms_{eixo}"] = calcular_rms(sinal_segmento)
                    linha[f"kurtosis_{eixo}"] = calcular_kurtosis(sinal_segmento)
                    linha[f"peak_value_{eixo}"] = calcular_peak_value(sinal_segmento)
                    linha[f"crest_factor_{eixo}"] = calcular_crest_factor(sinal_segmento)

                    espectro_amplitude = calcular_espectro_amplitude(sinal_segmento)
                    for nome_estagio, frequencia_base in [("fm1", fm1_hz), ("fm2", fm2_hz)]:
                        for ordem_harmonica in ordens_harmonicas:
                            frequencia_harmonica = ordem_harmonica * frequencia_base
                            energia_banda, amplitude_maxima = extrair_metricas_banda(
                                espectro_amplitude,
                                frequencia_harmonica,
                                largura_banda_harmonica_hz,
                            )
                            linha[f"energy_{nome_estagio}_h{ordem_harmonica}_{eixo}"] = energia_banda
                            linha[f"amp_max_{nome_estagio}_h{ordem_harmonica}_{eixo}"] = amplitude_maxima

                linhas_features.append(linha)

    return pd.DataFrame(linhas_features)


features_modelo = extrair_features_harmonicas(datasets_segmentados)

colunas_identificacao = [
    "dataset_operacao",
    "condicao_operacao",
    "rpm",
    "torque_nm",
    "classe",
    "classe_nome",
    "segmento_id",
    "bloco_id",
    "grupo_bloco",
    "estrato",
    "tempo_inicial_s_reconstruido",
    "tempo_final_s_reconstruido",
    "fm1_hz",
    "fm2_hz",
]

colunas_features = [coluna for coluna in features_modelo.columns if coluna not in colunas_identificacao]
X_features = features_modelo[colunas_features].copy()
y_features = features_modelo["classe"].copy()

print(f"Total de amostras para modelagem: {X_features.shape[0]}")
print(f"Total de features: {X_features.shape[1]}")
display(features_modelo.head())
"""
    ),
    md_cell("## 7. Holdout por Blocos para Avaliacao"),
    code_cell(
        """def construir_split_por_blocos(features_df, test_size=0.3, random_state=42):
    blocos_treino = []
    blocos_teste = []

    for estrato, grupo_df in features_df.groupby("estrato"):
        blocos = np.array(sorted(grupo_df["grupo_bloco"].unique().tolist()))
        rng = np.random.RandomState(random_state)
        blocos = blocos.copy()
        rng.shuffle(blocos)

        n_teste = max(1, int(np.ceil(blocos.shape[0] * test_size)))
        blocos_teste_estrato = blocos[:n_teste]
        blocos_treino_estrato = blocos[n_teste:]

        blocos_teste.extend(blocos_teste_estrato.tolist())
        blocos_treino.extend(blocos_treino_estrato.tolist())

    mask_treino = features_df["grupo_bloco"].isin(blocos_treino)
    mask_teste = features_df["grupo_bloco"].isin(blocos_teste)

    indices_treino = np.sort(features_df.index[mask_treino].to_numpy())
    indices_teste = np.sort(features_df.index[mask_teste].to_numpy())

    return indices_treino, indices_teste


indices_treino, indices_teste = construir_split_por_blocos(
    features_modelo,
    test_size=test_size_blocos,
    random_state=42,
)

X_train = X_features.loc[indices_treino].copy()
X_test = X_features.loc[indices_teste].copy()
y_train = y_features.loc[indices_treino].copy()
y_test = y_features.loc[indices_teste].copy()

metadados_train = features_modelo.loc[indices_treino, colunas_identificacao].reset_index(drop=True)
metadados_test = features_modelo.loc[indices_teste, colunas_identificacao].reset_index(drop=True)

print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")
"""
    ),
    md_cell("## 8. Separacao de Validacao para Early Stopping"),
    code_cell(
"""features_train_holdout = features_modelo.loc[indices_treino].copy()

indices_fit, indices_val = construir_split_por_blocos(
    features_train_holdout,
    test_size=val_size_blocos,
    random_state=123,
)

X_fit = X_features.loc[indices_fit].copy()
X_val = X_features.loc[indices_val].copy()
y_fit = y_features.loc[indices_fit].copy()
y_val = y_features.loc[indices_val].copy()

print(f"Fit: {X_fit.shape[0]} amostras")
print(f"Validacao: {X_val.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")
"""
    ),
    md_cell("## 9. Treinamento e Avaliacao do XGBoost"),
    code_cell(
        """modelo_avaliacao = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=1.0,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=len(mapa_classes),
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
    early_stopping_rounds=50,
)

modelo_avaliacao.fit(
    X_fit,
    y_fit,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

best_iteration_raw = modelo_avaliacao.best_iteration
best_iteration = int(best_iteration_raw if best_iteration_raw is not None else modelo_avaliacao.n_estimators - 1)
best_n_estimators = best_iteration + 1

y_pred_train = modelo_avaliacao.predict(X_train)
y_pred_test = modelo_avaliacao.predict(X_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
macro_f1_train = f1_score(y_train, y_pred_train, average="macro")
macro_f1_test = f1_score(y_test, y_pred_test, average="macro")
balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)

print(f"Accuracy treino: {accuracy_train:.4f}")
print(f"Accuracy teste: {accuracy_test:.4f}")
print(f"Gap accuracy: {accuracy_train - accuracy_test:.4f}")
print(f"Macro F1 treino: {macro_f1_train:.4f}")
print(f"Macro F1 teste: {macro_f1_test:.4f}")
print(f"Gap Macro F1: {macro_f1_train - macro_f1_test:.4f}")
print(f"Balanced accuracy teste: {balanced_accuracy_test:.4f}")
print(f"Best iteration: {best_iteration}")
print(f"Best n_estimators: {best_n_estimators}")

print("\\nClassification report - teste:\\n")
print(classification_report(y_test, y_pred_test, target_names=[mapa_classes[i] for i in sorted(mapa_classes)], zero_division=0))

matriz_confusao = confusion_matrix(y_test, y_pred_test, labels=sorted(mapa_classes))
plt.figure(figsize=(8, 6))
sns.heatmap(
    matriz_confusao,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[mapa_classes[i] for i in sorted(mapa_classes)],
    yticklabels=[mapa_classes[i] for i in sorted(mapa_classes)],
)
plt.title("Matriz de Confusao - XGBoost Rock Pi")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.show()

resumo_split = pd.concat(
    [
        metadados_train.assign(split="Treino"),
        metadados_test.assign(split="Teste"),
    ],
    ignore_index=True,
)

resumo_split = (
    resumo_split.groupby(["split", "dataset_operacao", "classe"])
    .size()
    .reset_index(name="amostras")
    .sort_values(["split", "dataset_operacao", "classe"])
    .reset_index(drop=True)
)

display(resumo_split)
"""
    ),
    md_cell("## 10. Treinamento Final para Embarque"),
    code_cell(
        """modelo_final = XGBClassifier(
    n_estimators=best_n_estimators,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=1.0,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=len(mapa_classes),
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
)

modelo_final.fit(X_features, y_features)

print("Modelo final treinado com todas as amostras disponiveis fora da reserva da Rock Pi.")
"""
    ),
    md_cell("## 11. Exportacao dos Artefatos do Modelo"),
    code_cell(
        """caminho_modelo_xgb = MODEL_ARTIFACTS_DIR / "modelo_xgboost_rockpi_simplificado.joblib"
caminho_modelo_generico = MODEL_ARTIFACTS_DIR / "modelo_rockpi_simplificado.joblib"
caminho_colunas = MODEL_ARTIFACTS_DIR / "feature_columns_rockpi_simplificado.json"
caminho_metadata = MODEL_ARTIFACTS_DIR / "model_metadata_rockpi_simplificado.json"
caminho_metricas = MODEL_ARTIFACTS_DIR / "metricas_modelo_rockpi_simplificado.json"
caminho_resumo_split = MODEL_ARTIFACTS_DIR / "resumo_split_modelo_rockpi_simplificado.csv"
caminho_resumo_exportacao = MODEL_ARTIFACTS_DIR / "resumo_amostras_rockpi.csv"

joblib.dump(modelo_final, caminho_modelo_xgb)
joblib.dump(modelo_final, caminho_modelo_generico)
caminho_colunas.write_text(json.dumps(colunas_features, ensure_ascii=False, indent=2), encoding="utf-8")

metadata_modelo = {
    "model_type": "XGBClassifier",
    "artifact_model_path": caminho_modelo_xgb.name,
    "artifact_generic_model_path": caminho_modelo_generico.name,
    "datasets_treinados": list(config_datasets.keys()),
    "reserved_test_seconds_per_class": duracao_reserva_rockpi_s,
    "window_duration_seconds": duracao_janela_s,
    "sample_rate_hz": fs,
    "feature_set": "tempo_fm1_fm2_harmonics_h1_h5_band10",
    "features": colunas_features,
    "evaluation_protocol": "holdout_por_blocos_com_early_stopping",
    "early_stopping_rounds": 50,
    "best_iteration_holdout": int(best_iteration),
    "best_n_estimators_holdout": int(best_n_estimators),
}
caminho_metadata.write_text(json.dumps(metadata_modelo, ensure_ascii=False, indent=2), encoding="utf-8")

metricas_modelo = {
    "accuracy_train_holdout": float(accuracy_train),
    "accuracy_test_holdout": float(accuracy_test),
    "gap_accuracy_holdout": float(accuracy_train - accuracy_test),
    "macro_f1_train_holdout": float(macro_f1_train),
    "macro_f1_test_holdout": float(macro_f1_test),
    "gap_macro_f1_holdout": float(macro_f1_train - macro_f1_test),
    "balanced_accuracy_train_holdout": float(balanced_accuracy_train),
    "balanced_accuracy_test_holdout": float(balanced_accuracy_test),
    "n_fit_holdout": int(X_fit.shape[0]),
    "n_val_holdout": int(X_val.shape[0]),
    "n_train_holdout": int(X_train.shape[0]),
    "n_test_holdout": int(X_test.shape[0]),
    "n_train_final": int(X_features.shape[0]),
    "best_iteration_holdout": int(best_iteration),
    "best_n_estimators_holdout": int(best_n_estimators),
}
caminho_metricas.write_text(json.dumps(metricas_modelo, ensure_ascii=False, indent=2), encoding="utf-8")

resumo_split.to_csv(caminho_resumo_split, index=False)
resumo_exportacao_rockpi.to_csv(caminho_resumo_exportacao, index=False)

print("Artefatos exportados:")
print("-", caminho_modelo_xgb)
print("-", caminho_modelo_generico)
print("-", caminho_colunas)
print("-", caminho_metadata)
print("-", caminho_metricas)
print("-", caminho_resumo_split)
print("-", caminho_resumo_exportacao)
"""
    ),
]


def main() -> None:
    notebook = {
        "cells": NOTEBOOK_CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path = Path("notebooks") / "predictive_maintenance_treinamento_simplificado_rockpi.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook gerado em: {output_path}")


if __name__ == "__main__":
    main()
