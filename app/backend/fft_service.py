from __future__ import annotations

from pathlib import Path

import numpy as np


FS = 10_000
PONTOS_POR_LINHA = 200
DURACAO_LINHA_S = PONTOS_POR_LINHA / FS
DURACAO_JANELA_S = 1.0
LINHAS_POR_JANELA = int(DURACAO_JANELA_S / DURACAO_LINHA_S)
AMOSTRAS_POR_JANELA = LINHAS_POR_JANELA * PONTOS_POR_LINHA

ZR1 = 100.0
ZS1 = 20.0
ZR2 = 100.0
ZS2 = 28.0
REDUCAO_PRIMEIRO_ESTAGIO = 6.0


def calculate_kinematic_params(rpm: float) -> dict[str, float]:
    fsh1 = rpm / 60.0
    fm1 = ((ZR1 * ZS1) / (ZR1 + ZS1)) * fsh1

    fsh2 = (rpm / REDUCAO_PRIMEIRO_ESTAGIO) / 60.0
    fm2 = ((ZR2 * ZS2) / (ZR2 + ZS2)) * fsh2

    return {
        "fm1": float(fm1),
        "fm2": float(fm2),
    }


def extract_window_signal(sample_npz: np.lib.npyio.NpzFile, axis: str, window_index: int) -> tuple[np.ndarray, float, float]:
    rows_key = f"{axis}_rows"
    if rows_key not in sample_npz:
        raise KeyError(f"Axis rows not found in sample: {rows_key}")

    rows = sample_npz[rows_key]
    total_windows = rows.shape[0] // LINHAS_POR_JANELA
    if window_index >= total_windows:
        raise IndexError(f"window_index {window_index} out of range for {total_windows} windows")

    row_start = window_index * LINHAS_POR_JANELA
    row_end = row_start + LINHAS_POR_JANELA
    window_rows = rows[row_start:row_end]
    signal = window_rows.reshape(-1).astype(float)
    start_s = row_start * DURACAO_LINHA_S
    end_s = row_end * DURACAO_LINHA_S
    return signal, float(start_s), float(end_s)


def compute_fft(signal: np.ndarray, fmin: float, fmax: float, apply_hann: bool) -> tuple[np.ndarray, np.ndarray]:
    centered = signal - np.mean(signal)
    if apply_hann:
        window = np.hanning(signal.shape[0])
        processed = centered * window
        norm = window.sum()
    else:
        processed = centered
        norm = signal.shape[0]

    spectrum = np.fft.rfft(processed)
    amplitude = np.abs(spectrum) * 2.0 / norm
    amplitude[0] = 0.0
    freq = np.fft.rfftfreq(signal.shape[0], d=1 / FS)

    mask = (freq >= fmin) & (freq <= fmax)
    return freq[mask], amplitude[mask]


def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_kurtosis(signal: np.ndarray) -> float:
    centered = signal - np.mean(signal)
    std = np.std(centered)
    if std == 0:
        return 0.0
    fourth_moment = np.mean(np.power(centered, 4))
    return float(fourth_moment / (std ** 4))


def compute_peak_value(signal: np.ndarray) -> float:
    return float(np.max(np.abs(signal)))
