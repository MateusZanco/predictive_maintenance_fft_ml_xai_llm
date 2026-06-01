from __future__ import annotations

from pathlib import Path

import numpy as np

from .fft_service import DURACAO_LINHA_S, LINHAS_POR_JANELA, calculate_kinematic_params
from .schemas import SampleMetadata, SampleSummary


class SampleRepository:
    def __init__(self, sample_dir: Path):
        self.sample_dir = sample_dir

    def list_sample_paths(self) -> list[Path]:
        if not self.sample_dir.exists():
            return []
        return sorted(self.sample_dir.glob("rockpi_raw_*_cont*.*")) + sorted(
            p for p in self.sample_dir.glob("rockpi_raw_*.npz") if "cont" not in p.name
        )

    def list_samples(self) -> list[SampleSummary]:
        summaries = []
        seen = set()
        for path in sorted(self.sample_dir.glob("rockpi_raw_*.npz")):
            if path.name in seen:
                continue
            seen.add(path.name)
            summaries.append(self._build_summary(path))
        return summaries

    def _read(self, sample_id: str) -> np.lib.npyio.NpzFile:
        path = self.sample_dir / sample_id
        if not path.exists():
            raise FileNotFoundError(sample_id)
        return np.load(path, allow_pickle=True)

    def _decode_scalar(self, value):
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        return value

    def _build_summary(self, path: Path) -> SampleSummary:
        with np.load(path, allow_pickle=True) as sample_npz:
            x_rows = sample_npz["x_rows"]
            duration_s = float(self._decode_scalar(sample_npz["duracao_total_s"]))
            available_windows = int(x_rows.shape[0] // LINHAS_POR_JANELA)
            classe = None
            classe_nome = None
            if "classe" in sample_npz.files:
                classe = int(self._decode_scalar(sample_npz["classe"]))
            if "classe_nome" in sample_npz.files:
                classe_nome = str(self._decode_scalar(sample_npz["classe_nome"]))
            return SampleSummary(
                sample_id=path.name,
                dataset_operacao=str(self._decode_scalar(sample_npz["dataset_operacao"])),
                condicao_operacao=str(self._decode_scalar(sample_npz["condicao_operacao"])),
                rpm=float(self._decode_scalar(sample_npz["rpm"])),
                torque_nm=float(self._decode_scalar(sample_npz["torque_nm"])),
                duration_s=duration_s,
                available_windows=available_windows,
                classe=classe,
                classe_nome=classe_nome,
            )

    def get_metadata(self, sample_id: str) -> SampleMetadata:
        path = self.sample_dir / sample_id
        if not path.exists():
            raise FileNotFoundError(sample_id)

        with np.load(path, allow_pickle=True) as sample_npz:
            summary = self._build_summary(path)
            x_rows = sample_npz["x_rows"]
            points_per_row = int(self._decode_scalar(sample_npz["pontos_por_linha"]))
            fs = int(self._decode_scalar(sample_npz["fs"]))
            rpm = float(self._decode_scalar(sample_npz["rpm"]))
            params = calculate_kinematic_params(rpm)
            return SampleMetadata(
                **summary.model_dump(),
                fs=fs,
                points_per_row=points_per_row,
                row_duration_s=DURACAO_LINHA_S,
                total_rows=int(x_rows.shape[0]),
                total_flat_samples_per_axis=int(x_rows.size),
                available_axes=["x", "y", "z"],
                fm1=params["fm1"],
                fm2=params["fm2"],
            )

    def open_sample(self, sample_id: str) -> np.lib.npyio.NpzFile:
        return self._read(sample_id)
