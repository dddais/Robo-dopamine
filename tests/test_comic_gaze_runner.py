from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rank_grm_heads_by_comics as comic_rank  # noqa: E402
import run_targetbbox_success_experiments as runner  # noqa: E402


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        python_bin=sys.executable,
        conda_env=None,
        gpus="0,1",
        ranking_method="comic-gaze",
        gaze_heads_root=str(tmp_path / "gaze-heads"),
        comics_root=str(tmp_path / "comics"),
        comic_use_raw=True,
        comic_num_samples=500,
        comic_n_panels=6,
        comic_target_height=256,
        comic_gap=6,
        comic_seed=42,
        comic_skip_early_layers=2,
        comic_max_pixels=None,
        comic_min_pixels=None,
        curve_modes=["forward"],
        top_ks=[8, 64],
        target_label="after_cam_high",
        grounding_box_threshold=0.12,
        swap_bias=6.0,
        wrong_region_samples=1,
        video_num_samples=30,
        source_root=None,
        carrot_source_root=str(tmp_path / "carrot"),
        cube_source_root=str(tmp_path / "cube"),
        bottle_source_root=str(tmp_path / "bottle"),
    )


def test_comic_ranking_uses_raw_diagonal_not_selectivity():
    diagonal = np.array([[0.20, 0.19], [0.02, 0.01]], dtype=np.float64)
    off_diagonal = np.array([[0.19, 0.01], [0.00, 0.00]], dtype=np.float64)

    rows = comic_rank._ranking_rows(diagonal, off_diagonal)

    assert [(row["layer"], row["head"]) for row in rows[:2]] == [(0, 0), (0, 1)]
    assert rows[0]["score"] == 0.20
    assert np.isclose(rows[0]["diagonal_minus_off_diagonal"], 0.01)
    assert np.isclose(rows[1]["diagonal_minus_off_diagonal"], 0.18)


def test_comic_rank_job_is_single_global_raw_corpus_job(tmp_path):
    args = _args(tmp_path)
    out_root = tmp_path / "results"
    jobs = runner.rank_jobs(args, out_root, out_root / "logs")

    assert len(jobs) == 1
    job = jobs[0]
    assert job.name == "rank_comics_last_prompt_comic_gaze"
    assert job.output == runner.comic_gaze_rank_path(out_root)
    assert "rank_grm_heads_by_comics.py" in job.cmd
    assert "--use-raw" in job.cmd
    assert job.cmd[job.cmd.index("--n-samples") + 1] == "500"
    assert job.cmd[job.cmd.index("--seed") + 1] == "42"
    assert "--max-pixels" not in job.cmd
    assert "--min-pixels" not in job.cmd


def test_comic_curve_and_video_jobs_reuse_same_global_ranking(tmp_path, monkeypatch):
    args = _args(tmp_path)
    out_root = tmp_path / "results"
    expected_rank = runner.comic_gaze_rank_path(out_root)

    monkeypatch.setattr(
        runner,
        "sample_path_for",
        lambda root, task, mode: tmp_path / f"{task}_{mode}.json",
    )

    curve_jobs = runner.curve_jobs(args, out_root, out_root / "logs")
    assert len(curve_jobs) == 3 * 3 * 1 * 2
    for job in curve_jobs:
        rank_arg = Path(job.cmd[job.cmd.index("--head-ranking-json") + 1])
        assert rank_arg == expected_rank

    video_jobs = runner.video_jobs(args, out_root, out_root / "logs")
    assert len(video_jobs) == 3
    for job in video_jobs:
        rank_arg = Path(job.cmd[job.cmd.index("--head-ranking-json") + 1])
        assert rank_arg == expected_rank
