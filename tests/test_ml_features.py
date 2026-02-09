import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "cleanshot.py"
SAMPLES = REPO_ROOT / "samples"


def run_cli(*args):
    cmd = ["python", str(CLI), *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def test_outlier_scale_encode_schema(tmp_path):
    output_path = tmp_path / "cleaned.parquet"
    schema_path = tmp_path / "schema.json"
    result = run_cli(
        str(SAMPLES / "dirty.csv"),
        str(output_path),
        "--outlier-mode",
        "percentile",
        "--outlier-pct",
        "0.01,0.99",
        "--scale",
        "standardize",
        "--encode",
        "onehot",
        "--encode-max-categories",
        "10",
        "--schema-out",
        str(schema_path),
        "--preset",
        "ml_ready",
        "--no-dedup",
        "--no-verbose",
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    assert schema_path.exists()
    payload = json.loads(schema_path.read_text())
    report = payload["report"]
    assert report["outlier_bounds"]
    assert "encoded_columns" in report
    assert "scaled_columns" in report


def test_train_val_test_split(tmp_path):
    output_path = tmp_path / "cleaned.parquet"
    result = run_cli(
        str(SAMPLES / "dirty.csv"),
        str(output_path),
        "--split",
        "0.6,0.2,0.2",
        "--no-verbose",
        "--no-dedup",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cleaned_train.parquet").exists()
    assert (tmp_path / "cleaned_val.parquet").exists()
    assert (tmp_path / "cleaned_test.parquet").exists()


def test_fusion_merge(tmp_path):
    sensor_dir = tmp_path / "sensors"
    sensor_dir.mkdir()
    imu = sensor_dir / "imu.csv"
    gps = sensor_dir / "gps.csv"
    imu.write_text("device_id,timestamp,accel\n1,2024-01-01T00:00:00,0.1\n")
    gps.write_text("device_id,timestamp,lat\n1,2024-01-01T00:00:01,42.0\n")
    output_path = tmp_path / "fused.parquet"

    result = run_cli(
        "--fusion-source",
        f"imu={imu}",
        "--fusion-source",
        f"gps={gps}",
        "--fusion-id-col",
        "device_id",
        "--fusion-time-col",
        "timestamp",
        "--fusion-time-tolerance",
        "2s",
        str(output_path),
        "--no-verbose",
    )
    if result.returncode != 0:
        raise AssertionError(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    assert result.returncode == 0, result.stderr
    assert output_path.exists()


def test_report_out(tmp_path):
    output_path = tmp_path / "cleaned.parquet"
    report_path = tmp_path / "report.md"
    result = run_cli(
        str(SAMPLES / "dirty.csv"),
        str(output_path),
        "--report-out",
        str(report_path),
        "--no-verbose",
        "--no-dedup",
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    assert report_path.exists()
    assert "# CleanShot Report" in report_path.read_text(encoding="utf-8")
