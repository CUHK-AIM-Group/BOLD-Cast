from __future__ import annotations
import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Central configuration: edit defaults here, or override them from the CLI.
# =============================================================================

DATASET_CONFIG: Dict[str, dict] = {
    "UKB": {
        "time_len": 81,
        "tr_seconds": 0.735,
        "start_time": "1950-01-01 00:00:00",
        "roi_dir": "UKB_roi",
        "phenotype_csv": "UKB.csv",
        "subject_prefix": "",
        "max_points": 490,
    },
    "HCP-YA": {
        "time_len": 81,
        "tr_seconds": 0.700,
        "start_time": "2002-01-01 00:00:00",
        "roi_dir": "HCP-YA_roi",
        "phenotype_csv": "HCP.csv",
        "subject_prefix": "",
        "max_points": 1200,
    },
    "HCP-D": {
        "time_len": 81,
        "tr_seconds": 0.700,
        "start_time": "2003-01-01 00:00:00",
        "roi_dir": "HCP-D_roi",
        "phenotype_csv": "hcp-d-rest.csv",
        "subject_prefix": "HCP-D",
        "max_points": 478,
    },
    "HCP-A": {
        "time_len": 81,
        "tr_seconds": 0.700,
        "start_time": "2004-01-01 00:00:00",
        "roi_dir": "HCP-A_roi",
        "phenotype_csv": "hcp-a-rest.csv",
        "subject_prefix": "HCP-A",
        "max_points": 478,
    },
    "ABIDE": {
        "time_len": 30,
        "tr_seconds": 2.000,
        "start_time": "2005-01-01 00:00:00",
        "roi_dir": "ABIDE_roi",
        "phenotype_csv": "ABIDE.csv",
        "subject_prefix": "",
        "max_points": 170,
    },
}

SPLITS = ("train", "val", "test")
DEFAULT_ATLAS = "CC200"
DEFAULT_NUM_ROIS = 190


@dataclass
class RuntimeConfig:
    dataset: str
    dataset_root: Path
    input_root: Path
    ts_root: Path
    sp_root: Path
    roi_dir: Path
    phenotype_csv: Optional[Path]
    atlas: str
    num_rois: int
    time_len: int
    tr_seconds: float
    start_time: datetime
    max_points: Optional[int]
    seq_len: int
    label_len: int
    pred_len: int
    llm_ckp_dir: Path
    gpu: int
    batch_size: int
    overwrite_ts: bool
    overwrite_sp: bool
    split_ratio: tuple
    split_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified BOLD-Cast preprocessing for Stage I and Stage II"
    )

    # Shared dataset / atlas parameters.
    parser.add_argument("--dataset", choices=DATASET_CONFIG.keys(), required=True)
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--atlas", type=str, default=DEFAULT_ATLAS)
    parser.add_argument("--num_rois", type=int, default=DEFAULT_NUM_ROIS)
    parser.add_argument("--time_len", type=int, default=None)
    parser.add_argument("--tr_seconds", type=float, default=None)

    # Raw inputs. Relative paths are interpreted under dataset_root.
    # roi_dir contains all unsplit subject ROI files.
    parser.add_argument("--roi_dir", type=str, default=None)
    parser.add_argument(
        "--split_ratio",
        type=str,
        default="0.8,0.1,0.1",
        help="train,val,test subject split ratio; default: 0.8,0.1,0.1",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="random seed used only for deterministic subject splitting",
    )
    parser.add_argument("--phenotype_csv", type=str, default=None)
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help="Output root, default: dataset/<dataset>_input",
    )

    # Stage-II window dimensions. Defaults are derived from time_len.
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--label_len", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=None)

    # GPT2 preprocessing.
    parser.add_argument("--llm_ckp_dir", type=str, default="Stage II/gpt2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)

    # Execution switches.
    parser.add_argument(
        "--skip_timestamp",
        action="store_true",
        help="Skip timestamp NPZ generation and use existing ts files.",
    )
    parser.add_argument(
        "--skip_sp",
        action="store_true",
        help="Skip Stage-II GPT2 time-semantic preprocessing.",
    )
    parser.add_argument("--overwrite_ts", action="store_true")
    parser.add_argument("--overwrite_sp", action="store_true")

    return parser.parse_args()


def _resolve_under(root: Path, value: Optional[str], default: Optional[str]) -> Optional[Path]:
    selected = value if value is not None else default
    if selected is None:
        return None
    p = Path(selected).expanduser()
    return p if p.is_absolute() else root / p


def build_config(args: argparse.Namespace) -> RuntimeConfig:
    base = DATASET_CONFIG[args.dataset]
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    time_len = args.time_len if args.time_len is not None else int(base["time_len"])
    tr_seconds = (
        args.tr_seconds if args.tr_seconds is not None else float(base["tr_seconds"])
    )

    try:
        split_ratio = tuple(float(x.strip()) for x in args.split_ratio.split(","))
    except Exception as exc:
        raise ValueError("--split_ratio must look like 0.8,0.1,0.1") from exc
    if len(split_ratio) != 3 or any(x < 0 for x in split_ratio):
        raise ValueError("--split_ratio must contain three non-negative values")
    ratio_sum = sum(split_ratio)
    if ratio_sum <= 0:
        raise ValueError("--split_ratio sum must be > 0")
    split_ratio = tuple(x / ratio_sum for x in split_ratio)

    # Keep Stage I and Stage II on the same basic temporal unit.
    label_len = args.label_len if args.label_len is not None else time_len
    pred_len = args.pred_len if args.pred_len is not None else time_len
    seq_len = args.seq_len if args.seq_len is not None else 2 * time_len

    if args.input_root:
        input_root = Path(args.input_root).expanduser()
        if not input_root.is_absolute():
            input_root = dataset_root / input_root
    else:
        input_root = dataset_root / f"{args.dataset}_input"

    llm_path = Path(args.llm_ckp_dir).expanduser()
    if not llm_path.is_absolute():
        llm_path = Path(__file__).resolve().parent / llm_path

    return RuntimeConfig(
        dataset=args.dataset,
        dataset_root=dataset_root,
        input_root=input_root.resolve(),
        ts_root=(input_root / "ts").resolve(),
        sp_root=(input_root / "sp").resolve(),
        roi_dir=_resolve_under(dataset_root, args.roi_dir, base["roi_dir"]).resolve(),
        phenotype_csv=(
            _resolve_under(dataset_root, args.phenotype_csv, base["phenotype_csv"]).resolve()
            if _resolve_under(dataset_root, args.phenotype_csv, base["phenotype_csv"])
            else None
        ),
        atlas=args.atlas,
        num_rois=args.num_rois,
        time_len=time_len,
        tr_seconds=tr_seconds,
        start_time=datetime.fromisoformat(base["start_time"]),
        max_points=base["max_points"],
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        llm_ckp_dir=llm_path.resolve(),
        gpu=args.gpu,
        batch_size=args.batch_size,
        overwrite_ts=args.overwrite_ts,
        overwrite_sp=args.overwrite_sp,
        split_ratio=split_ratio,
        split_seed=args.split_seed,
    )


def ensure_dirs(cfg: RuntimeConfig) -> None:
    for split in SPLITS:
        (cfg.ts_root / split).mkdir(parents=True, exist_ok=True)
        (cfg.sp_root / split).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Step 1: Stage-II-style timestamp NPZ generation.
# =============================================================================


def discover_roi_files(roi_dir: Path) -> List[Path]:
    """Discover all unsplit ROI files in one dataset directory."""
    if not roi_dir.is_dir():
        raise FileNotFoundError(f"ROI directory not found: {roi_dir}")
    files = sorted(roi_dir.glob("*_cc200.npy"))
    if not files:
        files = sorted(roi_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No ROI .npy files found in {roi_dir}")
    return files


def split_roi_files(files: List[Path], ratio: tuple, seed: int) -> Dict[str, List[Path]]:
    """Deterministically split subjects into train/val/test."""
    n = len(files)
    if n < 3:
        raise ValueError(f"Need at least 3 subjects to create train/val/test, got {n}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    shuffled = [files[i] for i in order]

    n_train = int(np.floor(n * ratio[0]))
    n_val = int(np.floor(n * ratio[1]))
    # Keep at least one validation and one test subject when possible.
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    n_test = n - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:n_train + n_val + n_test],
    }


def subject_id_from_path(path: Path) -> str:
    """Extract a stable numeric subject id from common ROI filenames."""
    stem = path.stem
    stem = re.sub(r"(?:\.nii)?_cc200$", "", stem, flags=re.IGNORECASE)
    groups = re.findall(r"\d+", stem)
    if not groups:
        raise ValueError(f"Cannot extract subject id from filename: {path.name}")
    return max(groups, key=len)


def load_phenotype(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.is_file():
        if path is not None:
            print(f"[warning] Phenotype CSV not found: {path}; only fMRI/text/corr will be saved.")
        return None
    return pd.read_csv(path)


def one_hot_sex(value) -> Optional[np.ndarray]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().upper()
    if s in {"M", "MALE", "1", "1.0"}:
        return np.asarray([1, 0], dtype=np.int64)
    if s in {"F", "FEMALE", "0", "0.0", "2", "2.0"}:
        return np.asarray([0, 1], dtype=np.int64)
    return None


def match_row(df: Optional[pd.DataFrame], dataset: str, sid: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None

    candidates = []
    if dataset == "hcpya":
        candidates = ["Subject", "subject_id", "subject", "participant_id"]
        targets = {sid, str(int(sid)) if sid.isdigit() else sid}
    elif dataset == "hcpd":
        candidates = ["subject_id", "Subject", "subject", "participant_id"]
        targets = {sid, f"HCD{sid}"}
    elif dataset == "hcpa":
        candidates = ["subject_id", "Subject", "subject", "participant_id"]
        targets = {sid, f"HCA{sid}"}
    else:
        candidates = ["subject_id", "Subject", "subject", "participant_id", "eid"]
        targets = {sid, str(int(sid)) if sid.isdigit() else sid}

    normalized_targets = {str(x).strip().upper() for x in targets}
    for col in candidates:
        if col not in df.columns:
            continue
        values = df[col].astype(str).str.strip().str.upper()
        hit = df[values.isin(normalized_targets)]
        if not hit.empty:
            return hit.iloc[0]
    return None


def metadata_for_subject(dataset: str, row: Optional[pd.Series]) -> Dict[str, np.ndarray]:
    """Reproduce the useful metadata keys from the original timestamp scripts.

    Missing phenotype columns are simply skipped; fMRI/text/corr are always generated.
    """
    if row is None:
        return {}

    out: Dict[str, np.ndarray] = {}

    # Sex: original files used a two-element one-hot vector.
    sex_col = next((c for c in ("sex", "SEX", "Gender", "gender") if c in row.index), None)
    if sex_col is not None:
        sex = one_hot_sex(row[sex_col])
        if sex is not None:
            out["sex"] = sex

    if dataset == "abide" and "DX_GROUP" in row.index:
        # Original convention: healthy=[1,0], ASD=[0,1].
        try:
            dx = int(row["DX_GROUP"])
            out["ASD"] = np.asarray([1, 0] if dx == 2 else [0, 1], dtype=np.int64)
        except Exception:
            pass

    if dataset == "hcpya":
        hcp_fields = {
            "ReadEng": "ReadEng_Unadj",
            "ProcSpeed": "ProcSpeed_Unadj",
            "Flanker": "Flanker_Unadj",
            "PicSeq": "PicSeq_Unadj",
            "SCPT": "SCPT_SEN",
        }
        for key, col in hcp_fields.items():
            if col in row.index and pd.notna(row[col]):
                out[key] = np.asarray([row[col]])

    # Preserve common scalar cognitive/demographic fields when present.
    # This does not replace dataset-specific downstream processing; it merely keeps
    # useful columns in the subject NPZ if they exist.
    reserved = {
        "subject_id", "Subject", "subject", "participant_id", "eid",
        "sex", "SEX", "Gender", "gender", "DX_GROUP",
    }
    for col in row.index:
        if col in reserved or col in out:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        if np.isscalar(value) and isinstance(value, (int, float, np.integer, np.floating)):
            out.setdefault(str(col), np.asarray([value]))

    return out


def make_timestamps(start: datetime, n: int, tr_seconds: float) -> np.ndarray:
    # Keep the original human-readable timestamp format for compatibility.
    # Fractional TRs may map multiple samples to the same displayed second, which is
    # acceptable because Stage-II semantic prompts below use the exact TR separately.
    return np.asarray(
        [(start + timedelta(seconds=i * tr_seconds)).strftime("%Y/%m/%d %H:%M:%S") for i in range(n)],
        dtype=str,
    )


def orient_fmri(arr: np.ndarray, num_rois: int, source: Path) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"ROI file must be 2-D, got {arr.shape}: {source}")
    if arr.shape[0] == num_rois:
        return arr
    if arr.shape[1] == num_rois:
        return arr.T
    raise ValueError(
        f"Cannot identify ROI axis in {source}: shape={arr.shape}, expected one axis={num_rois}"
    )


def generate_timestamp_npz(cfg: RuntimeConfig) -> None:
    print("\n[1/3] Generating Stage-II timestamp NPZ files ...")
    all_files = discover_roi_files(cfg.roi_dir)
    files_by_split = split_roi_files(all_files, cfg.split_ratio, cfg.split_seed)
    phenotype = load_phenotype(cfg.phenotype_csv)

    counts = {s: 0 for s in SPLITS}
    skipped = 0
    current_time = cfg.start_time

    # Split membership is generated once from the unsplit ROI subject list.
    # The fixed seed makes the split reproducible across repeated runs.
    for split in SPLITS:
        files = files_by_split[split]
        for source in tqdm(files, desc=f"timestamp-{split}"):
            sid = subject_id_from_path(source)
            dest = cfg.ts_root / split / f"{sid}.npz"
            if dest.exists() and not cfg.overwrite_ts:
                counts[split] += 1
                continue

            fmri = orient_fmri(
                np.asarray(np.load(source), dtype=np.float32), cfg.num_rois, source
            )
            if cfg.max_points is not None:
                fmri = fmri[:, : min(cfg.max_points, fmri.shape[1])]

            if fmri.shape[1] < cfg.time_len:
                print(f"[warning] skip {split}/{sid}: only {fmri.shape[1]} time points")
                skipped += 1
                continue

            corr = np.corrcoef(fmri).astype(np.float32)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            timestamps = make_timestamps(current_time, fmri.shape[1], cfg.tr_seconds)
            current_time = current_time + timedelta(
                seconds=fmri.shape[1] * cfg.tr_seconds
            )

            payload = {
                "corr": corr,
                "fMRI": fmri.astype(np.float32),
                "text": timestamps,
            }
            row = match_row(phenotype, cfg.dataset, sid)
            payload.update(metadata_for_subject(cfg.dataset, row))

            np.savez(dest, **payload)
            counts[split] += 1

    print(f"timestamp output: {cfg.ts_root}")
    print(f"subjects: {counts}; skipped={skipped}")


# =============================================================================
# Step 2: Shared Stage-I + Stage-II validation/preprocessing checks.
# =============================================================================


def validate_subject_npz(path: Path, cfg: RuntimeConfig) -> None:
    with np.load(path, allow_pickle=True) as data:
        missing = [k for k in ("fMRI", "text", "corr") if k not in data.files]
        if missing:
            raise KeyError(f"{path} is missing required key(s): {missing}")
        fmri = np.asarray(data["fMRI"])
        text = np.asarray(data["text"])
        corr = np.asarray(data["corr"])

    if fmri.ndim != 2 or fmri.shape[0] != cfg.num_rois:
        raise ValueError(
            f"{path}: fMRI shape={fmri.shape}, expected [{cfg.num_rois}, time]"
        )
    if fmri.shape[1] < cfg.time_len:
        raise ValueError(
            f"{path}: Stage I needs >= {cfg.time_len} time points, got {fmri.shape[1]}"
        )
    if len(text) != fmri.shape[1]:
        raise ValueError(f"{path}: text length={len(text)} != fMRI time={fmri.shape[1]}")
    if corr.shape != (cfg.num_rois, cfg.num_rois):
        raise ValueError(
            f"{path}: corr shape={corr.shape}, expected ({cfg.num_rois}, {cfg.num_rois})"
        )

    # Stage II's Dataset_our uses seq_len + pred_len to form a forecasting sample.
    minimum_stage2 = cfg.seq_len + cfg.pred_len
    if fmri.shape[1] < minimum_stage2:
        raise ValueError(
            f"{path}: Stage II needs >= seq_len + pred_len = {minimum_stage2} time points, "
            f"got {fmri.shape[1]}. Adjust seq_len/pred_len or the source crop."
        )


def validate_all(cfg: RuntimeConfig) -> Dict[str, List[Path]]:
    print("\n[2/3] Validating shared Stage-I / Stage-II subject NPZ files ...")
    split_files: Dict[str, List[Path]] = {}
    for split in SPLITS:
        files = sorted((cfg.ts_root / split).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No NPZ files found in {cfg.ts_root / split}")
        for path in tqdm(files, desc=f"validate-{split}"):
            validate_subject_npz(path, cfg)
        split_files[split] = files
        print(f"{split}: {len(files)} subjects OK")

    print(
        "Stage I uses these same files directly and later performs self-reconstruction: "
        f"x=fMRI[:, :{cfg.time_len}], y=x."
    )
    return split_files


# =============================================================================
# Step 3: Stage-II GPT2 time-semantic preprocessing.
# =============================================================================


def make_prompt(timestamp: str, window_len: int, tr_seconds: float) -> str:
    start = datetime.strptime(str(timestamp), "%Y/%m/%d %H:%M:%S")
    end = start + timedelta(seconds=(window_len - 1) * tr_seconds)
    return (
        "This is Time Series from "
        f"{start.strftime('%Y/%m/%d %H:%M:%S')} to "
        f"{end.strftime('%Y/%m/%d %H:%M:%S')}"
    )


def load_stage2_preprocess_model(cfg: RuntimeConfig):
    import torch
    repo_root = Path(__file__).resolve().parent
    stage2_dir = repo_root / "Stage II"
    if not stage2_dir.is_dir():
        raise FileNotFoundError(
            f"Stage II directory not found next to this script: {stage2_dir}"
        )
    if not cfg.llm_ckp_dir.exists():
        raise FileNotFoundError(f"GPT2 checkpoint directory not found: {cfg.llm_ckp_dir}")

    # Import the original Stage-II preprocessing model without copying model logic.
    sys.path.insert(0, str(stage2_dir))
    try:
        from models.Preprocess import Model  # type: ignore
    finally:
        # Keep Stage II importable for Model's own runtime dependencies.
        pass

    class ModelArgs:
        gpu = cfg.gpu if torch.cuda.is_available() else "cpu"
        llm_ckp_dir = str(cfg.llm_ckp_dir)

    model = Model(ModelArgs())
    model.eval()
    return model


def generate_sp_embeddings(cfg: RuntimeConfig, split_files: Dict[str, List[Path]]) -> None:
    import torch
    print("\n[3/3] Generating Stage-II GPT2 time-semantic embeddings ...")
    model = load_stage2_preprocess_model(cfg)

    # Original Stage-II preprocessing uses token_len = seq_len - label_len.
    token_len = cfg.seq_len - cfg.label_len
    if token_len <= 0:
        raise ValueError(
            f"seq_len - label_len must be positive, got {cfg.seq_len} - {cfg.label_len}"
        )
    if token_len != cfg.time_len:
        print(
            f"[warning] token_len={token_len} differs from shared time_len={cfg.time_len}. "
            "This is allowed because you explicitly overrode Stage-II dimensions."
        )

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"GPT2 preprocessing device: {device}")
    print(
        f"shared time_len={cfg.time_len}; Stage II seq/label/pred="
        f"{cfg.seq_len}/{cfg.label_len}/{cfg.pred_len}; token_len={token_len}"
    )

    with torch.no_grad():
        for split in SPLITS:
            for npz_path in tqdm(split_files[split], desc=f"sp-{split}"):
                out_path = cfg.sp_root / split / f"{npz_path.name}.pt"
                if out_path.exists() and not cfg.overwrite_sp:
                    continue

                with np.load(npz_path, allow_pickle=True) as data:
                    timestamps = [str(x) for x in data["text"]]

                prompts = [
                    make_prompt(ts, token_len, cfg.tr_seconds) for ts in timestamps
                ]

                outputs = []
                for begin in range(0, len(prompts), cfg.batch_size):
                    batch = prompts[begin : begin + cfg.batch_size]
                    out = model(batch)
                    outputs.append(out.detach().cpu())

                result = torch.cat(outputs, dim=0)
                torch.save(result, out_path)

    print(f"Stage-II semantic embeddings saved to: {cfg.sp_root}")


def print_config(cfg: RuntimeConfig) -> None:
    print("=" * 78)
    print("BOLD-Cast unified preprocessing")
    print("=" * 78)
    print(f"dataset       : {cfg.dataset}")
    print(f"dataset_root  : {cfg.dataset_root}")
    print(f"roi_dir       : {cfg.roi_dir}  (unsplit ROI subjects)")
    print(f"split_ratio   : {cfg.split_ratio}")
    print(f"split_seed    : {cfg.split_seed}")
    print(f"input_root    : {cfg.input_root}")
    print(f"atlas / ROIs  : {cfg.atlas} / {cfg.num_rois}")
    print(f"time_len      : {cfg.time_len}")
    print(f"TR (seconds)  : {cfg.tr_seconds}")
    print(f"seq/label/pred: {cfg.seq_len}/{cfg.label_len}/{cfg.pred_len}")
    print(f"LLM checkpoint: {cfg.llm_ckp_dir}")
    print("=" * 78)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    ensure_dirs(cfg)
    print_config(cfg)

    if not args.skip_timestamp:
        generate_timestamp_npz(cfg)

    split_files = validate_all(cfg)

    if not args.skip_sp:
        generate_sp_embeddings(cfg, split_files)

    print("\nDone.")
    print(f"Stage I root_path should be: {cfg.ts_root}")
    print(f"Stage II root_path should be: {cfg.ts_root}")
    print(f"Stage II semantic path is   : {cfg.sp_root}")


if __name__ == "__main__":
    main()
