"""Reproducible conversion of official SAM 2.1 checkpoints to MLX FP16."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODEL_ID = "facebook/sam2.1-hiera-base-plus"
CONVERTER_COMMIT = "fbeafa7531be454b8f42d89c13d1e90024a63f6b"
EXPECTED_SOURCE_SHA256 = "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _cast_floating_tree(value: Any, mx, dtype):
    if isinstance(value, dict):
        return {key: _cast_floating_tree(item, mx, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_cast_floating_tree(item, mx, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_floating_tree(item, mx, dtype) for item in value)
    if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating):
        return value.astype(dtype)
    return value


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def _runtime_sidecar_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".json")


def converted_checkpoint_is_current(source: Path, output: Path) -> bool:
    manifest_path = _manifest_path(output)
    sidecar_path = _runtime_sidecar_path(output)
    if not output.exists() or not manifest_path.exists() or not sidecar_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        manifest.get("model_id") == MODEL_ID
        and manifest.get("precision") == "float16"
        and manifest.get("converter_commit") == CONVERTER_COMMIT
        and manifest.get("source_sha256") == _sha256(source)
    )


def ensure_fp16_checkpoint(source: str | Path, output: str | Path) -> Path:
    """Convert an official Meta checkpoint once and return the MLX FP16 path.

    Conversion first writes to temporary siblings and only replaces the final
    files after the result has loaded strictly. An interrupted conversion can
    therefore never leave a checkpoint that looks complete.
    """

    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Official SAM 2.1 checkpoint not found: {source}")
    if converted_checkpoint_is_current(source, output):
        return output

    source_sha256 = _sha256(source)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "The SAM 2.1 Base+ source checkpoint does not match Meta's expected "
            f"SHA-256. Expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}."
        )

    import mlx.core as mx
    from mlx_sam.convert import convert_checkpoint
    from mlx_sam.weights import load_image_segmenter

    output.parent.mkdir(parents=True, exist_ok=True)
    # MLX selects the serializer from the final suffix, so temporary files
    # must still end in .safetensors.
    fp32_tmp = output.with_name(output.stem + ".fp32.part.safetensors")
    fp16_tmp = output.with_name(output.stem + ".fp16.part.safetensors")
    for path in (fp32_tmp, fp16_tmp):
        path.unlink(missing_ok=True)

    try:
        convert_checkpoint(source, fp32_tmp, MODEL_ID)
        model = load_image_segmenter(fp32_tmp, model_id=MODEL_ID)
        model.update(_cast_floating_tree(model.parameters(), mx, mx.float16))
        mx.eval(model.parameters())
        model.save_weights(str(fp16_tmp))

        # Strictly reload before publishing the converted artifact.
        verified = load_image_segmenter(fp16_tmp, model_id=MODEL_ID)
        mx.eval(verified.parameters())
        tensor_count = len(mx.load(str(fp16_tmp)))

        runtime_sidecar = {
            "format": "mlx",
            "model_id": MODEL_ID,
            "source": str(source),
            "variant": "fp16",
        }
        manifest = {
            **runtime_sidecar,
            "precision": "float16",
            "source_sha256": source_sha256,
            "converter_commit": CONVERTER_COMMIT,
            "tensor_count": tensor_count,
            "size_bytes": fp16_tmp.stat().st_size,
        }

        os.replace(fp16_tmp, output)
        _runtime_sidecar_path(output).write_text(
            json.dumps(runtime_sidecar, indent=2) + "\n", encoding="utf-8"
        )
        _manifest_path(output).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return output
    finally:
        fp32_tmp.unlink(missing_ok=True)
        fp16_tmp.unlink(missing_ok=True)
