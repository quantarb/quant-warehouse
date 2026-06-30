#!/usr/bin/env bash
set -euo pipefail

VERSION="${VECTOR_TA_VERSION:-0.2.8}"
WORKDIR="${VECTOR_TA_BUILD_DIR:-/tmp/vector_ta_cuda_build}"

python -m pip install "maturin>=1.9,<2.0" patchelf
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"

python -m pip download --no-deps --no-binary=:all: "vector-ta==${VERSION}" -d "${WORKDIR}"
tar -xf "${WORKDIR}/vector_ta-${VERSION}.tar.gz" -C "${WORKDIR}"

cd "${WORKDIR}/vector_ta-${VERSION}"
python -m maturin build --release --features "python cuda" -i python
python -m pip install --force-reinstall "target/wheels/vector_ta-${VERSION}"-*.whl
python -m pip install --force-reinstall "numpy>=1.26,<2.5"

python - <<'PY'
import importlib.metadata as md
import importlib.util

import vector_ta

extension = importlib.util.find_spec("vector_ta.vector_ta").origin
cuda_symbol_count = sum(1 for name in dir(vector_ta) if "cuda" in name.lower())

print(f"vector-ta {md.version('vector-ta')}")
print(f"extension {extension}")
print(f"cuda_symbol_count {cuda_symbol_count}")
if cuda_symbol_count <= 0:
    raise SystemExit("VectorTA CUDA symbols were not found.")
PY
