SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$REPO_ROOT/deps"

mkdir -p "$DEPS_DIR"

python -m pip install mink==0.0.7 gymnasium==1.1.1

python -m pip install ipdb serial deepdiff
python -m pip install tabulate

python -m pip install torchrl==0.5.0 tensordict==0.5.0 torchcodec==0.4.0

python -m pip install protobuf==3.20.3 diffusers==0.33.1 multidict==6.0.5

python -m pip install "numba==0.58.1" "llvmlite==0.41.1"
python -m pip install "draccus==0.9.3"
