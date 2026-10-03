SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$REPO_ROOT/deps"

mkdir -p "$DEPS_DIR"

python -m pip install -r resfit/lerobot/lerobot_requirements.txt
python -m pip install --upgrade torch torchvision torchcodec
