#mkb AAG
set -e  # stop on error

echo " Starting setup..."

# ───────────────────────────────
# System deps
# ───────────────────────────────
echo " Updating system..."
apt-get update
apt-get install -y zstd curl

# ───────────────────────────────
# Python deps
# ───────────────────────────────
echo " Installing Python packages..."

pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install vllm
pip install transformers accelerate
pip install snac
pip install websockets
pip install requests
pip install soundfile numpy

# ───────────────────────────────
# Ollama install
# ───────────────────────────────
echo " Installing Ollama..."

curl -fsSL https://ollama.com/install.sh | sh

# Starting Ollama in background
echo "⚡ Starting Ollama..."
nohup ollama serve > ollama.log 2>&1 &

# Waiting for server to boot
sleep 5

# Pulling model
echo " Pulling Gemma 3 4B..."
ollama pull gemma3:4b

echo " Setup complete! Maa Ka Bh@sDa AAAAG"

echo " Starting your server..."
python server.py