set -e

# Okay im not saying this is good but it works for now
pip install torch torchvision torchaudio xformers numpy --upgrade

apt-get install ffmpeg -y

cd alltracker || (echo "\"alltracker\" directory not found. Did you forget to clone recursively?"; exit)
# pip install -r requirements.txt
bash ./download_reference_model.sh
cd ..

cd VideoDepthAnything  || (echo "\"VideoDepthAnything\" directory not found. Did you forget to clone recursively?"; exit)
# pip install -r requirements.txt
bash ./get_weights.sh
cd checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true
cd ..
cd ..