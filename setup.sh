set -e
cd alltracker || echo "\"alltracker\" directory not found. Did you forget to clone recursively?" && exit
pip install -r requirements.txt
./download_reference_model.sh
cd ..

cd Depth-Anything-V2  || echo "\"Depth-Anything-V2\" directory not found. Did you forget to clone recursively?" && exit
pip install -r requirements.txt
cd checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true
cd ..
cd ..