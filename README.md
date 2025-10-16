# podcast-img2vid-pipeline


README.md（完整成稿）
podcast-img2vid-pipeline

把「靜態圖片為主的 Podcast 影片」自動轉成有動態畫面的版本：
Step 1 擷取場景與代表圖 → Step 2 本地 I2V（Image-to-Video，Stable Video Diffusion）→ Step 3 依時間戳重組影片並合成原音訊。

適合有大量靜態插圖、每幾秒換一張圖的兒童 Podcast / 教學影片，把畫面動起來、提升觀感。

Features

場景偵測：用 PySceneDetect 擷取每段圖片與時間戳（timestamps.csv）。

本地 I2V：Windows / WSL2 直接跑 Stable Video Diffusion（Diffusers），免按秒計費。

聲畫對齊：自動對齊至原始音訊長度；短片不足可 loop / freeze 補齊。

一鍵重組：批次拼接所有動態短片，輸出 H.264 + AAC 的成品 MP4。

Directory Layout
repo-root/
├─ animate_podcast_v2.py          # Step 1 & Step 3（場景擷取 / 重組合成）
├─ step2_local_i2v_svd.py         # Step 2（本地 I2V：Stable Video Diffusion）
├─ 001.mp4                        # 你的原始 Podcast 影片（範例檔名）
├─ 1_extracted_images/            # Step 1 輸出（image_001.png, …）
├─ 2_animated_clips/              # Step 2 輸出（image_001.mp4, …）
├─ requirements.txt               # 主要 Python 依賴（可選）
└─ README.md

Environment Setup
A. Windows 11（原生）

安裝 Python 3.10/3.11 (64-bit)，安裝時勾選 Add Python to PATH。

安裝 FFmpeg（將 ffmpeg\bin 加入 PATH）。

建立 venv：

cd <your repo>
python -m venv .venv
.\.venv\Scripts\Activate.ps1


安裝 PyTorch（CUDA 12.4 版） 與依賴：

pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install diffusers transformers accelerate safetensors imageio-ffmpeg opencv-python tqdm pandas scenedetect moviepy
pip install "scenedetect[opencv]"


驗證 GPU：

python - << 'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0), "CUDA:", torch.version.cuda)
PY


登入 Hugging Face（下載模型權重）：

huggingface-cli login


在模型頁面（stabilityai/stable-video-diffusion-img2vid）若需接受條款，請先按「Accept」。

B. WSL2（Ubuntu 22.04/24.04）

需先在 Windows 更新 NVIDIA 驅動，並執行 wsl --update。在 WSL 內 不需要安裝 Linux 版顯卡驅動或 CUDA Toolkit。

sudo apt update
sudo apt install -y python3-venv python3-pip git ffmpeg
mkdir -p ~/podcast-i2vid && cd ~/podcast-i2vid    # 或進入你的 repo
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install diffusers transformers accelerate safetensors imageio-ffmpeg opencv-python tqdm pandas scenedetect moviepy
pip install "scenedetect[opencv]"

python - << 'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "CUDA:", torch.version.cuda)
PY

huggingface-cli login


效能建議：大量 I/O 放在 WSL 的 ext4（例如 ~/podcast-i2vid），避免 /mnt/c 造成磁碟瓶頸。

How to Use
1) Step 1 — 場景偵測與擷取代表圖
# Windows PowerShell 或 WSL bash，需在已啟用 venv 的情況下
python animate_podcast_v2.py "001.mp4" --step 1 -t 30


產出：

1_extracted_images/image_001.png, image_002.png, ...

timestamps.csv（每段起訖與時長）

-t/--threshold 越低越敏感（預設 30，適合多數靜態圖切換的情境）。

2) Step 2 — 本地 I2V（把每張圖轉成短片）
python step2_local_i2v_svd.py


預設模型：stabilityai/stable-video-diffusion-img2vid（Diffusers）

腳本會讀取 1_extracted_images/*.png，輸出到 2_animated_clips/*.mp4

可在檔案內調整重點參數：

HEIGHT, WIDTH：解析度（例：576x1024）；解析度越高越吃 VRAM/時間

NUM_FRAMES：常見 14 或 25

FPS：常見 8–12

GUIDANCE_SCALE：動態強度 1.0–1.5

DECODE_CHUNK_SIZE：低記憶體技巧（1 或 2）

SEED：固定種子，便於重現

3) Step 3 — 重組與合成（保留原音訊）
# -p 可選 loop 或 freeze（短於片段時長時的補齊策略）
python animate_podcast_v2.py "001.mp4" --step 3 -p loop


程式會：

讀 timestamps.csv 與 2_animated_clips/image_XXX.mp4

逐段套 loop/freeze 使短片長度吻合時間戳

正規化解析度

合併所有片段並加回原始音訊

輸出：001_animated_v2.mp4

Requirements（可選的 requirements.txt）

若你希望一鍵安裝（CPU/GPU 請依 PyTorch 官網調整），可放下列內容供參考：

# requirements.txt
diffusers
transformers
accelerate
safetensors
imageio-ffmpeg
opencv-python
tqdm
pandas
scenedetect
moviepy


PyTorch 請依你的平台與 CUDA 版本另外安裝（例如 cu124）。

Troubleshooting

CUDA available: False

Windows 端先確認 nvidia-smi 正常；wsl --update（若用 WSL）；重進 venv；重新安裝對應 CUDA 版的 PyTorch（cu124）。

Hugging Face 權限/403

huggingface-cli login 重新登入；到模型頁面接受條款。

VRAM 不足（OOM）

降解析度（例如 480×848 或 576×1024）、NUM_FRAMES=14、FPS=8–12、DECODE_CHUNK_SIZE=1。

關閉其他佔 GPU 的程式（包含瀏覽器硬體加速）。

輸出卡頓或合成長度不對

Step 3 預設會對齊到音訊長度；若畫面短於音訊，會自動補尾巴（freeze）。

效能慢

先用低參數測流程，確認穩定再提高；固定 SEED 方便比對品質差異。

檔名對不到

必須遵循 image_001.png → image_001.mp4 的對應規則。

Roadmap（可選）

 Step 2 CLI 參數化（--height/--width/--num-frames/--fps/--seed）

 Provider 介面（可切換 Runway/Pika/Luma API 作為雲端備援）

 Ken Burns 輕運鏡後處理（推拉/平移）

 Drift 校正（整片微量重採樣，讓聲畫長度誤差 < 10ms）

License

MIT License（除非你另有授權需求）

Disclaimer

請確認向量資產與圖片擁有合法授權；若包含兒童影像，請遵守平台與法規的內容規範。

使用第三方模型需遵守其條款（Stable Video Diffusion、Diffusers、Hugging Face 等）。

致謝

PySceneDetect

Diffusers / Stable Video Diffusion (img2vid)

MoviePy