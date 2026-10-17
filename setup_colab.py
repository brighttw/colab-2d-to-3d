#!/usr/bin/env python3
"""
Setup script for Headless ComfyUI 3D Multi-View Generation on Google Colab (L4 GPU).
Saves model weights directly to Google Drive via symlinks to conserve Colab compute units.
"""
import os
import re
import subprocess
import sys
import threading
import time

# 1. 掛載 Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive 掛載成功。")
except ImportError:
    print("非 Colab 環境，跳過 Google Drive 掛載步驟。")

# 2. 定義目錄結構與持久化路徑
DRIVE_BASE = "/content/drive/MyDrive/ComfyUI_3D_Pipeline"
CACHE_HF_DIR = os.path.join(DRIVE_BASE, "hf_cache")
DIFFUSION_MODELS_DIR = os.path.join(DRIVE_BASE, "models", "diffusion_models")
REMBG_MODELS_DIR = os.path.join(DRIVE_BASE, "models", "rembg")

os.makedirs(CACHE_HF_DIR, exist_ok=True)
os.makedirs(DIFFUSION_MODELS_DIR, exist_ok=True)
os.makedirs(REMBG_MODELS_DIR, exist_ok=True)

# 導向 HuggingFace 快取至雲端硬碟 (加速 Paint/Delight 權重載入)
os.environ["HF_HOME"] = CACHE_HF_DIR

# 3. 複製 ComfyUI 本體
COMFY_DIR = "/content/ComfyUI"
if not os.path.exists(COMFY_DIR):
    print("正在下載 ComfyUI 核心...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", COMFY_DIR], check=True)

# 4. 建立軟連結
os.makedirs(os.path.join(COMFY_DIR, "models"), exist_ok=True)
target_diff = os.path.join(COMFY_DIR, "models", "diffusion_models")
if os.path.islink(target_diff):
    os.unlink(target_diff)
elif os.path.isdir(target_diff):
    subprocess.run(["rm", "-rf", target_diff], check=True)
os.symlink(DIFFUSION_MODELS_DIR, target_diff)
print(f"已連結 diffusion_models -> {DIFFUSION_MODELS_DIR}")

# 5. 安裝 Custom Nodes
CUSTOM_NODES_DIR = os.path.join(COMFY_DIR, "custom_nodes")
hy3d_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Hunyuan3DWrapper")
essentials_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI_essentials")

if not os.path.exists(hy3d_dir):
    print("下載 ComfyUI-Hunyuan3DWrapper...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/kijai/ComfyUI-Hunyuan3DWrapper.git", hy3d_dir], check=True)

if not os.path.exists(essentials_dir):
    print("下載 ComfyUI_essentials (去背/影像前處理)...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/cubiq/ComfyUI_essentials.git", essentials_dir], check=True)

# 6. 安裝 GPU 依賴與編譯 Rasterizer
print("安裝 Python 模組依賴...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(COMFY_DIR, "requirements.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(hy3d_dir, "requirements.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.46.0", "accelerate==1.1.1", "diffusers==0.30.0", "rembg[gpu]", "trimesh", "pymeshlab", "xatlas"], check=True)

# 防禦性補丁：確保 pipelines.py 支援 DINOv2 權重鍵名映射且清空所有 meta tensor
pipe_py = os.path.join(hy3d_dir, "hy3dgen", "shapegen", "pipelines.py")
if os.path.exists(pipe_py):
    with open(pipe_py, "r", encoding="utf-8") as f:
        pipe_code = f.read()
    if "🔥 Conditioner 權重加載" not in pipe_code:
        cond_replacement = '''        if 'conditioner' in new_sd:
            c_sd = new_sd['conditioner']
            loaded_count = 0
            cond_params = list(conditioner.named_parameters())
            for name, param in cond_params:
                val = c_sd.get(name)
                if val is None:
                    alias = name.replace('.attention.q_proj.', '.attention.attention.query.') \\
                                .replace('.attention.k_proj.', '.attention.attention.key.') \\
                                .replace('.attention.v_proj.', '.attention.attention.value.') \\
                                .replace('.attention.o_proj.', '.attention.output.dense.')
                    val = c_sd.get(alias)
                if val is None:
                    alias2 = name.replace('.attention.attention.query.', '.attention.q_proj.') \\
                                 .replace('.attention.attention.key.', '.attention.k_proj.') \\
                                 .replace('.attention.attention.value.', '.attention.v_proj.') \\
                                 .replace('.attention.output.dense.', '.attention.o_proj.')
                    val = c_sd.get(alias2)
                if val is not None:
                    set_module_tensor_to_device(conditioner, name, device=offload_device, dtype=dtype, value=val)
                    loaded_count += 1
                else:
                    set_module_tensor_to_device(conditioner, name, device=offload_device, dtype=dtype, value=torch.zeros(param.shape, dtype=dtype))
            print(f"🔥 Conditioner 權重加載: 成功加載 {loaded_count}/{len(cond_params)} 個參數！", flush=True)

        for m in [conditioner, model, vae]:
            for p_name, param in m.named_parameters():
                if getattr(param, 'is_meta', False):
                    set_module_tensor_to_device(m, p_name, device=offload_device, dtype=dtype, value=torch.zeros(param.shape, dtype=dtype))
            for b_name, buf in m.named_buffers():
                if getattr(buf, 'is_meta', False):
                    set_module_tensor_to_device(m, b_name, device=offload_device, dtype=buf.dtype, value=torch.zeros(buf.shape, dtype=buf.dtype))

        image_processor = instantiate_from_config(config['image_processor'])'''

        cond_regex = re.compile(r"        if 'conditioner' in new_sd:.*?image_processor = instantiate_from_config\(config\['image_processor'\]\)", re.DOTALL)
        if cond_regex.search(pipe_code):
            pipe_code = cond_regex.sub(cond_replacement, pipe_code, count=1)
            with open(pipe_py, "w", encoding="utf-8") as f:
                f.write(pipe_code)
            print("✅ 已對 pipelines.py 注入完整 DINOv2 權重映射與 Meta Tensor 清理補丁！")

# 防禦性補丁 2：確保 HunyuanPaintPipeline 支援 RGBA 自動白底合成轉 RGB
paint_pipe_py = os.path.join(hy3d_dir, "hy3dgen", "texgen", "hunyuanpaint", "pipeline.py")
if os.path.exists(paint_pipe_py):
    with open(paint_pipe_py, "r", encoding="utf-8") as f:
        p_code = f.read()
    p_target = "posterior = self.vae.encode(images.to(dtype)).latent_dist"
    p_safe = """if images.shape[1] == 4:
            alpha = images[:, 3:4]
            images = images[:, :3] * alpha + (1.0 - alpha)
        posterior = self.vae.encode(images.to(dtype)).latent_dist"""
    if p_target in p_code and "images.shape[1] == 4" not in p_code:
        p_code = p_code.replace(p_target, p_safe)
        with open(paint_pipe_py, "w", encoding="utf-8") as f:
            f.write(p_code)
        print("✅ 已對 hunyuanpaint/pipeline.py 注入 RGBA->RGB 兼容補丁！")

# 防禦性補丁 3：確保 nodes.py Hy3DSampleMultiView 支援 4 通道 ref_image
nodes_py = os.path.join(hy3d_dir, "nodes.py")
if os.path.exists(nodes_py):
    with open(nodes_py, "r", encoding="utf-8") as f:
        n_code = f.read()
    n_target = "multiview_images = pipeline("
    n_safe = """if hasattr(ref_image, 'shape') and ref_image.shape[-1] == 4:
            ref_image = ref_image[..., :3] * ref_image[..., 3:4] + (1.0 - ref_image[..., 3:4])
        multiview_images = pipeline("""
    if n_target in n_code and "ref_image.shape[-1] == 4" not in n_code:
        n_code = n_code.replace(n_target, n_safe, 1)
        with open(nodes_py, "w", encoding="utf-8") as f:
            f.write(n_code)
        print("✅ 已對 nodes.py 注入 ref_image RGBA->RGB 兼容補丁！")

# 編譯 Hunyuan3D 自訂 CUDA 渲染核心
rasterizer_dir = os.path.join(hy3d_dir, "hy3dgen", "texgen", "custom_rasterizer")
if os.path.exists(rasterizer_dir):
    print("編譯 custom_rasterizer CUDA 模組...")
    subprocess.run([sys.executable, "setup.py", "install"], cwd=rasterizer_dir, check=True)

# 7. 下載 Hunyuan3D-2 MV 核心 DiT 權重 (若 Drive 已存在則跳過)
mv_model_file = os.path.join(DIFFUSION_MODELS_DIR, "hunyuan3d-dit-v2-mv-fp16.safetensors")
if not os.path.exists(mv_model_file):
    print("正在將 hunyuan3d-dit-v2-mv-fp16.safetensors 下載至 Google Drive...")
    hf_url = "https://huggingface.co/tencent/Hunyuan3D-2mv/resolve/main/hunyuan3d-dit-v2-mv/model.fp16.safetensors"
    subprocess.run(["wget", "-c", "-O", mv_model_file, hf_url], check=True)
else:
    print("✅ 雲端硬碟已存在 hunyuan3d-dit-v2-mv-fp16.safetensors，跳過下載。")

# 8. 安裝 cloudflared
if not os.path.exists("/usr/local/bin/cloudflared"):
    print("安裝 Cloudflare Tunnel 客戶端...")
    subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"], check=True)
    subprocess.run(["dpkg", "-i", "cloudflared-linux-amd64.deb"], check=True)

# 9. 啟動背景穿透與 ComfyUI API 服務
def run_tunnel():
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8188", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in iter(proc.stdout.readline, ''):
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            print("\n" + "=" * 60)
            print(f"🚀 ComfyUI API 公開連線端點: {url}")
            print(f"請將此 URL 複製並設定給本地端 Agent (例如 COMFY_ENDPOINT={url})")
            print("=" * 60 + "\n")
            break

tunnel_thread = threading.Thread(target=run_tunnel, daemon=True)
tunnel_thread.start()

print("啟動 ComfyUI Headless 服務...")
# --highvram: 充分利用 L4 24GB 顯存；--disable-auto-launch 與 --dont-print-server 保持日誌精簡
subprocess.run([
    sys.executable,
    os.path.join(COMFY_DIR, "main.py"),
    "--listen", "127.0.0.1",
    "--port", "8188",
    "--highvram",
    "--disable-auto-launch"
])