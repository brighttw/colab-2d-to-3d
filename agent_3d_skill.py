#!/usr/bin/env python3
"""
Agent 3D Skill: Automate 2D multi-view image upload, ComfyUI 3D generation, and Godot 4 asset download.
Part of the Colab 2D-to-3D pipeline using Hunyuan3D-2 MV.
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from PIL import Image, ImageOps
import websocket


def _resolve_urls(endpoint: str):
    """Normalize endpoint and derive HTTP and WebSocket URLs."""
    clean_endpoint = endpoint.strip().rstrip("/")
    if not clean_endpoint.startswith("http://") and not clean_endpoint.startswith("https://"):
        clean_endpoint = "https://" + clean_endpoint

    parsed = urllib.parse.urlparse(clean_endpoint)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/ws"
    http_base = f"{parsed.scheme}://{parsed.netloc}"
    return http_base, ws_url


def upload_image(http_base: str, image_path: str, filename_override: Optional[str] = None, mirror_horizontal: bool = False) -> str:
    """Upload an image to ComfyUI /upload/image endpoint and return the uploaded filename."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到輸入圖片: {image_path}")

    upload_url = f"{http_base}/upload/image"

    if mirror_horizontal:
        img = Image.open(image_path)
        img = ImageOps.mirror(img)
        img_buffer = io.BytesIO()
        img_format = img.format if img.format else "PNG"
        img.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        file_bytes = img_buffer.read()
        file_name = filename_override or f"mirrored_{os.path.basename(image_path)}"
    else:
        with open(image_path, "rb") as f:
            file_bytes = f.read()
        file_name = filename_override or os.path.basename(image_path)

    files = {"image": (file_name, file_bytes)}
    data = {"overwrite": "true"}

    response = requests.post(upload_url, files=files, data=data, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"圖片上傳失敗 ({response.status_code}): {response.text}")

    resp_json = response.json()
    uploaded_name = resp_json.get("name", file_name)
    return uploaded_name


def monitor_progress_ws(ws_url: str, client_id: str, prompt_id: str, node_titles: Dict[str, str], timeout_seconds: int = 1200):
    """Monitor task progress via ComfyUI WebSocket stream."""
    ws_endpoint = f"{ws_url}?clientId={client_id}"
    ws = websocket.create_connection(ws_endpoint, timeout=timeout_seconds)

    print(f"📡 WebSocket 連線成功: {ws_endpoint}")
    print(f"⏳ 開始等待 3D 生成任務完成 (Prompt ID: {prompt_id})...")

    current_node = None
    start_time = time.time()

    try:
        while True:
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"任務執行逾時 ({timeout_seconds}s)")

            raw_msg = ws.recv()
            if not raw_msg:
                continue

            if isinstance(raw_msg, bytes):
                # Binary preview data, skip
                continue

            msg = json.loads(raw_msg)
            msg_type = msg.get("type")
            data = msg.get("data", {})

            if msg_type == "status":
                queue_remaining = data.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
                # print status occasionally if needed

            elif msg_type == "execution_start":
                if data.get("prompt_id") == prompt_id:
                    print("🚀 Pipeline 執行已正式啟動！")

            elif msg_type == "executing":
                node_id = data.get("node")
                msg_prompt_id = data.get("prompt_id")

                if msg_prompt_id == prompt_id:
                    if node_id is None:
                        print("🎉 3D 模型生成完畢，所有節點已成功執行！")
                        break
                    else:
                        current_node = node_id
                        title = node_titles.get(str(node_id), f"Node #{node_id}")
                        print(f"⚙️  正在處理節點 [{node_id}] {title}...")

            elif msg_type == "progress":
                val = data.get("value", 0)
                max_val = data.get("max", 1)
                pct = int((val / max_val) * 100) if max_val > 0 else 0
                title = node_titles.get(str(current_node), f"Node #{current_node}")
                sys.stdout.write(f"\r   進度: [{title}] {val}/{max_val} ({pct}%)")
                sys.stdout.flush()
                if val >= max_val:
                    print()

            elif msg_type == "execution_error":
                if data.get("prompt_id") == prompt_id:
                    err_node = data.get("node_id")
                    err_msg = data.get("exception_message", "未知錯誤")
                    raise RuntimeError(f"節點 [{err_node}] 執行出錯: {err_msg}")

    finally:
        ws.close()


def poll_history_fallback(http_base: str, prompt_id: str, timeout_seconds: int = 1200) -> Dict[str, Any]:
    """Fallback polling for history if WebSocket connection fails."""
    history_url = f"{http_base}/history/{prompt_id}"
    start_time = time.time()
    print("⚠️ 正在使用 HTTP 輪詢方式監聽任務進度...")

    while time.time() - start_time < timeout_seconds:
        resp = requests.get(history_url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if prompt_id in data:
                return data[prompt_id]
        time.sleep(5)

    raise TimeoutError(f"HTTP 輪詢逾時 ({timeout_seconds}s)")


def fetch_and_save_glb(http_base: str, prompt_id: str, output_glb_path: str, expected_prefix: str):
    """Retrieve output .glb filename from ComfyUI history and download it to local disk."""
    history_url = f"{http_base}/history/{prompt_id}"
    resp = requests.get(history_url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"無法取得執行歷史 ({resp.status_code}): {resp.text}")

    history = resp.json().get(prompt_id, {})
    outputs = history.get("outputs", {})

    target_file = None
    target_subfolder = ""
    target_type = "output"

    # Search for exported mesh node or any .glb output
    for node_id, node_out in outputs.items():
        # Check files / meshes in output
        for key in ("files", "images", "mesh"):
            items = node_out.get(key, [])
            for item in items:
                fname = item.get("filename", "")
                if fname.endswith(".glb"):
                    target_file = fname
                    target_subfolder = item.get("subfolder", "")
                    target_type = item.get("type", "output")
                    break
            if target_file:
                break
        if target_file:
            break

    if not target_file:
        # Fallback to expected filename prefix
        target_file = f"{expected_prefix}.glb"

    print(f"📦 準備自 ComfyUI 下載生成模型: {target_file}")
    view_url = f"{http_base}/view?filename={urllib.parse.quote(target_file)}&subfolder={urllib.parse.quote(target_subfolder)}&type={urllib.parse.quote(target_type)}"

    dl_resp = requests.get(view_url, timeout=180, stream=True)
    if dl_resp.status_code != 200:
        raise RuntimeError(f"下載模型失敗 ({dl_resp.status_code}): {dl_resp.text}")

    out_path = Path(output_glb_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        for chunk in dl_resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✅ 模型已成功儲存至: {out_path.resolve()} (檔案大小: {file_size_mb:.2f} MB)")
    return str(out_path.resolve())


def generate_godot_model(
    front_img: str,
    back_img: str,
    side_img: str,
    output_glb_path: str,
    target_faces: int = 15000,
    side_orientation: str = "left",
    endpoint: Optional[str] = None,
    workflow_path: str = "workflow_api.json"
) -> str:
    """
    Main function to execute the 2D-to-3D generation pipeline.

    :param front_img: Path to front-view image
    :param back_img: Path to back-view image
    :param side_img: Path to side-view image
    :param output_glb_path: Target path for .glb file (e.g. godot_project/assets/models/character.glb)
    :param target_faces: Target polygon count (decimation)
    :param side_orientation: 'left' or 'right'
    :param endpoint: ComfyUI server URL (e.g. https://xxx.trycloudflare.com or from COMFY_ENDPOINT env)
    :param workflow_path: Path to workflow_api.json
    :return: Absolute path to the saved .glb model
    """
    if endpoint is None:
        endpoint = os.getenv("COMFY_ENDPOINT")
    if not endpoint:
        raise ValueError("未指定 ComfyUI 端點！請傳入 endpoint 參數或設定 COMFY_ENDPOINT 環境變數。")

    if not os.path.exists(workflow_path):
        raise FileNotFoundError(f"找不到工作流設定檔: {workflow_path}")

    http_base, ws_url = _resolve_urls(endpoint)
    client_id = str(uuid.uuid4())

    print("============================================================")
    print("🚀 開始執行 2D 三視圖生 3D 模型管線")
    print(f"   端點: {http_base}")
    print(f"   正面: {front_img}")
    print(f"   背面: {back_img}")
    print(f"   側面: {side_img} (視角: {side_orientation})")
    print(f"   面數: {target_faces}")
    print(f"   輸出: {output_glb_path}")
    print("============================================================")

    # 1. 處理側面鏡像 (Hunyuan3D-2 MV 預設 left view 規範)
    mirror_side = (side_orientation.lower() == "right")

    # 2. 上傳三視圖
    print("\n📤 [1/4] 正在上傳三視圖至雲端 ComfyUI...")
    uploaded_front = upload_image(http_base, front_img)
    uploaded_back = upload_image(http_base, back_img)
    uploaded_side = upload_image(http_base, side_img, mirror_horizontal=mirror_side)
    print(f"   已上傳正面圖: {uploaded_front}")
    print(f"   已上傳背面圖: {uploaded_back}")
    print(f"   已上傳側面圖: {uploaded_side} {'(已水平鏡像轉為左視角)' if mirror_side else ''}")

    # 3. 讀取並注入工作流
    print("\n⚙️  [2/4] 正在配置工作流參數...")
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    output_prefix = f"godot_mesh_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    # 動態注入圖片檔名與設定
    if "10" in workflow and "inputs" in workflow["10"]:
        workflow["10"]["inputs"]["image"] = uploaded_front
    if "11" in workflow and "inputs" in workflow["11"]:
        workflow["11"]["inputs"]["image"] = uploaded_back
    if "12" in workflow and "inputs" in workflow["12"]:
        workflow["12"]["inputs"]["image"] = uploaded_side
    if "32" in workflow and "inputs" in workflow["32"]:
        workflow["32"]["inputs"]["max_facenum"] = int(target_faces)
    if "40" in workflow and "inputs" in workflow["40"]:
        workflow["40"]["inputs"]["filename_prefix"] = output_prefix

    node_titles = {
        nid: ninfo.get("_meta", {}).get("title", ninfo.get("class_type", f"Node_{nid}"))
        for nid, ninfo in workflow.items()
    }

    # 4. 發送 Prompt 任務
    print("\n📨 [3/4] 正在送出推論任務至 ComfyUI...")
    prompt_payload = {
        "prompt": workflow,
        "client_id": client_id
    }
    prompt_resp = requests.post(f"{http_base}/prompt", json=prompt_payload, timeout=30)
    if prompt_resp.status_code != 200:
        raise RuntimeError(f"提交任務失敗 ({prompt_resp.status_code}): {prompt_resp.text}")

    prompt_data = prompt_resp.json()
    prompt_id = prompt_data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"伺服器未回傳有效 prompt_id: {prompt_data}")

    # 5. 監控生成進度
    try:
        monitor_progress_ws(ws_url, client_id, prompt_id, node_titles)
    except Exception as e:
        print(f"⚠️  WebSocket 監控中斷 ({e})，切換至 HTTP 輪詢模式...")
        poll_history_fallback(http_base, prompt_id)

    # 6. 下載並儲存 .glb
    print("\n📥 [4/4] 正在下載產出的 3D GLB 模型...")
    saved_path = fetch_and_save_glb(http_base, prompt_id, output_glb_path, output_prefix)

    print("\n✨ 3D 模型生成全流程完成！")
    return saved_path


def main():
    parser = argparse.ArgumentParser(description="全自動三視圖生成 Godot 4 相容 3D GLB 模型 CLI 工具")
    parser.add_argument("front", help="正面視角圖路徑 (front.png)")
    parser.add_argument("back", help="背面視角圖路徑 (back.png)")
    parser.add_argument("side", help="側面視角圖路徑 (side.png)")
    parser.add_argument("output", help="輸出 .glb 儲存路徑 (例: godot_project/assets/models/character.glb)")
    parser.add_argument("target_faces", nargs="?", type=int, default=15000, help="目標三角面數 (預設: 15000)")
    parser.add_argument("endpoint", nargs="?", default=None, help="ComfyUI 伺服器端點 (預設讀取 COMFY_ENDPOINT 環境變數)")
    parser.add_argument("--side-orientation", choices=["left", "right"], default="left", help="側面圖角度 (預設: left)")
    parser.add_argument("--workflow", default="workflow_api.json", help="ComfyUI 工作流 JSON 檔案路徑")

    args = parser.parse_args()

    endpoint = args.endpoint or os.getenv("COMFY_ENDPOINT")
    if not endpoint:
        print("❌ 錯誤: 請提供 endpoint 參數或設定 COMFY_ENDPOINT 環境變數！", file=sys.stderr)
        sys.exit(1)

    try:
        saved_path = generate_godot_model(
            front_img=args.front,
            back_img=args.back,
            side_img=args.side,
            output_glb_path=args.output,
            target_faces=args.target_faces,
            side_orientation=args.side_orientation,
            endpoint=endpoint,
            workflow_path=args.workflow
        )
        print(f"\n🎉 成功產出資產: {saved_path}")
    except Exception as err:
        print(f"\n❌ 生成流程發生異常: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
