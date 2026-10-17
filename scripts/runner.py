#!/usr/bin/env python3
"""
Colab Runner for 2D Multi-View to 3D GLB Pipeline.
Automates Colab session creation, file uploading, remote inference, and asset downloading using google-colab-cli.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional


SKILL_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = SKILL_DIR / "assets" / "Hunyuan3D_2_Colab.ipynb"
WORKFLOW_JSON = SKILL_DIR / "workflow_api.json"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
AUTH = os.environ.get("COLAB_AUTH", "oauth2")


class ColabCommandError(RuntimeError):
    def __init__(self, label: str, returncode: int, output: list[str]):
        self.label = label
        self.returncode = returncode
        self.output = output
        tail = "\n".join(line for line in output[-15:] if line)
        super().__init__(f"{label} 失敗 (結束代碼 {returncode})" + (f":\n{tail}" if tail else ""))


class ColabTimeoutError(TimeoutError):
    pass


def clean_output(value: str) -> str:
    return ANSI_RE.sub("", value).replace("\r", "").strip()


def find_colab_cli() -> str:
    path = shutil.which("colab")
    if not path:
        custom_paths = [
            Path.home() / ".local" / "bin" / "colab",
            Path.home() / ".cargo" / "bin" / "colab",
            Path("/opt/homebrew/bin/colab"),
        ]
        for p in custom_paths:
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
        raise FileNotFoundError("找不到 colab 命令列工具！請先安裝 google-colab-cli。")
    return path


def call_colab(
    arguments: list[str],
    *,
    label: str,
    timeout: float,
    on_line: Callable[[str], None] | None = None,
) -> str:
    colab_bin = find_colab_cli()
    command = [colab_bin, f"--auth={AUTH}", *arguments]
    child = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def collect() -> None:
        assert child.stdout is not None
        for line in child.stdout:
            lines.put(line.rstrip("\n"))
        lines.put(None)

    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    output: list[str] = []
    eof = False
    started = time.monotonic()

    while not eof or child.poll() is None:
        if time.monotonic() - started > timeout and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            reader.join(timeout=2)
            if child.stdout is not None:
                child.stdout.close()
            raise ColabTimeoutError(f"{label} 超過 {timeout:g} 秒逾時限制。")

        try:
            item = lines.get(timeout=0.25)
        except queue.Empty:
            continue

        if item is None:
            eof = True
            continue

        line = clean_output(item)
        output.append(line)
        if on_line:
            on_line(line)

    returncode = child.wait()
    reader.join(timeout=2)
    if child.stdout is not None:
        child.stdout.close()

    if returncode != 0:
        raise ColabCommandError(label, returncode, output)
    return "\n".join(line for line in output if line)


def start_session(session: str, gpu: str = "L4", high_mem: bool = True) -> None:
    print(f"🚀 正在建立 Google Colab 會話 (Session: {session}, GPU: {gpu})...")
    args = ["new", "--session", session, "--gpu", gpu]
    if high_mem:
        args.append("--high-mem")
    output = call_colab(args, label="建立 Colab 會話", timeout=900, on_line=lambda l: print(f"   [Colab] {l}"))
    print(f"✅ 會話建立完成！")


def stop_session(session: str) -> None:
    print(f"🛑 正在釋放並停止 Colab 會話以節省額度: {session}...")
    try:
        call_colab(["stop", "--session", session], label="停止 Colab 會話", timeout=300, on_line=lambda l: print(f"   [Colab] {l}"))
        print("✅ 會話已成功停止。")
    except Exception as e:
        print(f"⚠️ 停止會話時出現警告: {e}")


def run_single_inference(
    front_path: str,
    back_path: str,
    side_path: str,
    output_path: str,
    target_faces: int = 15000,
    gpu: str = "L4",
    high_mem: bool = True,
    session: Optional[str] = None,
    keep_session: bool = False,
    timeout: float = 3600,
) -> str:
    """Run full automated 2D-to-3D inference on Google Colab via CLI."""
    front = Path(front_path).resolve()
    back = Path(back_path).resolve()
    side = Path(side_path).resolve()
    output = Path(output_path).resolve()

    if not front.is_file():
        raise FileNotFoundError(f"正面圖片不存在: {front}")
    if not back.is_file():
        raise FileNotFoundError(f"背面圖片不存在: {back}")
    if not side.is_file():
        raise FileNotFoundError(f"側面圖片不存在: {side}")
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"找不到推論 Notebook: {NOTEBOOK}")
    if not WORKFLOW_JSON.is_file():
        raise FileNotFoundError(f"找不到工作流設定: {WORKFLOW_JSON}")

    owns_session = session is None
    if session is None:
        session = f"hy3d-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"

    output.parent.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("✨ 全自動 Colab 3D 模型生成管線啟動")
    print(f"   Session: {session} | GPU: {gpu}")
    print(f"   正面: {front.name} | 背面: {back.name} | 側面: {side.name}")
    print(f"   目標三角面數: {target_faces}")
    print(f"   目標導出位置: {output}")
    print("============================================================")

    try:
        if owns_session:
            start_session(session, gpu=gpu, high_mem=high_mem)

        # 1. 上傳檔案至遠端 /content/
        remote_front = "/content/input_front.png"
        remote_back = "/content/input_back.png"
        remote_side = "/content/input_side.png"
        remote_workflow = "/content/workflow_api.json"
        remote_glb = "/content/output_model.glb"

        print("\n📤 [1/4] 正在上傳三視圖與工作流至 Colab 遠端環境...")
        call_colab(["upload", "--session", session, str(front), remote_front], label="上傳正面圖", timeout=300)
        call_colab(["upload", "--session", session, str(back), remote_back], label="上傳背面圖", timeout=300)
        call_colab(["upload", "--session", session, str(side), remote_side], label="上傳側面圖", timeout=300)
        call_colab(["upload", "--session", session, str(WORKFLOW_JSON), remote_workflow], label="上傳工作流", timeout=120)
        print("✅ 檔案上傳完畢！")

        # 2. 執行推論 Notebook
        print("\n⚙️  [2/4] 正在 Colab 遠端 GPU 執行推論 Notebook (這可能需要數分鐘)...")
        exec_args = [
            "exec",
            "--session", session,
            "--timeout", str(int(timeout)),
            "--env", f"HY3D_FRONT_IMAGE={remote_front}",
            "--env", f"HY3D_BACK_IMAGE={remote_back}",
            "--env", f"HY3D_SIDE_IMAGE={remote_side}",
            "--env", f"HY3D_TARGET_FACES={target_faces}",
            "--env", f"HY3D_OUTPUT_PATH={remote_glb}",
            "--file", str(NOTEBOOK),
        ]
        call_colab(exec_args, label="執行 Hunyuan3D-2 推論", timeout=timeout + 60, on_line=lambda l: print(f"   [Notebook] {l}"))
        print("🎉 遠端推論成功完成！")

        # 3. 下載產出模型
        print(f"\n📥 [3/4] 正在自 Colab 下載生成的 GLB 檔案至: {output}...")
        call_colab(["download", "--session", session, remote_glb, str(output)], label="下載 GLB 模型", timeout=600)

        if not output.is_file() or output.stat().st_size == 0:
            raise FileNotFoundError(f"下載的 GLB 模型檔案無效或為空: {output}")

        file_size_mb = output.stat().st_size / (1024 * 1024)
        print(f"✅ 模型下載完成！檔案大小: {file_size_mb:.2f} MB")
        return str(output)

    finally:
        if owns_session and not keep_session:
            print("\n🧹 [4/4] 正在自動關閉 Colab 執行階段以釋放資源與 Compute Units...")
            stop_session(session)


def main():
    parser = argparse.ArgumentParser(description="全自動 Colab 3D 模型生成 Runner")
    parser.add_argument("--front", required=True, help="正面視角圖路徑")
    parser.add_argument("--back", required=True, help="背面視角圖路徑")
    parser.add_argument("--side", required=True, help="側面視角圖路徑")
    parser.add_argument("--output", required=True, help="輸出 GLB 儲存路徑")
    parser.add_argument("--faces", type=int, default=15000, help="目標三角面數 (預設: 15000)")
    parser.add_argument("--gpu", default="L4", help="Colab GPU 類型 (預設: L4)")
    parser.add_argument("--session", default=None, help="重用既有 Colab 會話名稱")
    parser.add_argument("--keep-session", action="store_true", help="執行完畢後保持會話不關閉")
    parser.add_argument("--timeout", type=float, default=3600, help="推論逾時秒數 (預設: 3600)")

    args = parser.parse_args()

    try:
        saved = run_single_inference(
            front_path=args.front,
            back_path=args.back,
            side_path=args.side,
            output_path=args.output,
            target_faces=args.faces,
            gpu=args.gpu,
            session=args.session,
            keep_session=args.keep_session,
            timeout=args.timeout,
        )
        print(f"\n🎉 3D 模型已成功產生至: {saved}")
    except Exception as exc:
        print(f"\n❌ 生成失敗: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
