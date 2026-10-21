#!/usr/bin/env python3
"""
Colab Runner for 2D Multi-View to 3D GLB Pipeline.
Automates Colab session creation, file uploading, remote inference (including single-session warm-start batch execution),
and asset downloading using google-colab-cli.
"""

from __future__ import annotations

import argparse
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
from typing import Callable, Optional


SKILL_DIR = Path(__file__).resolve().parents[1]
TRELLIS_NOTEBOOK = SKILL_DIR / "assets" / "Trellis_Colab.ipynb"
HY3D_NOTEBOOK = SKILL_DIR / "assets" / "Hunyuan3D_2_Colab.ipynb"
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
    print(f"🚀 正在建立 Google Colab 會話 (Session: {session}, GPU: {gpu})...", flush=True)
    args = ["new", "--session", session, "--gpu", gpu]
    if high_mem:
        args.append("--high-mem")
    call_colab(args, label="建立 Colab 會話", timeout=900, on_line=lambda l: print(f"   [Colab] {l}", flush=True))
    print("✅ 會話建立完成！", flush=True)


def stop_session(session: str) -> None:
    print(f"🛑 正在釋放並停止 Colab 會話以節省額度: {session}...", flush=True)
    try:
        call_colab(["stop", "--session", session], label="停止 Colab 會話", timeout=300, on_line=lambda l: print(f"   [Colab] {l}", flush=True))
        print("✅ 會話已成功停止。", flush=True)
    except Exception as e:
        print(f"⚠️ 停止會話時出現警告: {e}", flush=True)


def run_single_inference(
    front_path: str,
    back_path: str,
    side_path: str,
    output_path: str,
    engine: str = "hunyuan",
    target_faces: int = 40000,
    gpu: str = "L4",
    high_mem: bool = True,
    session: Optional[str] = None,
    keep_session: bool = False,
    timeout: float = 3600,
    skip_workflow_upload: bool = False,
) -> str:
    """Run automated 2D-to-3D inference on Google Colab via CLI."""
    front = Path(front_path).resolve()
    back = Path(back_path).resolve()
    side = Path(side_path).resolve()
    output = Path(output_path).resolve()
    engine = engine.lower().strip()

    if not front.is_file():
        raise FileNotFoundError(f"正面圖片不存在: {front}")
    if not back.is_file():
        raise FileNotFoundError(f"背面圖片不存在: {back}")
    if not side.is_file():
        raise FileNotFoundError(f"側面圖片不存在: {side}")

    if engine == "trellis":
        notebook = TRELLIS_NOTEBOOK
        if not notebook.is_file():
            raise FileNotFoundError(f"找不到 TRELLIS 推論 Notebook: {notebook}")
    elif engine == "hunyuan":
        notebook = HY3D_NOTEBOOK
        if not notebook.is_file():
            raise FileNotFoundError(f"找不到 Hunyuan3D 推論 Notebook: {notebook}")
        if not WORKFLOW_JSON.is_file():
            raise FileNotFoundError(f"找不到工作流設定: {WORKFLOW_JSON}")
    else:
        raise ValueError(f"不支援的模型引擎: {engine} (僅支援 'trellis' 或 'hunyuan')")

    owns_session = session is None
    if session is None:
        prefix = "trellis" if engine == "trellis" else "hy3d"
        session = f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"

    output.parent.mkdir(parents=True, exist_ok=True)

    print("============================================================", flush=True)
    print("✨ 全自動 Colab 3D 模型生成管線啟動", flush=True)
    print(f"   引擎: {engine.upper()} | Session: {session} | GPU: {gpu}", flush=True)
    print(f"   正面: {front.name} | 背面: {back.name} | 側面: {side.name}", flush=True)
    if engine == "hunyuan":
        print(f"   目標三角面數: {target_faces}", flush=True)
    print(f"   目標導出位置: {output}", flush=True)
    print("============================================================", flush=True)

    try:
        if owns_session:
            start_session(session, gpu=gpu, high_mem=high_mem)

        remote_front = "/content/input_front.png"
        remote_back = "/content/input_back.png"
        remote_side = "/content/input_side.png"
        remote_glb = "/content/output_model.glb"

        print("\n📤 [1/4] 正在上傳三視圖至 Colab 遠端環境...", flush=True)
        call_colab(["upload", "--session", session, str(front), remote_front], label="上傳正面圖", timeout=300)
        call_colab(["upload", "--session", session, str(back), remote_back], label="上傳背面圖", timeout=300)
        call_colab(["upload", "--session", session, str(side), remote_side], label="上傳側面圖", timeout=300)

        if engine == "hunyuan" and not skip_workflow_upload:
            remote_workflow = "/content/workflow_api.json"
            call_colab(["upload", "--session", session, str(WORKFLOW_JSON), remote_workflow], label="上傳工作流", timeout=120)

        print("✅ 檔案上傳完畢！", flush=True)

        print(f"\n⚙️  [2/4] 正在 Colab 遠端 GPU 執行 {engine.upper()} 推論 Notebook...", flush=True)
        t_infer = time.monotonic()
        if engine == "trellis":
            exec_args = [
                "exec",
                "--session", session,
                "--timeout", str(int(timeout)),
                "--env", f"TRELLIS_FRONT_IMAGE={remote_front}",
                "--env", f"TRELLIS_BACK_IMAGE={remote_back}",
                "--env", f"TRELLIS_SIDE_IMAGE={remote_side}",
                "--env", f"TRELLIS_OUTPUT_PATH={remote_glb}",
                "--file", str(notebook),
            ]
            call_label = "執行 TRELLIS 推論"
        else:
            exec_args = [
                "exec",
                "--session", session,
                "--timeout", str(int(timeout)),
                "--env", f"HY3D_FRONT_IMAGE={remote_front}",
                "--env", f"HY3D_BACK_IMAGE={remote_back}",
                "--env", f"HY3D_SIDE_IMAGE={remote_side}",
                "--env", f"HY3D_TARGET_FACES={target_faces}",
                "--env", f"HY3D_OUTPUT_PATH={remote_glb}",
                "--file", str(notebook),
            ]
            call_label = "執行 Hunyuan3D-2 推論"

        call_colab(exec_args, label=call_label, timeout=timeout + 60, on_line=lambda l: print(f"   [Notebook] {l}", flush=True))
        infer_sec = time.monotonic() - t_infer
        print(f"🎉 遠端 {engine.upper()} 推論成功完成！(Notebook 耗時: {infer_sec:.1f}s)", flush=True)

        print(f"\n📥 [3/4] 正在自 Colab 下載生成的 GLB 檔案至: {output}...", flush=True)
        call_colab(["download", "--session", session, remote_glb, str(output)], label="下載 GLB 模型", timeout=600)

        if not output.is_file() or output.stat().st_size == 0:
            raise FileNotFoundError(f"下載的 GLB 模型檔案無效或為空: {output}")

        file_size_mb = output.stat().st_size / (1024 * 1024)
        print(f"✅ 模型下載完成！檔案大小: {file_size_mb:.2f} MB", flush=True)
        return str(output)

    finally:
        if owns_session and not keep_session:
            print("\n🧹 [4/4] 正在自動關閉 Colab 執行階段以釋放資源與 Compute Units...", flush=True)
            stop_session(session)


def run_batch_prefixes(
    prefixes: list[str],
    inputs_dir: str = "assets/inputs",
    output_dir: str = "godot_project/assets/models",
    engine: str = "hunyuan",
    target_faces: int = 40000,
    gpu: str = "L4",
    session: Optional[str] = None,
    keep_session: bool = False,
    timeout: float = 3600,
    auto_bake: bool = False,
    style: str = "popmart",
) -> list[str]:
    """
    在單一 Colab 會話中執行多模型「1 次冷啟動 + N-1 次熱啟動」批次推論，
    測試並驗證熱啟動加速效果，最後自動關閉會話以節省 CCU。
    """
    owns_session = session is None
    if session is None:
        session = f"hy3d-batch-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"

    in_base = Path(inputs_dir).resolve()
    out_base = Path(output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    results: list[str] = []
    timings: list[tuple[str, float, str]] = []

    print("============================================================", flush=True)
    print(f"🔥 啟動單一 Colab Session 熱啟動批次生成佇列 (共 {len(prefixes)} 個模型)", flush=True)
    print(f"   Session: {session} | GPU: {gpu} | 目標面數: {target_faces} | 風格: {style}", flush=True)
    print(f"   佇列名單: {', '.join(prefixes)}", flush=True)
    print("============================================================", flush=True)

    t_batch_start = time.monotonic()
    try:
        if owns_session:
            start_session(session, gpu=gpu, high_mem=True)

        for idx, pfx in enumerate(prefixes, 1):
            t_item = time.monotonic()
            mode_str = "冷啟動 (Cold Start - 含安裝與權重載入)" if idx == 1 else "🔥 熱啟動 (Warm Start - VRAM 已常駐)"
            print(f"\n>>> [{idx}/{len(prefixes)}] 開始處理角色: {pfx} [{mode_str}] <<<", flush=True)

            front_p = in_base / f"{pfx}_front.png"
            back_p = in_base / f"{pfx}_back.png"
            side_p = in_base / f"{pfx}_side.png"
            out_p = out_base / f"{pfx}.orig.glb"

            saved = run_single_inference(
                front_path=str(front_p),
                back_path=str(back_p),
                side_path=str(side_p),
                output_path=str(out_p),
                engine=engine,
                target_faces=target_faces,
                gpu=gpu,
                session=session,
                keep_session=True,
                timeout=timeout,
                skip_workflow_upload=(idx > 1),
            )
            dt = time.monotonic() - t_item
            results.append(saved)
            timings.append((pfx, dt, mode_str))
            print(f"✅ [{idx}/{len(prefixes)}] {pfx} 幾何產生完成！單項幾何耗時: {dt:.1f} 秒", flush=True)

            if auto_bake:
                bake_script = Path(__file__).resolve().parent / "bake_multiview_textures.py"
                final_glb = out_base / f"{pfx}.glb"
                print(f"🎨 自動呼叫本機 Blender 進行多視角貼圖烘焙 (風格: {style})...", flush=True)
                bake_cmd = [
                    sys.executable,
                    str(bake_script),
                    "--mesh", str(out_p),
                    "--front", str(front_p),
                    "--back", str(back_p),
                    "--side", str(side_p),
                    "-o", str(final_glb),
                    "--style", style,
                ]
                subprocess.run(bake_cmd, check=True)

        total_sec = time.monotonic() - t_batch_start
        est_ccu = (total_sec / 3600.0) * 4.8
        print("\n============================================================", flush=True)
        print("📊 Colab L4 批次熱啟動推論效能統計報告", flush=True)
        print("============================================================", flush=True)
        for pfx, dt, m_str in timings:
            print(f"   - {pfx:20s} : {dt:6.1f} 秒 ({m_str})", flush=True)
        print("------------------------------------------------------------", flush=True)
        print(f"   總計耗時: {total_sec:.1f} 秒 ({total_sec/60.0:.2f} 分鐘) | 預估消耗 CCU: ~{est_ccu:.2f} CCU", flush=True)
        print("============================================================", flush=True)
        return results

    finally:
        if owns_session and not keep_session:
            print("\n🧹 批次任務全數完成，正在關閉 Colab 會話以釋放資源...", flush=True)
            stop_session(session)


def main():
    parser = argparse.ArgumentParser(description="全自動 Colab 3D 模型生成 Runner (支援單一與熱啟動批次模式)")
    parser.add_argument("--front", help="正面視角圖路徑")
    parser.add_argument("--back", help="背面視角圖路徑")
    parser.add_argument("--side", help="側面視角圖路徑")
    parser.add_argument("--output", help="輸出 GLB 儲存路徑")
    parser.add_argument("--batch-prefixes", nargs="+", help="批次執行的角色前綴清單 (例如: hedgehog_new penguin_new salamander_new electric_eel_new)")
    parser.add_argument("--inputs-dir", default="assets/inputs", help="批次輸入圖所在目錄 (預設: assets/inputs)")
    parser.add_argument("--output-dir", default="godot_project/assets/models", help="批次輸出模型目錄 (預設: godot_project/assets/models)")
    parser.add_argument("--engine", choices=["trellis", "hunyuan"], default="hunyuan", help="3D 生成引擎 (預設: hunyuan)")
    parser.add_argument("--faces", type=int, default=40000, help="目標三角面數 (Hunyuan3D 使用，預設: 40000)")
    parser.add_argument("--gpu", default="L4", help="Colab GPU 類型 (預設: L4)")
    parser.add_argument("--session", default=None, help="重用既有 Colab 會話名稱")
    parser.add_argument("--keep-session", action="store_true", help="執行完畢後保持會話不關閉")
    parser.add_argument("--timeout", type=float, default=3600, help="推論逾時秒數 (預設: 3600)")
    parser.add_argument("--style", "-s", default="popmart", help="材質風格 (預設: popmart，可選: popmart, anime, ceramic, handpainted, mecha, clay)")
    parser.add_argument("--auto-bake", action="store_true", help="推論完成後自動呼叫本機 Blender 執行遮擋感知 PBR 貼圖烘焙")

    args = parser.parse_args()

    try:
        if args.batch_prefixes:
            saved_list = run_batch_prefixes(
                prefixes=args.batch_prefixes,
                inputs_dir=args.inputs_dir,
                output_dir=args.output_dir,
                engine=args.engine,
                target_faces=args.faces,
                gpu=args.gpu,
                session=args.session,
                keep_session=args.keep_session,
                timeout=args.timeout,
                auto_bake=args.auto_bake,
                style=args.style,
            )
            print(f"\n🎉 批次 {len(saved_list)} 個 3D 模型已全數成功產生！")
        else:
            if not (args.front and args.back and args.side and args.output):
                parser.error("單一模式下必須提供 --front, --back, --side 與 --output，或使用 --batch-prefixes 啟動批次模式。")
            saved = run_single_inference(
                front_path=args.front,
                back_path=args.back,
                side_path=args.side,
                output_path=args.output,
                engine=args.engine,
                target_faces=args.faces,
                gpu=args.gpu,
                session=args.session,
                keep_session=args.keep_session,
                timeout=args.timeout,
            )
            print(f"\n🎉 3D 模型已成功產生至: {saved}")

            if args.auto_bake:
                bake_script = Path(__file__).resolve().parent / "bake_multiview_textures.py"
                out_p = Path(saved).resolve()
                final_glb = out_p.with_suffix(".glb") if out_p.name.endswith(".orig.glb") else out_p
                orig_mesh = out_p
                print(f"🎨 自動呼叫本機 Blender 進行多視角貼圖烘焙 (風格: {args.style})...", flush=True)
                bake_cmd = [
                    sys.executable,
                    str(bake_script),
                    "--mesh", str(orig_mesh),
                    "--front", str(Path(args.front).resolve()),
                    "--back", str(Path(args.back).resolve()),
                    "--side", str(Path(args.side).resolve()),
                    "-o", str(final_glb),
                    "--style", args.style,
                ]
                subprocess.run(bake_cmd, check=True)
    except Exception as exc:
        print(f"\n❌ 生成失敗: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
