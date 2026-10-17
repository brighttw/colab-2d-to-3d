# 全自動三視圖生 3D 模型管線 (Colab 2D-to-3D for Godot 4)

本專案是一套專為 **Godot 4** 遊戲引擎打造的 2D 轉 3D 自動化資產生成管線。透過 **Hunyuan3D-2 MV** 配合 **ComfyUI Headless API**，將正、背、側三視圖 2D 概念圖全自動轉換為具備 PBR 材質、已完成拓撲網格減面（Decimation）並相容 Godot 4 規範的 `.glb` 模型資產。

本專案原生封裝為 **AI Agent Skill**，支援在 Google Antigravity 或相容 AI Agent 環境中透過自然語言一鍵驅動；同時亦提供獨立的命令列 CLI 工具與 Python 腳本。

```
[本機 / AI Agent]                             [Google Colab (L4 GPU)]
三視圖 (PNG) ----------(HTTP Upload)---------> ComfyUI /upload/image
                                                       │
                                            [RMBG 去背 + 518x518 正規化]
                                                       │
                                            [Hunyuan3D-2 MV 幾何生成]
                                                       │
                                            [Watertight 解碼 + 網格減面]
                                                       │
                                            [xatlas UV + 多視角 PBR 烘焙]
                                                       │
.glb 檔案   <--------(HTTP Download)-------  [Hy3DExportMesh 導出]
    │
[Godot 4 專案]
(自動 Headless 重新導入 + 碰撞體/材質映射)
```

---

## 專案結構

```text
.
├── .agents/skills/colab-2d-to-3d/ # 【核心】AI Agent 技能包 (遵循 Antigravity 規格)
│   ├── SKILL.md                  # Skill 規格、觸發條件與 Agent 執行指南
│   └── references/
│       ├── pipeline_architecture.md # ComfyUI 節點管線與 Hunyuan3D-2 MV 架構
│       └── godot_guidelines.md   # Godot 4 碰撞體與 PBR 材質整合指南
├── install_skill.sh              # 一鍵安裝 Skill 至全域環境 (~/.gemini/config/skills/)
├── run_colab_inference.sh        # 【推薦】一鍵全自動 Colab CLI 啟動器 (免開網頁)
├── scripts/
│   ├── runner.py                 # 管理 Colab 會話、上傳、遠端推論與模型下載
│   ├── preprocess_views.py       # 2D 三視圖切分、去水平灰線/腳底灰影、側面鼻吻對齊與工作流調優
│   └── optimize_glb.py           # 3D GLB 孤島 UV 縫合、跨接縫安全細分、Taubin 平滑與貼圖後處理
├── assets/
│   ├── Hunyuan3D_2_Colab.ipynb   # 遠端無人值守推論 Notebook
│   └── inputs/                   # 存放三視圖輸入圖片 (front, back, side)
├── ref/                          # 參考素材目錄 (受 .gitignore 保護)
├── setup_colab.py                # (備用) 網頁手動部署與 Cloudflare 隧道腳本
├── workflow_api.json             # ComfyUI Headless API 參數化推論工作流 (JSON)
├── agent_3d_skill.py             # 本地端 / Direct API 調用腳本
├── requirements.txt              # 本地端 Python 依賴清單
└── godot_project/                # Godot 4 遊戲專案目錄
    ├── project.godot             # Godot 4 專案設定檔
    ├── main.tscn                 # 3D 預覽展示主場景
    └── assets/
        └── models/               # 產出的 .glb 模型儲存位置
```

---

## 一、AI Agent Skill 使用方式 (推薦)

本專案已封裝為標準 **Antigravity AI Agent Skill**。若您正在使用 Google Antigravity 或相容的 AI Agent，**完全不需要記憶複雜的參數或指令**，Agent 會根據您的需求自動加載技能並執行完整流程。

### 1. 本地工作區自動加載 (Workspace Skill)
當您在當前專案資料夾開啟 Antigravity 時，Agent 會自動探測並加載 `.agents/skills/colab-2d-to-3d/`。

### 2. 全域安裝 (跨任何專案使用)
若希望在電腦上的其他遊戲專案中也能直接請 Agent 調用這套 3D 生成管線，可執行全域安裝腳本：

```bash
# 軟連結至全域 ~/.gemini/config/skills/ (推薦，隨本專案更新自動同步)
./install_skill.sh --symlink

# 或直接複製安裝至全域
./install_skill.sh --global
```

### 3. 自然語言 Prompt 指令範例

您可以直接對 AI Agent 下達如下指示：

> 💬 **「幫我將 assets/inputs/ 裡面的三視圖轉換成 Godot 4 的 3D 模型，存到 godot_project/assets/models/character.glb」**

> 💬 **「我準備了一組主角角色的三視圖，側面圖是右側視角，請以 20,000 面在 Colab L4 GPU 上生成模型，完成後自動更新 Godot 專案資產。」**

> 💬 **「幫我將這組道具圖做成低面數（約 4,000 面）的 3D 模型並設定好碰撞體。」**

### 4. Agent 自動執行的幕後工作流
當 Agent 收到您的請求時，它會依據 [`SKILL.md`](file:///.agents/skills/colab-2d-to-3d/SKILL.md) 執行標準作業程序（SOP）：
1. **環境檢查**：自動確認 `colab --auth=oauth2 auth status` 是否已登入，以及 Python 依賴環境。
2. **2D 三視圖前處理**：透過 `scripts/preprocess_views.py` 自動切分總表、消除背景水平對齊線與地板接觸陰影、對齊正側面口鼻比例與頭尾高度。
3. **推論執行**：自動呼叫 `./run_colab_inference.sh` 建立遠端 Colab L4 GPU 運算執行個體並下載 `.glb`。
4. **3D 後處理與 Godot 整合**：透過 `scripts/optimize_glb.py` 自動縫合孤島 UV、修復 `xatlas` 法線斷裂、執行 Taubin 體積保留平滑、增強眼神高光與腮紅，並以 headless 模式觸發 Godot 4 重新導入資產。

---

## 二、命令列 (CLI) 與端到端三步曲使用方式

若不透過 AI Agent，您也可以直接在終端機中執行標準三階段管線：

### 步驟 1：2D 三視圖前處理與工作流調優（選用但強烈推薦）

針對單張三視圖總表（或獨立三張圖），自動切分、清除背景水平輔助線與腳底陰影、液化回縮側面過度突出的口鼻，並自動寫入 L4 黃金工作流參數：

```bash
python3 scripts/preprocess_views.py \
  --sheet ref/Cute_hedgehog_standing_pose.jpg \
  --out-dir assets/inputs \
  --prefix hedgehog \
  --side-snout-pull 0.26 \
  --tune-workflow workflow_api.json
```

### 步驟 2：一鍵遠端 Colab CLI 推論（免開瀏覽器・推薦）

透過 Google Colab CLI 自動建立遠端會話、上傳圖片、執行推論並取回模型：

```bash
./run_colab_inference.sh \
  --front assets/inputs/hedgehog_front.png \
  --back assets/inputs/hedgehog_back.png \
  --side assets/inputs/hedgehog_side.png \
  --output godot_project/assets/models/hedgehog.glb \
  --faces 20000 \
  --gpu L4
```

#### 常用參數：
- `--front, -f`: 正面圖路徑
- `--back, -b`: 背面圖路徑
- `--side, -s`: 側面圖路徑
- `--output, -o`: 輸出 `.glb` 路徑
- `--faces`: 目標三角面數（預設 `15000`，針對具備毛刺/複雜頭髮的主角建議設為 `20000`）
- `--gpu`: GPU 型號，預設 `L4`（24GB VRAM），亦支援 `A100`
- `--keep-session`: 推論完成後保持會話運行（避免重複開機，適合批次生成）

### 步驟 3：3D 模型幾何/貼圖後處理與 Godot 4 場景更新

修復 `xatlas` 展開造成的接縫法線斷裂、縫合孤島破面、執行跨接縫安全細分與 Taubin 曲面平滑，並自動更新 Godot 4 柔光攝影棚場景與重新匯入：

```bash
python3 scripts/optimize_glb.py godot_project/assets/models/hedgehog.glb \
  --snout-compact 0.14 \
  --update-godot-scene
```

---

### 進階模式：直連已運行的 ComfyUI 端點 (Direct API)

若您已在 Colab 或本機啟動了 ComfyUI（例如透過 `setup_colab.py` 產生的 Cloudflare 隧道網址）：

```bash
export COMFY_ENDPOINT="https://xxxx-xxxx.trycloudflare.com"

python3 agent_3d_skill.py \
  assets/inputs/hedgehog_front.png \
  assets/inputs/hedgehog_back.png \
  assets/inputs/hedgehog_side.png \
  godot_project/assets/models/hedgehog.glb \
  20000 \
  "$COMFY_ENDPOINT" \
  --side-orientation left
```

亦可作為 Python 模組直接導入：

```python
from agent_3d_skill import generate_godot_model

glb_path = generate_godot_model(
    front_img="assets/inputs/hedgehog_front.png",
    back_img="assets/inputs/hedgehog_back.png",
    side_img="assets/inputs/hedgehog_side.png",
    output_glb_path="godot_project/assets/models/hedgehog.glb",
    target_faces=20000,
    side_orientation="left",
    endpoint="https://xxxx-xxxx.trycloudflare.com"
)
print(f"模型產出成功: {glb_path}")
```

---

## 三、Google Colab L4 GPU (24GB VRAM) 黃金穩定參數配置

本專案之 `workflow_api.json` 已預設針對 **Google Colab L4 GPU (22.3 GB 可用顯存)** 完成實戰調優。以下為兼顧「五官幾何飽滿度」、「多視角貼圖細膩度」與「零 OOM 穩定完跑」的黃金參數表：

| 節點 ID 與名稱 | 參數名稱 | L4 黃金設定值 | 調優原理與防踩坑說明 |
| :--- | :--- | :--- | :--- |
| **Node 30**<br>`Hy3DGenerateMeshMultiView` | `steps`<br>`guidance_scale` | `30`<br>`5.5` | 穩定融合正、側、背三視角隱空間特徵，避免過高 CFG 造成表面高頻突起噪聲。 |
| **Node 31**<br>`Hy3DVAEDecode` | `octree_resolution`<br>`num_chunks` | `256`<br>`8000` | **關鍵參數**：設為 `256` 可適度抑制背刺/髮絲等極高頻碎刺體素，防止後續減面與 `xatlas` 時背刺搶佔 >95% 的頂點與 UV 面積；若盲目調高至 `320+` 反而在減面時壓縮臉部配額並增加 OOM 風險。 |
| **Node 32**<br>`Hy3DPostprocessMesh` | `max_facenum`<br>`remove_floaters`<br>`smooth_normals` | `20000` (主角建議)<br>`true`<br>`true` | 針對有毛刺、配件的角色，建議透過 CLI 傳入 `--faces 20000`，確保臉頰與五官分到足夠的三角面預算。 |
| **Node 35**<br>`Hy3DRenderMultiView` | `render_size`<br>`texture_size` | `1024`<br>`2048` | **防崩潰鐵律**：`render_size` (`1024`) **必須與 Node 36 的 `view_size` (`1024`) 完全一致**，否則多視角法線圖與位置圖張量維度不匹配會直接導致 `Hy3DSampleMultiView` 報錯或顯存爆滿；最終匯出貼圖則維持 `2048×2048` 高解析度。 |
| **Node 36**<br>`Hy3DSampleMultiView` | `view_size`<br>`steps`<br>`guidance_scale` | `1024`<br>`35`<br>`5.5` | 將多視角繪製採樣步數提升至 `35`、`guidance_scale` 設為 `5.5`，能顯著還原 2D 原畫中的柔和雙頰腮紅與毛色漸層。 |
| **Node 38**<br>`CV2InpaintTexture` | `inpaint_radius`<br>`inpaint_method` | `5`<br>`"ns"` | 將 Navier-Stokes (`ns`) 接縫修補半徑由 `3` 提升至 `5`，有效消除 UV 島邊界處的黑色背景滲色（Seam Bleeding）。 |

### 面數預算與應用場景對照表

| 物件類型 | 建議目標面數 (`--faces`) | 貼圖解析度 | 適用情境 |
| :--- | :--- | :--- | :--- |
| **小道具 / 掉落物 (Props)** | 3,000 ～ 5,000 | 1024×1024 | 藥水瓶、金幣寶箱、小型武器 |
| **場景物件 / 載具 (Environment)** | 8,000 ～ 12,000 | 2048×2048 | 建築組件、載具、大型障礙物 |
| **主要角色 / 帶毛刺角色 (Hero Assets)** | 15,000 ～ 20,000 | 2048×2048 | 玩家操控角色、具高頻毛刺或複雜髮型之主角 |

---

## 四、Godot 4 專案導入與驗證

模型下載至 `godot_project/assets/models/` 後：

### 1. 刷新專案資產快取 (Headless)
```bash
godot --headless --path ./godot_project --import --quit
```
*(若使用 `python3 scripts/optimize_glb.py ... --update-godot-scene` 則會自動完成場景三點柔光配置與 Headless 重新匯入)*

### 2. 物理碰撞體設定建議
- **動態角色 (`CharacterBody3D`)**：建議在 Godot 選取網格後使用 **Mesh -> Create Single Convex Collision Sibling**（凸包碰撞體，運算效能最佳且不易穿模）。
- **靜態場景 (`StaticBody3D`)**：可使用 **Mesh -> Create Trimesh Static Body**（精確三角形網格碰撞）。

### 3. 注意事項
專案內的展示場景 `godot_project/main.tscn` 預設引用 `res://assets/models/hedgehog.glb`。若初次 clone 專案尚未執行生成流程，開啟該場景時請先執行一次推論產出模型，或於 Godot 編輯器中掛載您產出的任意 `.glb` 檔案。

---

## 五、常見問題與排錯指南 (Troubleshooting)

| 異常現象 | 可能原因 | 排除步驟 |
| :--- | :--- | :--- |
| **Colab 配額不足或超時** | 免費額度耗盡或資源緊張 | 1. 檢查 `colab --auth=oauth2 auth status` 是否正確。<br>2. 嘗試切換為 `--gpu A100` 或至 Colab 頁面確認運算單元 (Compute Units)。 |
| **20K 面推論在 UV 展開階段發生 `ReadTimeout`** | `xatlas` 在 C++ 層展開高面數複雜網格時會短暫阻塞 Python GIL (約 90~180 秒) | `assets/Hunyuan3D_2_Colab.ipynb` 的輪詢機制已內建 `timeout=60` 與 `RequestException` 自動重試容錯，請勿手動縮短輪詢超時。 |
| **嘴下出現凹坑或正側面五官撕裂** | 正面圖與側面圖的口鼻突出長度或頭尾高度不一致 | 先執行 `scripts/preprocess_views.py --side-snout-pull 0.26` 將側面長鼻吻向內液化回縮並對齊三視圖基準線，再進行雲端推論。 |
| **模型表面有塊狀多邊形黑影與破洞** | `xatlas` 切割 UV 島時斷開了空間頂點法線，並產生零碎孤島面 | 推論完成後執行 `python3 scripts/optimize_glb.py <glb路徑> --update-godot-scene`，自動完成孤島 UV 縫合、空間頂點焊接與 Taubin 平滑。 |
| **生成的模型背面變形** | 側面圖方向與演算法規範相反 | Hunyuan3D-2 MV 預設期望為左側面視角。若輸入為右側視角，請指定 `--side-orientation right` 自動水平鏡像。 |

---

## 六、開放原始碼與安全性說明

本專案適合直接開源託管於公開 GitHub 倉庫：
- **無任何金鑰外洩**：專案不包含任何硬編碼的帳號、密碼、API Key 或個人主機絕對路徑。
- **Git 倉庫輕量化**：已透過 `.gitignore` 排除大體積 `.glb` 模型、貼圖暫存檔、Colab 執行紀錄（`*_output.ipynb`）與參考素材。
