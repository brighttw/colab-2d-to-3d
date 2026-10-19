# 全自動三視圖生 3D 模型管線 (Colab 2D-to-3D for Godot 4)

本專案是一套專為 **Godot 4** 遊戲引擎打造的商業盲盒公仔級 2D 轉 3D 自動化資產生成管線。透過雲端 **Hunyuan3D-2 MV (ComfyUI Headless API)** 高解析度體素解碼結合本機 **Blender 自遮擋感知多視角 PBR 烘焙 (Occlusion-Aware Harmonic PBR Baker)**，將正、側、背三視圖 2D 概念圖全自動轉換為具備 40,000 面精細幾何、無接縫無條紋 2K PBR 材質並相容 Godot 4 規範的 `.glb` 模型資產。

本專案原生封裝為 **AI Agent Skill**，支援在 Google Antigravity 或相容 AI Agent 環境中透過自然語言一鍵驅動；同時支援 **單一 Colab Session 熱啟動批次佇列 (`--batch-prefixes`)**，實測 4 個 40,000 面模型總耗時僅需約 10.5 分鐘（消耗約 `0.84 CCU`）。

```text
[本機 2D 前處理: preprocess_views.py]
三視圖總表 (JPG/PNG) ──> 外圍白底 Flood-Fill 去背 (100% 保留白肚皮/眼白)
                     ──> 2D 連通元件 (Connected-Component) 無損切分正/側/背並對齊 1024x1024
                                │
                                ▼ (HTTP Upload)
[Google Colab L4 GPU: runner.py + Hunyuan3D_2_Colab.ipynb]
ComfyUI 常駐熱啟動服務 ──> Exterior Flood-Fill 精準 Alpha 提取 (繞過 u2net 誤挖白肚皮)
                       ──> Hunyuan3D-2 MV 多視角擴散 (50 steps, CFG 6.0)
                       ──> 高解析度八叉樹體素解碼 (octree_resolution=384, 40,000 面幾何導出)
                                │
                                ▼ (HTTP Download .orig.glb)
[本機 PBR 烘焙與驗收: bake_multiview_textures.py + render_turntable.py]
Blender Headless (CPU) ──> 自動 Smart UV 展開 + 3D BVHTree 射線自遮擋剔除
                       ──> 側腹下垂手部陰影抑制 + 36 步 Mesh Graph Laplacian 諧波色彩補洞
                       ──> Cycles 2048x2048 AO 烘焙 + 柔光盲盒公仔 PBR 材質導出 (.glb)
                       ──> Godot 4 自動重新匯入 + 四視角/360° 旋轉影片 (.mp4) 自動驗收
```

---

## 專案結構

```text
.
├── .agents/skills/colab-2d-to-3d/ # 【核心】AI Agent 技能包 (遵循 Antigravity 規格)
│   ├── SKILL.md                  # Skill 規格、觸發條件、多視圖生圖 Prompt 模板與 SOP
│   └── references/
│       ├── pipeline_architecture.md # 雲端幾何 + 本機自遮擋 PBR 烘焙混合管線架構
│       └── godot_guidelines.md   # Godot 4 碰撞體與 PBR 材質整合指南
├── install_skill.sh              # 一鍵安裝 Skill 至全域環境 (~/.gemini/config/skills/)
├── run_colab_inference.sh        # 一鍵全自動 Colab CLI 啟動器 (免開網頁)
├── scripts/
│   ├── preprocess_views.py       # 外圍白底 Flood-Fill 去背、2D 連通元件三視圖切分與 1024x1024 對齊
│   ├── runner.py                 # 管理 Colab 會話、單一 Session 熱啟動批次佇列 (--batch-prefixes) 與下載
│   ├── bake_multiview_textures.py # 【核心】Blender 3D BVHTree 自遮擋剔除、拉普拉斯諧波補洞與 Cycles AO/PBR 烘焙
│   ├── render_turntable.py       # Blender Headless 四視角對比圖 (帶貼圖/白模) 與 360° 旋轉 MP4 驗收腳本
│   └── optimize_glb.py           # 備用網格平滑與 Godot 場景輔助工具
├── assets/
│   ├── Hunyuan3D_2_Colab.ipynb   # 遠端無人值守推論 Notebook (內建 DINOv2 與 Flood-Fill 去背補丁)
│   ├── Trellis_Colab.ipynb       # Microsoft TRELLIS 備用推論 Notebook
│   └── inputs/                   # 存放切分後的三視圖輸入圖片 ({prefix}_{front,side,back}.png)
├── ref/                          # 參考素材與原始三視圖總表目錄 (受 .gitignore 保護)
├── workflow_api.json             # ComfyUI Headless API 高解析度幾何推論工作流 (384 Octree / 40K Faces)
├── agent_3d_skill.py             # 本地端 / Direct API 調用腳本
├── requirements.txt              # 本地端 Python 依賴清單
└── godot_project/                # Godot 4 遊戲專案目錄
    ├── project.godot             # Godot 4 專案設定檔
    ├── main.tscn                 # 3D 預覽展示主場景
    └── assets/
        └── models/               # 產出的 .glb 模型與 _diffuse.png 貼圖儲存位置
```

---

## 一、AI Agent Skill 使用方式 (推薦)

本專案已封裝為標準 **Antigravity AI Agent Skill**。若您正在使用 Google Antigravity 或相容的 AI Agent，Agent 會自動加載 [`.agents/skills/colab-2d-to-3d/SKILL.md`](file:///.agents/skills/colab-2d-to-3d/SKILL.md) 並執行完整四階段管線。

### 1. 全域安裝 (跨任何專案使用)
```bash
# 軟連結至全域 ~/.gemini/config/skills/ (推薦，隨本專案更新自動同步)
./install_skill.sh --symlink
```

### 2. 自然語言 Prompt 指令範例
> 💬 **「幫我把 `ref/new/` 裡面的四張角色三視圖切分後，用 Colab L4 熱啟動批次生成 40,000 面的 3D 模型，完成本機 PBR 貼圖烘焙並渲染四視角與 360 度旋轉影片驗收。」**

---

## 二、端到端四階段標準工作流 (CLI)

### 階段 1：2D 三視圖總表無損切分與白肚皮保護去背 (`preprocess_views.py`)

針對單張包含「正面（左）、側面（中）、背面（右）」的三視圖總表，執行外圍白底 Flood-Fill 去背（100% 保留企鵝白肚皮、刺蝟胸口與眼白高光）與 2D 連通元件獨立提取（避免長尾巴與手部水平座標微幅重疊時被截斷），並統一正規化為 `1024x1024` 畫布：

```bash
python3 scripts/preprocess_views.py \
  --sheet ref/new/Hedgehog_character_new.jpg \
  --out-dir assets/inputs \
  --prefix hedgehog_new
```

---

### 階段 2：雲端 Colab L4 幾何生成（支援單一 Session 熱啟動批次佇列）

#### A. 單一 Session 熱啟動批次生成多個角色（強烈推薦・極省 CCU）
只需啟動一次 Colab L4 VM、載入一次 4.9GB 模型權重至 VRAM，即可在同一個常駐 ComfyUI 程序中連續高速生成多個角色（實測第 2 個角色起單模 GPU 推論僅需 **78～87 秒**）：

```bash
python3 scripts/runner.py \
  --batch-prefixes hedgehog_new penguin_new salamander_new electric_eel_new \
  --engine hunyuan \
  --faces 40000 \
  --gpu L4
```
執行完畢後會自動於 `godot_project/assets/models/` 下載各角色的原始高模 `<prefix>.orig.glb`，並自動停止 Colab Session 釋放算力。

#### B. 單一角色一鍵推論（含自動調用本機 PBR 烘焙）
```bash
./run_colab_inference.sh \
  --front assets/inputs/hedgehog_new_front.png \
  --side assets/inputs/hedgehog_new_side.png \
  --back assets/inputs/hedgehog_new_back.png \
  --output godot_project/assets/models/hedgehog_new.glb \
  --engine hunyuan \
  --faces 40000 \
  --gpu L4
```

---

### 階段 3：本機 3D 自遮擋感知與拉普拉斯諧波 PBR 貼圖烘焙 (`bake_multiview_textures.py`)

針對雲端下載的 `.orig.glb` 高密網格，使用本機 Blender Headless (CPU) 執行：
1. **自動 Smart UV Unwrap**：為無 UV 原始網格自動展開無重疊 UV 島。
2. **3D `BVHTree` 光線投射自遮擋剔除**：剔除 A-pose 手臂對側腹、背刺對刺溝的視線遮擋重影。
3. **側腹下垂手部陰影抑制 + 36 步 Mesh Graph Laplacian 諧波補洞**：徹底消除傳統三視角投影在側面產生的「斑馬條紋」與側視圖下垂手臂／黑翅在肚皮側邊留下的假印記。
4. **Cycles 2048×2048 幾何 AO 與柔光盲盒公仔 PBR 材質**：自動合成 `Roughness = 0.68`、`Specular IOR Level = 0.22` 的柔光搪膠質感並導出最終 `.glb`。

```bash
python3 scripts/bake_multiview_textures.py \
  --mesh godot_project/assets/models/hedgehog_new.orig.glb \
  --front assets/inputs/hedgehog_new_front.png \
  --side assets/inputs/hedgehog_new_side.png \
  --back assets/inputs/hedgehog_new_back.png \
  -o godot_project/assets/models/hedgehog_new.glb \
  --texture-size 2048
```

---

### 階段 4：四視角檢視圖與 360° 旋轉影片自動驗收 (`render_turntable.py`)

使用 Blender Cycles (CPU) 渲染正、斜、側、背四視角拼接對比圖（含帶貼圖 PBR 與純幾何白模 Clay）以及 360° 旋轉展示影片（`.mp4`）：

```bash
python3 scripts/render_turntable.py \
  --glb godot_project/assets/models/hedgehog_new.glb \
  --out-dir ./renders \
  --prefix hedgehog_new \
  --turntable
```

---

## 三、Colab L4 實測效能與 Google AI Pro (500 CCU) 產能基準

以下為單一 Colab L4 Session 連續生成 4 個 40,000 面模型之實測數據：

| 順序 | 角色代號 | 啟動狀態 | 遠端 GPU 推論耗時 | 單項總耗時 (含傳輸) | 三角面數 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[1/4]** | `hedgehog_new` (刺蝟) | 冷啟動 (含安裝與 4.9GB 權重載入) | ~90 秒 | **299.8 秒** | 40,000 |
| **[2/4]** | `penguin_new` (企鵝) | 🔥 熱啟動 (VRAM 常駐) | **87.10 秒** | **93.0 秒** | 40,000 |
| **[3/4]** | `salamander_new` (火蜥蜴) | 🔥 熱啟動 (VRAM 常駐) | **84.65 秒** | **147.5 秒** | 39,998 |
| **[4/4]** | `electric_eel_new` (電鰻) | 🔥 熱啟動 (VRAM 常駐) | **78.10 秒** | **83.2 秒** | 39,992 |
| **合計** | **4 個角色** | **1 冷啟動 + 3 熱啟動** | **339.8 秒** | **627.9 秒 (10.46 分鐘)** | **預估消耗 `~0.84 CCU`** |

- **單模平均成本**：熱啟動批次下平均每顆 40,000 面模型僅消耗約 **`0.21 CCU`**，Google AI Pro 每月 500 CCU 額度可穩定支撐超過 **2,000 次** 高品質模型生成。

---

## 四、雲端與本機黃金參數配置表

| 階段與元件 | 參數名稱 | 黃金設定值 | 技術原理與防踩坑說明 |
| :--- | :--- | :--- | :--- |
| **雲端 Node 13~15**<br>`ImageRemoveBackground+` | `Exterior Flood-Fill Patch` | `border_tol=245`<br>`gaussian_sigma=0.8` | 取代預設 `u2net` 去背，防止企鵝白肚皮或刺蝟胸口被挖穿成透明破洞。 |
| **雲端 Node 30**<br>`Hy3DGenerateMeshMultiView` | `steps`<br>`guidance_scale` | `50`<br>`6.0` | 深度收斂正、側、背三視角幾何細節，刻畫出獨立立體刺簇、手指與外鰓。 |
| **雲端 Node 31**<br>`Hy3DVAEDecode` | `octree_resolution`<br>`num_chunks` | `384`<br>`16000` | 將八叉樹體素解析度由 `256` 提升至 `384`（體素密度提升 3.375 倍），徹底消除圓臉與肚皮的階梯波紋。 |
| **雲端 Node 32**<br>`Hy3DPostprocessMesh` | `max_facenum`<br>`smooth_normals` | `40000`<br>`true` | 保留 40,000 面高密度遊戲網格並直接由 Node 40 匯出純幾何，避開雲端單視角 Paint 節點的側面條紋缺陷。 |
| **本機 PBR 烘焙**<br>`bake_multiview_textures.py` | `BVHTree` + `HarmonicInpaint`<br>`Roughness` / `Specular IOR` | `36 iterations`<br>`0.68` / `0.22` | 以 3D 射線遮擋剔除與網格拓撲拉普拉斯諧波擴散填補死角，搭配柔光搪膠 PBR 材質消除塑膠鏡面亮斑。 |

---

## 五、開放原始碼與安全性說明

- **無任何金鑰外洩**：專案不包含任何硬編碼的帳號、密碼、API Key 或個人主機絕對路徑。
- **Git 倉庫輕量化**：已透過 `.gitignore` 排除大體積 `.glb` 模型、貼圖暫存檔、Colab 執行紀錄（`*_output.ipynb`）與參考素材。
