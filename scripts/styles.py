#!/usr/bin/env python3
"""
Style Presets & 2D Prompt Generator for 2D-to-3D Pipeline (`scripts/styles.py`)

提供全流程 6 大風格預設（2D 參考圖提示詞樣板、Blender 貼圖烘焙 PBR 參數、AO 混合曲線與別名解析）。
支援風格：
- popmart: 泡泡瑪特盲盒公仔 (預設)
- anime: 日系動漫手辦
- ceramic: 亮面陶瓷 / 琺瑯樹脂
- handpainted: 手繪卡通 / 奇幻 RPG
- mecha: 科幻機甲 / 金屬玩具
- clay: 黏土定格動畫泥偶
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import sys
from typing import Dict, List, Optional

GLOBAL_WHISKER_NEGATIVE = "thin protruding whiskers outside silhouette, floating hair strands, webbed fingers, stray thin spikes"
GLOBAL_SILHOUETTE_GUIDE = "compact self-contained silhouette without thin floating whiskers or stray hair strands sticking out into the white background (any whiskers or facial markings must stay strictly painted inside the cheek contour)"


@dataclass(frozen=True)
class StyleConfig:
    id: str
    name: str
    aliases: List[str]
    roughness: float
    metallic: float
    specular_ior_level: float
    ao_strength: float
    ao_min_clamp: float
    prompt_style: str
    prompt_negative: str
    description: str
    coat_weight: float = 0.0
    coat_roughness: float = 0.03


STYLE_REGISTRY: Dict[str, StyleConfig] = {
    "popmart": StyleConfig(
        id="popmart",
        name="泡泡瑪特盲盒 (Pop Mart / Matte Vinyl Toy)",
        aliases=["blindbox", "vinyl", "toy", "pop_mart"],
        roughness=0.66,
        metallic=0.0,
        specular_ior_level=0.22,
        ao_strength=0.32,
        ao_min_clamp=0.35,
        coat_weight=0.0,
        prompt_style=(
            "Pop Mart blind box art toy figure style, cute chibi proportions, "
            "soft airbrushed gradients, smooth matte vinyl toy texture, rounded soft contours, "
            "clean soft studio illumination, circular toy pedestal"
        ),
        prompt_negative="photorealistic, gritty, heavy brushstrokes, high gloss reflection, metallic chrome, sharp cyber edges",
        description="柔和啞光乙烯基塑膠、無金屬感、溫潤均勻漸層，泡泡瑪特盲盒潮玩 (預設)",
    ),
    "anime": StyleConfig(
        id="anime",
        name="日系動漫手辦 (Japanese Anime Figure / Cel-Style)",
        aliases=["manga", "cel", "figurine", "anime_figure"],
        roughness=0.40,
        metallic=0.0,
        specular_ior_level=0.38,
        ao_strength=0.12,
        ao_min_clamp=0.65,
        coat_weight=0.0,
        prompt_style=(
            "Japanese anime scale PVC figure style, Good Smile Company / Alter collector quality, "
            "transparent acrylic circular base, gradient translucent flame horns and hair, "
            "vibrant anime color palette, glossy anime eyes with catchlights, satin smooth skin finish, delicate rim lighting"
        ),
        prompt_negative="photorealistic, clay thumbprints, rough painted texture, dark dirty ambient occlusion, metallic panels, weathered",
        description="二次元 PVC 手辦、半透明漸層細節、微光澤緞面皮膚、極淡乾淨陰影",
    ),
    "ceramic": StyleConfig(
        id="ceramic",
        name="亮面陶瓷 / 琺瑯樹脂 (Glossy Ceramic / Resin)",
        aliases=["resin", "porcelain", "enamel", "glossy", "glazed"],
        roughness=0.10,
        metallic=0.0,
        specular_ior_level=0.85,
        ao_strength=0.22,
        ao_min_clamp=0.45,
        coat_weight=1.0,
        coat_roughness=0.03,
        prompt_style=(
            "Glossy glazed ceramic and epoxy resin figurine style, thick glass-like clearcoat lacquer finish, "
            "intense sharp vertical specular reflections, liquid porcelain shine, polished smooth finish, "
            "rich saturated baked colors under studio softbox, glossy circular base"
        ),
        prompt_negative="matte, rough, dusty, fabric texture, unpolished, dull, flat diffuse, paper texture",
        description="晶透釉面高光、清澈亮面漆層 (Clearcoat)、宛如陶瓷燒製或環氧樹脂潮玩",
    ),
    "handpainted": StyleConfig(
        id="handpainted",
        name="手繪卡通 / 奇幻 RPG (Hand-Painted Fantasy RPG)",
        aliases=["rpg", "stylized", "hand_painted", "unlit", "game", "miniature"],
        roughness=0.98,
        metallic=0.0,
        specular_ior_level=0.02,
        ao_strength=0.04,
        ao_min_clamp=0.88,
        coat_weight=0.0,
        prompt_style=(
            "Hand-painted fantasy tabletop miniature figurine style, Citadel / Vallejo acrylic paint, "
            "visible fine directional brush hatching, painted edge highlights, rich saturated warm shadows, "
            "textured scenic cobblestone and moss miniature gaming base, Warhammer / D&D miniature aesthetic"
        ),
        prompt_negative="specular reflections, glossy clearcoat, metallic chrome, photorealistic 3D render, smooth plastic vinyl",
        description="桌遊塗裝微縮模型、筆觸排線細節、手繪邊緣高光、純漫反射無干擾反射",
    ),
    "mecha": StyleConfig(
        id="mecha",
        name="科幻機甲 / 金屬玩具 (Sci-Fi Mecha Metal)",
        aliases=["metal", "mech", "robot", "scifi", "metallic"],
        roughness=0.28,
        metallic=0.85,
        specular_ior_level=0.60,
        ao_strength=0.48,
        ao_min_clamp=0.22,
        coat_weight=0.0,
        prompt_style=(
            "Sci-fi mecha cyborg robot conversion, die-cast armored metal plating, "
            "segmented mechanical panels, panel line seams, articulated mechanical ball joints, "
            "metallic copper and anodized bronze finish, glowing amber optical sensor eyes, industrial alloy display pedestal, studio rim highlights"
        ),
        prompt_negative="organic clay, soft squishy cartoon, vinyl matte, plasticine, fabric, fuzzy, blurred seams",
        description="硬派機甲裝甲、機械分件線、冷冽重金屬/古銅反光、深邃結構陰影",
    ),
    "clay": StyleConfig(
        id="clay",
        name="黏土定格動畫 (Clay Stop-Motion / Plasticine)",
        aliases=["plasticine", "stopmotion", "craft", "clay_sculpt"],
        roughness=0.92,
        metallic=0.0,
        specular_ior_level=0.10,
        ao_strength=0.45,
        ao_min_clamp=0.28,
        coat_weight=0.0,
        prompt_style=(
            "Handcrafted clay stop-motion plasticine puppet style, Aardman / Laika animation character, "
            "matte tactile colored clay surface, subtle hand-sculpted clay modeling marks and thumb impressions, "
            "organic soft edges, earthy clay mound base, warm workshop directional light"
        ),
        prompt_negative="glossy plastic, metallic reflection, smooth chrome, digital sharp polygon edges, cel-shaded",
        description="定格動畫黏土泥偶、手工捏塑質感、重度啞光、濃郁黏土縫隙陰影",
    ),
    "fur": StyleConfig(
        id="fur",
        name="毛皮手辦 / 植絨公仔 (Fluffy Fur & Plush Pelt)",
        aliases=["fluffy", "pelt", "animal", "pet", "dog", "cat", "plush", "furry"],
        roughness=0.95,
        metallic=0.0,
        specular_ior_level=0.10,
        ao_strength=0.28,
        ao_min_clamp=0.40,
        coat_weight=0.0,
        prompt_style=(
            "Detailed soft fluffy fur coat texture, stylized animal pelt figurine, plush art toy sculpture, "
            "crisp directional fur clumps and fine hair tufts, warm rim lighting, soft studio illumination"
        ),
        prompt_negative="smooth plastic, glossy vinyl, metallic chrome, bald skin, flat wallpaper streaks, distorted seams",
        description="高頻毛髮纖維質感、超高粗糙度、無鏡面反光、邊緣毛茸柔光，專為貓狗寵物與植絨盲盒設計",
    ),
}

# 建立別名查找表
_ALIAS_MAP: Dict[str, str] = {}
for canonical_id, config in STYLE_REGISTRY.items():
    _ALIAS_MAP[canonical_id.lower()] = canonical_id
    for alias in config.aliases:
        _ALIAS_MAP[alias.lower()] = canonical_id


def get_style(name_or_alias: Optional[str]) -> StyleConfig:
    """
    根據名稱或別名取得 StyleConfig。若未傳入或找不到，預設回傳 'popmart'。
    """
    if not name_or_alias:
        return STYLE_REGISTRY["popmart"]

    normalized = name_or_alias.strip().lower().replace("-", "_").replace(" ", "_")
    target_id = _ALIAS_MAP.get(normalized, "popmart")
    return STYLE_REGISTRY[target_id]


def list_styles() -> List[StyleConfig]:
    return list(STYLE_REGISTRY.values())


def generate_prompt(style_name: str, subject: str) -> str:
    """
    生成標準 21:9 三視圖（正、左側、背）角色生成提示詞。
    """
    cfg = get_style(style_name)
    layout = (
        "Three full-body orthographic turnaround views in a single horizontal row, 21:9 aspect ratio: "
        "Left = Front view (A-pose), Middle = Left Side profile view (facing left), Right = Back view (A-pose). "
        "Strictly aligned head-to-toe heights across all three views."
    )
    background = (
        "Pure solid flat white background (#FFFFFF), no ground shadow, no floor line, no pedestal, "
        "no text, no labels, generous white margin between the three views so they never touch or overlap. "
        f"{GLOBAL_SILHOUETTE_GUIDE}."
    )
    combined_negative = f"{cfg.prompt_negative}, {GLOBAL_WHISKER_NEGATIVE}"
    return (
        f"2D character turnaround sheet of {subject}, {layout}. "
        f"Style: {cfg.prompt_style}. "
        f"Environment: {background} --ar 21:9 --no {combined_negative}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="2D-to-3D 風格管理與 Prompt 生成工具")
    subparsers = parser.add_subparsers(dest="subcommand", help="子命令")

    # list 子命令
    subparsers.add_parser("list", help="列出所有支援的風格與 PBR 參數")

    # prompt 子命令
    prompt_parser = subparsers.add_parser("prompt", help="產出指定風格的 2D 三視圖生成 Prompt")
    prompt_parser.add_argument("--style", "-s", default="popmart", help="風格名稱或別名 (預設: popmart)")
    prompt_parser.add_argument("--subject", required=True, help="角色主體描述 (例如: 'cute baby dragon warrior')")

    args = parser.parse_args()

    if args.subcommand == "list" or not args.subcommand:
        print("=" * 80)
        print(f"{'風格 ID':<12} {'名稱':<32} {'Rough':<8} {'Metal':<8} {'Spec':<8} {'AO':<6}")
        print("-" * 80)
        for s in list_styles():
            alias_str = f"[{', '.join(s.aliases[:2])}]"
            name_display = f"{s.name} {alias_str}"
            print(f"{s.id:<12} {name_display:<32} {s.roughness:<8.2f} {s.metallic:<8.2f} {s.specular_ior_level:<8.2f} {s.ao_strength:<6.2f}")
        print("=" * 80)
        print("💡 用法範例:")
        print("  python3 scripts/styles.py prompt --style anime --subject 'cute fox adventurer'")
        print("  python3 scripts/bake_multiview_textures.py --style ceramic ...")
    elif args.subcommand == "prompt":
        prompt = generate_prompt(args.style, args.subject)
        cfg = get_style(args.style)
        print(f"🎨 風格: {cfg.name} (ID: {cfg.id})")
        print("\n📋 產出的 2D 三視圖 Prompt (可直接複製到 Midjourney / DALL-E / Flux):")
        print("-" * 80)
        print(prompt)
        print("-" * 80)


if __name__ == "__main__":
    main()
