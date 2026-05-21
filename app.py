import base64
import json
import os
import random
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2
import requests

try:
    from onvif import ONVIFCamera
except Exception:
    ONVIFCamera = None

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Default prompt (preserved from v1)
# ──────────────────────────────────────────────────────────────

_DEFAULT_PROMPT = (
    "You are a CCTV monitoring AI. Analyze the image and describe what you observe. "
    "Note any unusual activities, safety concerns, or points of interest in plain language."
)


_DEFAULT_ZONE_PROMPT = (
    'You are a multi-camera CCTV monitoring AI for zone "{zone_name}". '
    'You are given {cam_count} simultaneous views:\n{cam_labels}\n\n'
    'Analyze all views together. Describe what you observe across the cameras '
    'and note any safety concerns or unusual activities.'
)


def _build_zone_prompt(zone_name: str, cam_labels: list[tuple[str, str]]) -> str:
    """Build a generic multi-view prompt for a zone (used as last-resort fallback)."""
    label_lines = "\n".join(
        f"- Image[{i}] = Camera \"{cid}\" ({lbl or cid})"
        for i, (cid, lbl) in enumerate(cam_labels)
    )
    return _DEFAULT_ZONE_PROMPT.format(
        zone_name=zone_name,
        cam_count=len(cam_labels),
        cam_labels=label_lines,
    )


def _apply_language(prompt: str, language: str) -> str:
    """Append language instruction to prompt if not English."""
    lang = (language or "English").strip()
    if lang and lang.lower() != "english":
        return prompt + f"\n\nIMPORTANT: You MUST respond entirely in {lang}."
    return prompt


# ──────────────────────────────────────────────────────────────
# Factory Safety Scenarios
# ──────────────────────────────────────────────────────────────

@dataclass
class ScenarioTemplate:
    id: str
    name: str          # zh-TW
    name_en: str       # English
    name_ja: str       # 日本語
    description: str   # zh-TW
    desc_en: str       # English
    desc_ja: str       # 日本語
    prompt: str        # English (default)
    prompt_zh: str     # 繁體中文
    prompt_ja: str     # 日本語
    trigger_level: int
    pre_sec: int
    post_sec: int

    def get_prompt(self, response_language: str = "English") -> str:
        """Return prompt in the appropriate language, falling back to English."""
        lang = (response_language or "English").lower()
        if ("chinese" in lang or "中文" in lang) and self.prompt_zh:
            return self.prompt_zh
        if ("japanese" in lang or "日本語" in lang) and self.prompt_ja:
            return self.prompt_ja
        return self.prompt

_DANGER_TAG_INSTRUCTION = (
    '\n\nAt the very end of your response, you MUST include exactly one tag in this format: [DANGER:X] '
    'where X is an integer 0-10 indicating the danger level. Example: [DANGER:0] means safe, [DANGER:7] means high risk.'
)
_DANGER_TAG_INSTRUCTION_ZH = (
    '\n\n請在回覆的最後，加上恰好一個標籤，格式為：[DANGER:X]，'
    'X 為 0-10 的整數表示危險等級。例如：[DANGER:0] 表示安全，[DANGER:7] 表示高風險。'
)
_DANGER_TAG_INSTRUCTION_JA = (
    '\n\n回答の最後に必ず次の形式のタグを1つだけ含めてください：[DANGER:X]'
    '（X は危険度を示す 0〜10 の整数）。例：[DANGER:0] は安全、[DANGER:7] は高リスクを示します。'
)

FACTORY_SCENARIOS: Dict[str, ScenarioTemplate] = {
    "fire_smoke": ScenarioTemplate(
        id="fire_smoke", name="🔥 火災/煙霧偵測",
        name_en="🔥 Fire / Smoke Detection",
        name_ja="🔥 火災・煙検知",
        description="偵測火焰、煙霧、火花等火災前兆",
        desc_en="Detect flames, smoke, sparks and fire precursors",
        desc_ja="火炎・煙・火花など火災の前兆を検知",
        prompt=(
            'You are a factory fire safety AI. Analyze the image for signs of fire, smoke, sparks, or overheating.\n'
            'Describe what you see in plain language: Are there visible flames, smoke plumes, sparks, or heat haze? '
            'How many people are visible and are any of them at risk? What is the location and severity?\n'
            'Danger scale: 0=safe, 1-3=possible steam or haze, 4-6=confirmed smoke or sparks, 7-10=active fire.'
        ) + _DANGER_TAG_INSTRUCTION,
        prompt_zh=(
            '你是工廠火災安全AI。分析畫面中是否有火災、煙霧、火花或過熱跡象。\n'
            '以白話描述所見：是否有可見火焰、煙霧、火花或熱霧？畫面中有多少人，是否有人處於危險中？位置與嚴重程度為何？\n'
            '危險等級：0=安全，1-3=可能有蒸氣或霧霾，4-6=確認有煙霧或火花，7-10=火焰燃燒中。'
        ) + _DANGER_TAG_INSTRUCTION_ZH,
        prompt_ja=(
            'あなたは工場の火災安全AIです。画像に火災・煙・火花・過熱の兆候がないか分析してください。\n'
            '平易な言葉で説明してください：可視炎、煙、火花、熱霞はありますか？人は何人見え、危険にさらされている人はいますか？場所と深刻度は？\n'
            '危険度：0=安全、1-3=蒸気・霞の可能性、4-6=煙・火花を確認、7-10=火炎燃焼中。'
        ) + _DANGER_TAG_INSTRUCTION_JA,
        trigger_level=5, pre_sec=20, post_sec=40,
    ),
    "ppe_check": ScenarioTemplate(
        id="ppe_check", name="⛑️ PPE 安全防護",
        name_en="⛑️ PPE Compliance",
        name_ja="⛑️ PPE安全確認",
        description="安全帽/反光背心/安全鞋等合規檢查",
        desc_en="Helmet / vest / footwear compliance check",
        desc_ja="ヘルメット・反射ベスト・安全靴の着用確認",
        prompt=(
            'You are a factory PPE compliance AI. Check whether workers in the image are wearing the required '
            'safety equipment: hard hat, reflective vest, safety footwear.\n'
            'Describe each visible worker: what PPE are they wearing or missing? Note any violations.\n'
            'Danger scale: 0=all compliant, 3-5=minor violations, 7-10=critical violations (no helmet near machinery).'
        ) + _DANGER_TAG_INSTRUCTION,
        prompt_zh=(
            '你是工廠PPE合規AI。檢查畫面中工人是否配戴必要的安全裝備：安全帽、反光背心、安全鞋。\n'
            '描述每位可見工人：配戴了哪些PPE、缺少哪些？記錄任何違規情況。\n'
            '危險等級：0=全員合規，3-5=輕微違規，7-10=嚴重違規（機械旁未戴安全帽）。'
        ) + _DANGER_TAG_INSTRUCTION_ZH,
        prompt_ja=(
            'あなたは工場のPPE適合性確認AIです。画像内の作業員が必要な安全装備（ヘルメット・反射ベスト・安全靴）を着用しているか確認してください。\n'
            '各作業員について：着用しているPPEと未着用のものを説明し、違反があれば記録してください。\n'
            '危険度：0=全員適合、3-5=軽微な違反、7-10=重大な違反（機械周辺でヘルメット未着用）。'
        ) + _DANGER_TAG_INSTRUCTION_JA,
        trigger_level=6, pre_sec=15, post_sec=30,
    ),
    "fall_injury": ScenarioTemplate(
        id="fall_injury", name="🚨 人員跌倒/受傷",
        name_en="🚨 Fall / Injury Detection",
        name_ja="🚨 転倒・負傷検知",
        description="偵測人員倒地、受傷、失去意識",
        desc_en="Detect person down, injured, or unconscious",
        desc_ja="転倒・負傷・意識喪失の検知",
        prompt=(
            'You are a factory worker safety AI. Look for any person who has fallen, is lying on the ground, '
            'appears injured, or is in an unusual posture that suggests distress.\n'
            'Describe the scene: How many people are visible? Is anyone down or in an abnormal position? '
            'What are the surrounding conditions?\n'
            'Danger scale: 0=all standing normally, 5-7=unusual posture or stumble, 8-10=person collapsed on ground.'
        ) + _DANGER_TAG_INSTRUCTION,
        prompt_zh=(
            '你是工廠工人安全AI。尋找任何跌倒、躺在地上、看似受傷或處於異常姿勢的人員。\n'
            '描述場景：有多少人可見？是否有人倒地或姿勢異常？周圍環境狀況如何？\n'
            '危險等級：0=所有人正常站立，5-7=姿勢異常或踉蹌，8-10=人員倒地。'
        ) + _DANGER_TAG_INSTRUCTION_ZH,
        prompt_ja=(
            'あなたは工場作業員の安全AIです。転倒・地面への倒れ込み・負傷・苦痛を示す異常な姿勢の人物を探してください。\n'
            'シーンを説明してください：何人見えますか？倒れているか異常な体勢の人はいますか？周囲の状況は？\n'
            '危険度：0=全員正常に立っている、5-7=異常な姿勢またはよろめき、8-10=地面に倒れた人物。'
        ) + _DANGER_TAG_INSTRUCTION_JA,
        trigger_level=7, pre_sec=15, post_sec=60,
    ),
    "restricted_zone": ScenarioTemplate(
        id="restricted_zone", name="🚧 危險區域入侵",
        name_en="🚧 Restricted Zone Intrusion",
        name_ja="🚧 立入禁止区域侵入",
        description="偵測人員進入標記警告區域或機械範圍",
        desc_en="Detect personnel entering hazard or machinery zones",
        desc_ja="警告エリアや機械周辺への侵入検知",
        prompt=(
            'You are a factory restricted zone monitor. Detect whether any person has entered a hazard zone, '
            'crossed warning tape, or is too close to dangerous machinery.\n'
            'Describe: How many people are near or inside restricted areas? What kind of zone markings are visible? '
            'Are there barriers or warning signs?\n'
            'Danger scale: 0=area clear, 3-5=person near boundary, 7-10=person inside restricted zone.'
        ) + _DANGER_TAG_INSTRUCTION,
        prompt_zh=(
            '你是工廠危險區域監控AI。偵測是否有人進入危險區、穿越警示膠帶，或過於接近危險機械。\n'
            '描述：有多少人在受限區域附近或內部？可見哪種區域標示？是否有隔離設施或警告標誌？\n'
            '危險等級：0=區域淨空，3-5=人員接近邊界，7-10=人員進入受限區域內。'
        ) + _DANGER_TAG_INSTRUCTION_ZH,
        prompt_ja=(
            'あなたは工場の立入禁止区域監視AIです。危険エリアへの侵入・警告テープの越境・危険機械への接近を検知してください。\n'
            '説明してください：制限区域の近くまたは内部に何人いますか？どのような区域標示が見えますか？バリアや警告標識はありますか？\n'
            '危険度：0=エリア淡空、3-5=境界付近の人物、7-10=立入禁止区域内の人物。'
        ) + _DANGER_TAG_INSTRUCTION_JA,
        trigger_level=6, pre_sec=15, post_sec=30,
    ),
    "equipment_anomaly": ScenarioTemplate(
        id="equipment_anomaly", name="⚙️ 設備異常監控",
        name_en="⚙️ Equipment Anomaly",
        name_ja="⚙️ 設備異常監視",
        description="機台漏液、冒煙、設備損壞、過熱跡象",
        desc_en="Fluid leaks, smoke, equipment damage, overheating",
        desc_ja="液漏れ・煙・機器損傷・過熱の兆候",
        prompt=(
            'You are a factory equipment monitoring AI. Look for fluid leaks, unusual smoke or steam coming from '
            'equipment, visible damage, sparks, or signs of overheating.\n'
            'Describe what equipment is visible and its condition. Note any abnormalities.\n'
            'Danger scale: 0=normal operation, 2-4=minor anomaly, 5-7=active leak or steam, 8-10=critical damage or fire risk.'
        ) + _DANGER_TAG_INSTRUCTION,
        prompt_zh=(
            '你是工廠設備監控AI。尋找液體洩漏、設備冒出的異常煙霧或蒸氣、可見損壞、火花或過熱跡象。\n'
            '描述畫面中可見的設備及其狀況，記錄任何異常情況。\n'
            '危險等級：0=正常運作，2-4=輕微異常，5-7=活動性洩漏或蒸氣，8-10=嚴重損壞或火災風險。'
        ) + _DANGER_TAG_INSTRUCTION_ZH,
        prompt_ja=(
            'あなたは工場設備監視AIです。液体漏洩、設備からの異常な煙や蒸気、可視損傷、火花、過熱の兆候を探してください。\n'
            '見える設備とその状態を説明し、異常があれば記録してください。\n'
            '危険度：0=正常稼働、2-4=軽微な異常、5-7=活動的な漏洩または蒸気、8-10=重大な損傷または火災リスク。'
        ) + _DANGER_TAG_INSTRUCTION_JA,
        trigger_level=5, pre_sec=20, post_sec=40,
    ),
    "chemical_spill": ScenarioTemplate(
        id="chemical_spill", name="☢️ 化學品洩漏",
        name_en="☢️ Chemical Spill",
        name_ja="☢️ 化学物質漏洩",
        description="化學品溢出、容器破損、有害蒸氣",
        desc_en="Chemical spill, damaged containers, hazardous vapor",
        desc_ja="化学物質の漏洩・容器破損・有害蒸気",
        prompt=(
            'You are a hazmat safety AI. Look for chemical spills on the floor, damaged or leaking containers, '
            'visible vapor or fumes, and any people who may be exposed.\n'
            'Describe the scene: What substances or containers are visible? Is there pooling liquid, vapor, or odor indicators? '
            'How many people are nearby?\n'
            'Danger scale: 0=safe, 4-6=spill detected but no human exposure, 7-10=people exposed to hazardous material.'
        ) + _DANGER_TAG_INSTRUCTION,
        prompt_zh=(
            '你是危險物品安全AI。尋找地板上的化學品溢出、破損或洩漏的容器、可見蒸氣或煙霧，以及任何可能暴露於危險物品的人員。\n'
            '描述場景：可見哪些物質或容器？是否有積液、蒸氣或氣味指標？附近有多少人？\n'
            '危險等級：0=安全，4-6=偵測到洩漏但無人暴露，7-10=人員暴露於危險物質中。'
        ) + _DANGER_TAG_INSTRUCTION_ZH,
        prompt_ja=(
            'あなたは危険物安全AIです。床上の化学物質流出、破損または漏洩する容器、可視蒸気またはヒューム、暴露の可能性がある人物を探してください。\n'
            'シーンを説明してください：どのような物質や容器が見えますか？液体の溜まり・蒸気・臭気の指標はありますか？近くに何人いますか？\n'
            '危険度：0=安全、4-6=流出を検知したが人体暴露なし、7-10=人員が危険物質に暴露。'
        ) + _DANGER_TAG_INSTRUCTION_JA,
        trigger_level=5, pre_sec=20, post_sec=40,
    ),
    "general_safety": ScenarioTemplate(
        id="general_safety", name="🏭 綜合工廠安全",
        name_en="🏭 General Factory Safety",
        name_ja="🏭 総合工場安全",
        description="全方位工安監控：火災/跌倒/PPE/設備/入侵",
        desc_en="All-round safety: fire / fall / PPE / equipment / intrusion",
        desc_ja="総合安全監視：火災・転倒・PPE・設備・侵入",
        prompt=(
            'You are a comprehensive factory safety AI. Monitor the image for ALL types of hazards: '
            'fire, smoke, falls, PPE violations, equipment problems, unauthorized zone access, and chemical spills.\n'
            'Describe the overall scene and any safety concerns. Note the number of visible people and any issues.\n'
            'Danger scale: 0=safe, 1-3=minor concern, 4-6=moderate hazard, 7-10=emergency requiring immediate action.'
        ) + _DANGER_TAG_INSTRUCTION,
        prompt_zh=(
            '你是全方位工廠安全AI。監控畫面中所有類型的危害：火災、煙霧、跌倒、PPE違規、設備問題、未授權進入區域及化學品洩漏。\n'
            '描述整體場景及任何安全疑慮，記錄可見人數及任何問題。\n'
            '危險等級：0=安全，1-3=輕微疑慮，4-6=中度危害，7-10=需要立即行動的緊急狀況。'
        ) + _DANGER_TAG_INSTRUCTION_ZH,
        prompt_ja=(
            'あなたは包括的な工場安全AIです。すべての種類の危険を監視してください：火災・煙・転倒・PPE違反・設備問題・不正区域侵入・化学物質漏洩。\n'
            '全体的なシーンと安全上の懸念を説明し、見える人数と問題点を記録してください。\n'
            '危険度：0=安全、1-3=軽微な懸念、4-6=中程度の危険、7-10=即時対応が必要な緊急事態。'
        ) + _DANGER_TAG_INSTRUCTION_JA,
        trigger_level=5, pre_sec=20, post_sec=40,
    ),
    "people_drinks": ScenarioTemplate(
        id="people_drinks", name="👥 人員/飲品計數",
        name_en="👥 People / Drinks Count",
        name_ja="👥 人数・飲料カウント",
        description="計算人員數量及桌面飲品容器",
        desc_en="Count people and beverage containers",
        desc_ja="人数と飲料容器のカウント",
        prompt=(
            'You are a multi-view room analysis AI. Analyze the scene for people and beverage containers.\n'
            'Counting rules:\n'
            '1. People: Count each unique individual visible. Identify by clothing color, position, and build to avoid double-counting. '
            'Briefly describe each person\'s position and clothing.\n'
            '2. Beverage containers: Count all drink containers (plastic bottles, cans, cups, glasses, mugs, etc.) '
            'on tables, held in hand, or on any surface. List each container\'s type and location.\n'
            '3. End with 1-2 sentences summarizing the overall scene.\n'
            'Danger level guide: 0=normal, 3=unusually high number of people.'
        ) + _DANGER_TAG_INSTRUCTION,
        prompt_zh=(
            '你是多視角房間分析AI。分析畫面中的人員與飲品容器。\n'
            '計算規則：\n'
            '1. 人員：計算畫面中每一位獨特的人，以衣著顏色、位置、體型辨識，避免重複計算。簡要描述每人的位置與衣著。\n'
            '2. 飲品容器：計算所有飲品容器（寶特瓶、罐裝飲料、杯子、玻璃杯、馬克杯等），'
            '包含桌面、手持、任何表面上的容器。列出每個容器的種類與位置。\n'
            '3. 最後用1-2句話總結整體場景。\n'
            '危險等級參考：0=正常，3=人數異常多。'
        ) + _DANGER_TAG_INSTRUCTION_ZH,
        prompt_ja=(
            'あなたは多視点ルーム分析AIです。シーン内の人物と飲料容器を分析してください。\n'
            'カウントルール：\n'
            '1. 人物：画面内の各ユニークな個人をカウントします。衣服の色・位置・体格で識別し二重カウントを避けてください。各人物の位置と衣服を簡潔に説明してください。\n'
            '2. 飲料容器：テーブル上・手持ち・あらゆる表面にある全ての飲料容器（ペットボトル・缶・カップ・グラス・マグカップ等）をカウントし、各容器の種類と位置をリストアップしてください。\n'
            '3. 最後に全体のシーンを1〜2文で要約してください。\n'
            '危険度ガイド：0=正常、3=人数が異常に多い。'
        ) + _DANGER_TAG_INSTRUCTION_JA,
        trigger_level=3, pre_sec=10, post_sec=20,
    ),
    "custom": ScenarioTemplate(
        id="custom", name="✏️ 自訂",
        name_en="✏️ Custom",
        name_ja="✏️ カスタム",
        description="使用自訂全域 Prompt + 關鍵字警報",
        desc_en="Use custom global prompt + keyword alerts",
        desc_ja="カスタムプロンプト＋キーワードアラート",
        prompt="", prompt_zh="", prompt_ja="",
        trigger_level=5, pre_sec=15, post_sec=30,
    ),
}


# ──────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────

class CameraConfig(BaseModel):
    """Per-camera configuration."""
    camera_id: str = Field(..., description="Unique camera identifier.")
    label: str = Field(default="", description="Human-friendly display name.")
    rtsp_url: str = Field(default="", description="RTSP stream URL.")
    enabled: bool = Field(default=True, description="Participate in capture and inference.")
    prompt: Optional[str] = Field(default=None, description="Per-camera VLM prompt; None inherits global.")
    interval_sec: float = Field(default=5.0, ge=0.1, le=60.0, description="Minimum seconds between inferences.")
    priority: int = Field(default=1, ge=1, le=10, description="Round-robin weight (higher = more frequent).")
    snapshot_enabled: Optional[bool] = Field(default=None, description="Override global snapshot_enabled; None inherits.")
    zone: str = Field(default="", description="Zone/room name. Cameras in the same zone do joint multi-view inference.")
    # ONVIF PTZ (optional, per-camera)
    onvif_host: str = Field(default="")
    onvif_port: int = Field(default=2020, ge=1, le=65535)
    onvif_username: str = Field(default="")
    onvif_password: str = Field(default="")


class CameraUpdate(BaseModel):
    """Partial update payload for PATCH /cameras/{id}. Only sent fields are applied."""
    label: Optional[str] = None
    rtsp_url: Optional[str] = None
    enabled: Optional[bool] = None
    prompt: Optional[str] = None
    interval_sec: Optional[float] = Field(default=None, ge=0.1, le=60.0)
    priority: Optional[int] = Field(default=None, ge=1, le=10)
    snapshot_enabled: Optional[bool] = None
    zone: Optional[str] = None
    onvif_host: Optional[str] = None
    onvif_port: Optional[int] = Field(default=None, ge=1, le=65535)
    onvif_username: Optional[str] = None
    onvif_password: Optional[str] = None


class EnableBody(BaseModel):
    enabled: bool


class GlobalConfig(BaseModel):
    """Application-wide settings shared by all cameras."""
    ollama_url: str = Field(default="http://10.22.22.166:30082")
    model: str = Field(default="ministral-3:8b")
    version: str = Field(default="1.0", description="'1.0' baseline, '1.1' enables keyword alerts.")
    prompt: str = Field(default=_DEFAULT_PROMPT, description="Default VLM prompt for cameras without per-camera prompt.")
    alert_keywords: list[str] = Field(default_factory=lambda: ["alarm", "alert", "danger", "警報", "危險"])
    snapshot_enabled: bool = Field(default=True)
    snapshot_keyword: str = Field(default="建立快照,Create snapshot")
    snapshot_dir: str = Field(default="snapshots")
    response_language: str = Field(default="English", description="Language for VLM responses: English, 繁體中文（Traditional Chinese）, 日本語")
    yolo_enabled: bool = Field(default=False, description="Enable YOLO pre-filter before VLM inference.")
    yolo_classes: list[str] = Field(default_factory=lambda: ["person"], description="COCO class names to trigger VLM.")
    yolo_confidence: float = Field(default=0.35, ge=0.1, le=0.9, description="YOLO detection confidence threshold.")
    # Factory scenario + structured output
    scenario: str = Field(default="general_safety", description="Active scenario ID from FACTORY_SCENARIOS.")
    # Video clip recording
    video_clip_enabled: bool = Field(default=False, description="Enable video clip recording on trigger.")
    video_pre_sec: int = Field(default=15, ge=5, le=60, description="Pre-buffer seconds before trigger.")
    video_post_sec: int = Field(default=30, ge=5, le=120, description="Post-buffer seconds after trigger.")
    video_fps: int = Field(default=5, ge=1, le=15, description="Video recording FPS.")
    video_clip_dir: str = Field(default="video_clips", description="Directory for video clips.")
    trigger_danger_level: int = Field(default=5, ge=1, le=10, description="Danger level threshold for alert/recording.")
    # Signal light
    alert_light_enabled: bool = Field(default=False, description="Enable signal light API on alert state change.")
    alert_light_url: str = Field(default="http://10.22.22.168:8080/api/signal", description="Signal light API endpoint URL.")


# ──────────────────────────────────────────────────────────────
# Runtime State
# ──────────────────────────────────────────────────────────────

@dataclass
class CameraState:
    """Per-camera runtime state. All fields are protected by self.lock."""
    # Latest captured frame
    last_frame_b64: Optional[str] = None
    last_frame_ts: Optional[float] = None
    # Latest VLM inference
    last_reply: str = ""
    last_reply_ts: Optional[float] = None
    last_infer_ms: Optional[int] = None
    last_infer_ts: Optional[float] = None  # time.monotonic(); used by round-robin scheduler
    # Errors
    last_error: str = ""
    # Alerts (v1.1)
    alert_active: bool = False
    alert_reason: str = ""
    alert_ts: Optional[float] = None
    # Snapshots
    last_snapshot_path: str = ""
    last_snapshot_text_path: str = ""
    last_snapshot_ts: Optional[float] = None
    last_snapshot_error: str = ""
    # ONVIF PTZ patrol
    onvif_patrol_active: bool = False
    onvif_patrol_thread: Optional[threading.Thread] = None
    onvif_patrol_stop_event: threading.Event = field(default_factory=threading.Event)
    last_onvif_error: str = ""
    # YOLO pre-filter
    last_frame_np: Optional[object] = None   # np.ndarray raw frame; written by capture thread
    last_yolo_result: str = ""               # "person:0.87" | "skip" | ""
    last_yolo_boxes: list = field(default_factory=list)  # [(x1,y1,x2,y2,class_name,conf)]
    # Pre-buffer ring buffer for video clips (stores (ts, jpeg_bytes) tuples)
    frame_ring: object = field(default_factory=lambda: deque(maxlen=900))
    frame_ring_last_ts: float = 0.0
    # Video recording state
    recording: bool = False
    record_until: float = 0.0
    record_thread: Optional[threading.Thread] = None
    last_clip_path: str = ""
    last_clip_ts: Optional[float] = None
    # Structured output
    last_danger_level: int = 0
    last_event_type: str = ""
    # Capture thread
    capture_thread: Optional[threading.Thread] = None
    capture_stop_event: threading.Event = field(default_factory=threading.Event)
    # Mutex protecting all fields above
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class RuntimeState:
    """Global runtime state."""
    running: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    # camera_id → CameraState
    cameras: Dict[str, CameraState] = field(default_factory=dict)
    infer_thread: Optional[threading.Thread] = None
    # Protects cameras dict structure (add / remove keys).
    # Individual camera data is protected by CameraState.lock.
    global_lock: threading.Lock = field(default_factory=threading.Lock)


# ──────────────────────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────────────────────

cfg = GlobalConfig()
cameras_cfg: List[CameraConfig] = []   # camera list – owned by /cameras endpoints
st = RuntimeState()
sess = requests.Session()

_CONFIG_FILE = Path(__file__).parent / "cameras.json"


# ──────────────────────────────────────────────────────────────
# Config persistence  (cameras.json + global config)
# ──────────────────────────────────────────────────────────────

def _save_config() -> None:
    """Persist cameras list and global config to cameras.json."""
    data = {
        "global": cfg.model_dump(),
        "cameras": [c.model_dump() for c in cameras_cfg],
    }
    tmp = _CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_CONFIG_FILE)


def _load_config() -> None:
    """Load cameras and global config from cameras.json on startup."""
    global cfg, cameras_cfg
    if not _CONFIG_FILE.exists():
        return
    try:
        data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        if "global" in data:
            cfg = GlobalConfig(**data["global"])
        if "cameras" in data:
            cameras_cfg = [CameraConfig(**c) for c in data["cameras"]]
        print(f"[CONFIG] Loaded {len(cameras_cfg)} camera(s) from {_CONFIG_FILE.name}")
    except Exception as e:
        print(f"[CONFIG] Failed to load {_CONFIG_FILE.name}: {e}")


_load_config()


# ──────────────────────────────────────────────────────────────
# Signal Light (alert lamp) integration
# ──────────────────────────────────────────────────────────────

_prev_global_alert: bool = False
_alert_light_lock = threading.Lock()


def _notify_signal_light(alert_on: bool) -> None:
    """Call the external signal light API. Runs in a daemon thread (fire-and-forget)."""
    if not cfg.alert_light_enabled or not cfg.alert_light_url:
        return
    try:
        if alert_on:
            payload = {
                "command": "SET_LIGHT",
                "color": "RED",
                "message": "alarm",
                "blink": True,
                "durationSec": 0,
            }
        else:
            payload = {
                "command": "SET_LIGHTGREEN",
                "message": "In operation",
                "blink": False,
                "durationSec": 0,
            }
        r = requests.post(cfg.alert_light_url, json=payload, timeout=3)
        print(f"[LIGHT] {'🔴 RED blink' if alert_on else '🟢 GREEN'} → {r.status_code}")
    except Exception as e:
        print(f"[LIGHT] signal light error: {e}")


def _update_signal_light() -> None:
    """Check if global alert state changed; if so fire the light API asynchronously."""
    global _prev_global_alert
    with _alert_light_lock:
        with st.global_lock:
            new_alert = any(s.alert_active for s in st.cameras.values())
        if new_alert != _prev_global_alert:
            _prev_global_alert = new_alert
            threading.Thread(
                target=_notify_signal_light, args=(new_alert,), daemon=True
            ).start()


# ──────────────────────────────────────────────────────────────
# FFmpeg GPU / capture backend detection
# ──────────────────────────────────────────────────────────────

_HW_MODE = os.getenv("VLM_HW_ACCEL", "auto").strip().lower() or "auto"
_HW_DEVICE = os.getenv("VLM_HW_DEVICE", "0").strip()


def _find_ffmpeg() -> str:
    """Find ffmpeg binary: env var → bundled → system PATH."""
    env = os.getenv("VLM_FFMPEG", "").strip()
    if env and Path(env).exists():
        return env
    # Check for bundled ffmpeg next to app.py
    bundled = Path(__file__).parent / "ffmpeg"
    if bundled.is_dir():
        for candidate in bundled.rglob("ffmpeg.exe"):
            return str(candidate)
        for candidate in bundled.rglob("ffmpeg"):
            if candidate.is_file():
                return str(candidate)
    return shutil.which("ffmpeg") or ""


_FFMPEG_BIN = _find_ffmpeg()
_FFMPEG_HAS_CUDA = False

if _FFMPEG_BIN and _HW_MODE not in {"off", "none", "cpu", "0"}:
    try:
        out = subprocess.run(
            [_FFMPEG_BIN, "-hwaccels"],
            capture_output=True, text=True, timeout=5,
        )
        _FFMPEG_HAS_CUDA = "cuda" in out.stdout.lower()
    except Exception:
        pass

# Ensure bundled ffmpeg's DLLs are findable
if _FFMPEG_BIN:
    _ffmpeg_dir = str(Path(_FFMPEG_BIN).parent)
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

print(f"[GPU] mode={_HW_MODE}  ffmpeg={'found' if _FFMPEG_BIN else 'NOT found'}"
      f"  cuda={'YES' if _FFMPEG_HAS_CUDA else 'no'}"
      f"  device={_HW_DEVICE}"
      f"  path={_FFMPEG_BIN[:80] if _FFMPEG_BIN else 'N/A'}")

# Set env-var for OpenCV's FFmpeg fallback path (CPU-only, no unsupported opts)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    "|fflags;nobuffer+discardcorrupt"
    "|flags;low_delay"
    "|max_delay;100000"
    "|stimeout;5000000"
    "|buffer_size;1048576"
    "|probesize;1048576"
    "|analyzeduration;500000"
)


# ──────────────────────────────────────────────────────────────
# General helpers
# ──────────────────────────────────────────────────────────────

def _normalize_ollama_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    return url


def _normalize_version(ver: str) -> str:
    v = (ver or "").strip().lower().lstrip("v")
    return "1.1" if v == "1.1" else "1.0"


def _check_alert(reply: str, keywords: list[str]) -> tuple[bool, str]:
    if not reply:
        return False, ""
    reply_lower = reply.lower()
    matched = [kw.strip() for kw in keywords if kw.strip() and kw.strip().lower() in reply_lower]
    if matched:
        return True, f"Matched keyword(s): {', '.join(dict.fromkeys(matched))}"
    return False, ""


def _get_camera_cfg(camera_id: str) -> Optional[CameraConfig]:
    with st.global_lock:
        for cam in cameras_cfg:
            if cam.camera_id == camera_id:
                return cam
    return None


# ──────────────────────────────────────────────────────────────
# Snapshot helpers  (per-camera subdirectory layout)
# ──────────────────────────────────────────────────────────────

def _normalize_snapshot_dir(snapshot_dir: str) -> Path:
    path = Path(snapshot_dir.strip()) if (snapshot_dir and snapshot_dir.strip()) else Path("snapshots")
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return path


def _camera_snapshot_dir(base_dir: str, camera_id: str) -> Path:
    """Return <base_dir>/<camera_id>/."""
    return _normalize_snapshot_dir(base_dir) / camera_id


def _save_temp_snapshot(image_b64: str, dir_path: Path) -> tuple[Path, Path]:
    dir_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    filename = ts.strftime("snapshot_%Y%m%d_%H%M%S_%f.jpg")
    final_path = dir_path / filename
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    temp_path.write_bytes(base64.b64decode(image_b64))
    return final_path, temp_path


def _finalize_snapshot(temp_path: Path, final_path: Path) -> Path:
    temp_path.replace(final_path)
    return final_path


def _discard_snapshot(temp_path: Optional[Path]) -> None:
    if not temp_path:
        return
    try:
        temp_path.unlink()
    except Exception:
        pass


def _save_snapshot_text(snapshot_path: Path, reply: str) -> Path:
    text_path = snapshot_path.with_suffix(".txt")
    text_path.write_text(reply or "", encoding="utf-8")
    return text_path


def _match_snapshot(reply: str, snapshot_keyword: str) -> bool:
    if not reply:
        return False
    keywords = [k.strip() for k in (snapshot_keyword or "").split(",") if k.strip()]
    if not keywords:
        return False
    reply_lower = reply.lower()
    return any(kw.lower() in reply_lower for kw in keywords)


def _resolve_snapshot_file(camera_id: str, filename: str) -> Optional[Path]:
    """Resolve a snapshot path safely, rejecting path-traversal attempts."""
    if not filename or Path(filename).name != filename:
        return None
    dir_path = _camera_snapshot_dir(cfg.snapshot_dir, camera_id).resolve()
    candidate = (dir_path / filename).resolve()
    try:
        common = os.path.commonpath([str(dir_path), str(candidate)])
    except ValueError:
        return None
    if common != str(dir_path):
        return None
    return candidate


# ──────────────────────────────────────────────────────────────
# ONVIF PTZ helpers (per-camera)
# ──────────────────────────────────────────────────────────────

def _get_onvif_ptz_profile(cam_cfg: CameraConfig):
    if ONVIFCamera is None:
        raise RuntimeError("onvif-zeep not installed")
    host = cam_cfg.onvif_host.strip()
    if not host:
        raise RuntimeError("ONVIF host not set")
    camera = ONVIFCamera(host, int(cam_cfg.onvif_port), cam_cfg.onvif_username, cam_cfg.onvif_password)
    media = camera.create_media_service()
    ptz = camera.create_ptz_service()
    profiles = media.GetProfiles()
    if not profiles:
        raise RuntimeError("No media profiles found")
    return ptz, profiles[0].token


def _ptz_stop(ptz, token: str) -> None:
    stop_req = ptz.create_type("Stop")
    stop_req.ProfileToken = token
    stop_req.PanTilt = True
    stop_req.Zoom = True
    ptz.Stop(stop_req)


def _ptz_move(ptz, token: str, pan_speed: float) -> None:
    move_req = ptz.create_type("ContinuousMove")
    move_req.ProfileToken = token
    move_req.Velocity = {"PanTilt": {"x": pan_speed, "y": 0.0}, "Zoom": 0.0}
    ptz.ContinuousMove(move_req)


def onvif_patrol_worker(camera_id: str) -> None:
    cam_state = st.cameras.get(camera_id)
    if cam_state is None:
        return
    cam_cfg = _get_camera_cfg(camera_id)
    if cam_cfg is None:
        return
    while not cam_state.onvif_patrol_stop_event.is_set() and not st.stop_event.is_set():
        try:
            ptz, token = _get_onvif_ptz_profile(cam_cfg)
            with cam_state.lock:
                cam_state.last_onvif_error = ""
            while not cam_state.onvif_patrol_stop_event.is_set() and not st.stop_event.is_set():
                _ptz_move(ptz, token, pan_speed=-0.24)
                time.sleep(10.0)
                _ptz_stop(ptz, token)
                if cam_state.onvif_patrol_stop_event.is_set():
                    break
                time.sleep(0.6)
                _ptz_move(ptz, token, pan_speed=0.24)
                time.sleep(10.0)
                _ptz_stop(ptz, token)
                time.sleep(0.6)
        except Exception as e:
            with cam_state.lock:
                cam_state.last_onvif_error = f"ONVIF patrol failed: {e}"
            time.sleep(2.0)


def start_onvif_patrol(camera_id: str) -> None:
    cam_state = st.cameras.get(camera_id)
    if cam_state is None or cam_state.onvif_patrol_active:
        return
    cam_state.onvif_patrol_stop_event.clear()
    t = threading.Thread(
        target=onvif_patrol_worker, args=(camera_id,),
        daemon=True, name=f"onvif-{camera_id}",
    )
    cam_state.onvif_patrol_thread = t
    cam_state.onvif_patrol_active = True
    t.start()


def stop_onvif_patrol(camera_id: str) -> None:
    cam_state = st.cameras.get(camera_id)
    if cam_state is None or not cam_state.onvif_patrol_active:
        return
    cam_state.onvif_patrol_stop_event.set()
    cam_state.onvif_patrol_active = False


# ──────────────────────────────────────────────────────────────
# Ollama VLM
# ──────────────────────────────────────────────────────────────

def ollama_vlm_chat(ollama_url: str, model: str, prompt: str,
                    images_b64: list[str]) -> str:
    """Call Ollama VLM with one or more images."""
    url = _normalize_ollama_url(ollama_url) + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "options": {"temperature": 0.2},
    }
    r = sess.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
            return data["message"]["content"]
        if "response" in data:
            return data["response"]
    return str(data)


# ──────────────────────────────────────────────────────────────
# Frame capture utilities
# ──────────────────────────────────────────────────────────────

def encode_jpeg_b64(frame, quality: int = 75) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _build_cv2_capture(rtsp: str) -> cv2.VideoCapture:
    """Open RTSP via OpenCV (CPU decode). Used as fallback when ffmpeg GPU is unavailable."""
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(rtsp)
    if cap.isOpened():
        buff = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
        if buff is not None:
            cap.set(buff, 1)
        for pname, val in [("CAP_PROP_OPEN_TIMEOUT_MSEC", 5000),
                           ("CAP_PROP_READ_TIMEOUT_MSEC", 3000)]:
            p = getattr(cv2, pname, None)
            if p is not None:
                cap.set(p, val)
    return cap


def _read_latest_frame(cap: cv2.VideoCapture, max_skip: int = 8):
    """Drop buffered frames to keep only the newest."""
    grabbed = False
    for _ in range(max_skip):
        if not cap.grab():
            break
        grabbed = True
    return cap.retrieve() if grabbed else cap.read()


# ──────────────────────────────────────────────────────────────
# Subprocess ffmpeg GPU capture
# ──────────────────────────────────────────────────────────────

def _probe_resolution(rtsp: str) -> tuple[int, int]:
    """Use ffprobe / ffmpeg to detect stream resolution. Returns (width, height) or (0, 0)."""
    ffprobe = shutil.which("ffprobe") or ""
    if not ffprobe and _FFMPEG_BIN:
        ffprobe = str(Path(_FFMPEG_BIN).parent / "ffprobe")
    if ffprobe and Path(ffprobe).exists():
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0:s=x",
                 "-rtsp_transport", "tcp", "-i", rtsp],
                capture_output=True, text=True, timeout=10,
            )
            parts = r.stdout.strip().split("x")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
    # Fallback: quick OpenCV probe
    cap = cv2.VideoCapture(rtsp)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
    return 0, 0


_DISPLAY_FPS = int(os.getenv("VLM_DISPLAY_FPS", "15"))


def _start_ffmpeg_process(rtsp: str, w: int, h: int, use_gpu: bool) -> subprocess.Popen:
    """Launch ffmpeg subprocess that decodes RTSP → raw BGR24 frames on stdout.
    Output is capped at _DISPLAY_FPS to reduce pipe throughput and CPU encoding load."""
    cmd = [_FFMPEG_BIN]
    if use_gpu:
        cmd += ["-hwaccel", "cuda", "-hwaccel_device", _HW_DEVICE, "-c:v", "h264_cuvid"]
    cmd += [
        "-rtsp_transport", "tcp",
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-max_delay", "50000",
        "-buffer_size", "65536",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-i", rtsp,
        "-vf", f"fps={_DISPLAY_FPS}",   # limit output framerate
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",
        "-sn",
        "-v", "error",
        "pipe:1",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=w * h * 3 * 2,   # ~2 frames buffer
    )


# ──────────────────────────────────────────────────────────────
# Worker threads
# ──────────────────────────────────────────────────────────────

def _capture_gpu(camera_id: str, rtsp_url: str, cam_state: CameraState) -> None:
    """Capture loop using subprocess ffmpeg with NVIDIA CUVID GPU decoding.

    Architecture:
      - A reader thread continuously reads raw frames from ffmpeg stdout
        into a single-slot buffer (always keeps the latest frame).
      - The main loop picks up the latest frame, JPEG-encodes it, and
        stores it in cam_state.  This decouples pipe I/O from encoding.
    """
    cam_stop = cam_state.capture_stop_event

    while not st.stop_event.is_set() and not cam_stop.is_set():
        w, h = _probe_resolution(rtsp_url)
        if w == 0 or h == 0:
            with cam_state.lock:
                cam_state.last_error = f"Cannot probe resolution: {rtsp_url}"
            time.sleep(2.0)
            continue

        frame_bytes = w * h * 3
        proc = _start_ffmpeg_process(rtsp_url, w, h, use_gpu=True)
        print(f"[CAPTURE-GPU] {camera_id}: started ffmpeg CUVID  {w}x{h}  @{_DISPLAY_FPS}fps")
        with cam_state.lock:
            cam_state.last_error = ""

        # Single-slot latest-frame buffer shared with reader thread
        latest_lock = threading.Lock()
        latest_frame: list = [None]   # mutable container: [np.ndarray | None]
        reader_alive = [True]

        def _reader():
            """Continuously reads frames from pipe, keeping only the newest."""
            try:
                while reader_alive[0]:
                    raw = proc.stdout.read(frame_bytes)
                    if len(raw) != frame_bytes:
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
                    with latest_lock:
                        latest_frame[0] = frame
            except Exception:
                pass
            finally:
                reader_alive[0] = False

        reader_t = threading.Thread(target=_reader, daemon=True, name=f"ffread-{camera_id}")
        reader_t.start()

        encode_interval = 1.0 / max(_DISPLAY_FPS, 1)
        try:
            while not st.stop_event.is_set() and not cam_stop.is_set():
                if not reader_alive[0]:
                    stderr = ""
                    try:
                        stderr = proc.stderr.read(500).decode(errors="replace") if proc.stderr else ""
                    except Exception:
                        pass
                    with cam_state.lock:
                        cam_state.last_error = f"GPU decode stopped: {stderr[:120]}" if stderr else "GPU decode stopped"
                    break

                # Grab latest frame (non-blocking)
                with latest_lock:
                    frame = latest_frame[0]
                    latest_frame[0] = None  # consumed

                if frame is not None:
                    try:
                        ts = time.time()
                        # YOLO overlay on display frame (clean frame stays for VLM)
                        display = frame.copy()
                        if cfg.yolo_enabled:
                            with cam_state.lock:
                                yb = cam_state.last_yolo_boxes
                            if yb:
                                for (bx1, by1, bx2, by2, cls_n, cf) in yb:
                                    cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                                    lbl = f"{cls_n} {cf:.0%}"
                                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    cv2.rectangle(display, (bx1, by1 - th - 6), (bx1 + tw + 4, by1), (0, 255, 0), -1)
                                    cv2.putText(display, lbl, (bx1 + 2, by1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        b64 = encode_jpeg_b64(display)
                        with cam_state.lock:
                            cam_state.last_frame_b64 = b64
                            cam_state.last_frame_ts = ts
                            cam_state.last_frame_np = frame  # clean frame for VLM
                            if cam_state.last_error:
                                cam_state.last_error = ""
                            # Ring buffer for video clips (clean frame)
                            if ts - cam_state.frame_ring_last_ts >= 1.0 / max(cfg.video_fps, 1):
                                _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                                cam_state.frame_ring.append((ts, jpg.tobytes()))
                                cam_state.frame_ring_last_ts = ts
                    except Exception as e:
                        with cam_state.lock:
                            cam_state.last_error = f"Frame encode failed: {e}"

                time.sleep(encode_interval)
        finally:
            reader_alive[0] = False
            proc.stdout.close()
            proc.stderr.close()
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            reader_t.join(timeout=2)

        time.sleep(1.0)


def _capture_cv2(camera_id: str, rtsp_url: str, cam_state: CameraState) -> None:
    """Capture loop using OpenCV VideoCapture (CPU decode fallback)."""
    cam_stop = cam_state.capture_stop_event

    while not st.stop_event.is_set() and not cam_stop.is_set():
        cap = _build_cv2_capture(rtsp_url)
        if not cap.isOpened():
            with cam_state.lock:
                cam_state.last_error = f"RTSP open failed: {rtsp_url}"
            time.sleep(2.0)
            continue

        print(f"[CAPTURE-CPU] {camera_id}: opened via OpenCV")
        with cam_state.lock:
            cam_state.last_error = ""

        try:
            fail_count = 0
            while not st.stop_event.is_set() and not cam_stop.is_set():
                ret, frame = _read_latest_frame(cap)
                if not ret or frame is None:
                    fail_count += 1
                    if fail_count >= 3:
                        with cam_state.lock:
                            cam_state.last_error = "RTSP read failed, reconnecting..."
                        break
                    time.sleep(0.01)
                    continue
                fail_count = 0
                try:
                    ts = time.time()
                    # YOLO overlay on display frame (clean frame stays for VLM)
                    display = frame.copy()
                    if cfg.yolo_enabled:
                        with cam_state.lock:
                            yb = cam_state.last_yolo_boxes
                        if yb:
                            for (bx1, by1, bx2, by2, cls_n, cf) in yb:
                                cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                                lbl = f"{cls_n} {cf:.0%}"
                                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(display, (bx1, by1 - th - 6), (bx1 + tw + 4, by1), (0, 255, 0), -1)
                                cv2.putText(display, lbl, (bx1 + 2, by1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    b64 = encode_jpeg_b64(display)
                    with cam_state.lock:
                        cam_state.last_frame_b64 = b64
                        cam_state.last_frame_ts = ts
                        cam_state.last_frame_np = frame  # clean frame for VLM
                        if cam_state.last_error.startswith("RTSP"):
                            cam_state.last_error = ""
                        # Ring buffer for video clips (clean frame)
                        if ts - cam_state.frame_ring_last_ts >= 1.0 / max(cfg.video_fps, 1):
                            _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                            cam_state.frame_ring.append((ts, jpg.tobytes()))
                            cam_state.frame_ring_last_ts = ts
                except Exception as e:
                    with cam_state.lock:
                        cam_state.last_error = f"Frame encode failed: {e}"
                time.sleep(0.01)
        finally:
            cap.release()

        time.sleep(1.0)


def capture_worker(camera_id: str, rtsp_url: str, cam_state: CameraState) -> None:
    """Per-camera RTSP capture thread. Uses GPU if available, CPU fallback."""
    if not rtsp_url.strip():
        return
    if _FFMPEG_HAS_CUDA:
        _capture_gpu(camera_id, rtsp_url, cam_state)
    else:
        _capture_cv2(camera_id, rtsp_url, cam_state)


# ──────────────────────────────────────────────────────────────
# YOLO pre-filter
# ──────────────────────────────────────────────────────────────
_yolo_model = None
_yolo_model_lock = threading.Lock()


def _get_yolo_model():
    """Lazy-load YOLO11n model (downloads ~6 MB on first call)."""
    global _yolo_model
    if _yolo_model is None:
        with _yolo_model_lock:
            if _yolo_model is None:
                from ultralytics import YOLO  # noqa: PLC0415
                _yolo_model = YOLO("yolo11n.pt")
                print("[YOLO] yolo11n.pt loaded")
    return _yolo_model


def _yolo_filter(frame_np, target_classes: list, confidence: float) -> tuple:
    """Return (should_infer: bool, result_str: str, boxes: list).
    should_infer=True when at least one target class is detected above threshold.
    boxes: list of (x1,y1,x2,y2,class_name,conf) for ALL detected objects."""
    if frame_np is None or not target_classes:
        return True, "", []
    try:
        model = _get_yolo_model()
        results = model(frame_np, verbose=False, conf=confidence)
        hits = []
        boxes = []
        for r in results:
            for box in r.boxes:
                name = model.names[int(box.cls)]
                conf_val = box.conf.item()
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                boxes.append((x1, y1, x2, y2, name, conf_val))
                if name in target_classes:
                    hits.append(f"{name}:{conf_val:.2f}")
        if hits:
            return True, ", ".join(hits), boxes
        return False, "skip", boxes
    except Exception as e:
        print(f"[YOLO] filter error: {e}")
        return True, "", []   # fail-open: allow VLM if YOLO errors


import re as _re

def _parse_structured_output(reply: str) -> dict:
    """Extract danger_level from VLM reply. Supports [DANGER:X] tag and legacy JSON.
    Returns dict with at least 'danger_level' key if found."""
    if not reply:
        return {}
    # Priority 1: [DANGER:X] tag (new natural-language format)
    m = _re.search(r'\[DANGER:\s*(\d+)\s*\]', reply, _re.IGNORECASE)
    if m:
        return {"danger_level": int(m.group(1))}
    # Priority 2: Legacy JSON format (backwards compatibility)
    try:
        m = _re.search(r'```json\s*(\{.*?\})\s*```', reply, _re.DOTALL)
        if not m:
            m = _re.search(r'\{[^{}]*"danger_level"[^{}]*\}', reply, _re.DOTALL)
        if not m:
            m = _re.search(r'\{.*\}', reply, _re.DOTALL)
        if m:
            return json.loads(m.group(1) if m.lastindex else m.group())
    except Exception:
        pass
    return {}


def _encode_clip(pre_frames, cam_state, cam_id, post_sec, video_fps, out_path):
    """Background thread: encode pre-buffer + post-buffer frames to MP4."""
    out = None
    try:
        if pre_frames:
            _, sample = pre_frames[0]
            arr = cv2.imdecode(np.frombuffer(sample, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return
            h, w = arr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, video_fps, (w, h))
            for _, jpg in pre_frames:
                f = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if f is not None:
                    out.write(f)
        # Continue recording post-buffer
        deadline = time.monotonic() + post_sec
        while time.monotonic() < deadline:
            with cam_state.lock:
                if cam_state.record_until > time.monotonic():
                    deadline = cam_state.record_until
                if cam_state.frame_ring:
                    _, jpg = cam_state.frame_ring[-1]
                else:
                    jpg = None
            if jpg and out:
                f = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if f is not None:
                    out.write(f)
            time.sleep(1.0 / max(video_fps, 1))
    except Exception as e:
        print(f"[CLIP] encode error for {cam_id}: {e}")
    finally:
        if out:
            out.release()
        with cam_state.lock:
            cam_state.recording = False
            cam_state.last_clip_path = out_path
            cam_state.last_clip_ts = time.time()
        print(f"[CLIP] saved {out_path}")


def _start_clip_recording(cam_state, cfg_obj, cam_id):
    """Trigger video clip recording from pre-buffer."""
    with cam_state.lock:
        if cam_state.recording:
            cam_state.record_until = time.monotonic() + cfg_obj.video_post_sec
            return  # extend existing recording
        pre_count = cfg_obj.video_pre_sec * cfg_obj.video_fps
        pre_frames = list(cam_state.frame_ring)[-pre_count:]
        cam_state.recording = True
        cam_state.record_until = time.monotonic() + cfg_obj.video_post_sec
    clip_dir = Path(cfg_obj.video_clip_dir) / cam_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(clip_dir / f"{ts_str}_{cam_id}.mp4")
    t = threading.Thread(target=_encode_clip,
                         args=(pre_frames, cam_state, cam_id, cfg_obj.video_post_sec,
                               cfg_obj.video_fps, out_path), daemon=True)
    with cam_state.lock:
        cam_state.record_thread = t
    t.start()


def infer_once_for_camera(cam_cfg: CameraConfig) -> None:
    """Run one VLM inference cycle for the given camera."""
    cam_id = cam_cfg.camera_id
    cam_state = st.cameras.get(cam_id)
    if cam_state is None:
        return

    with cam_state.lock:
        frame_np = cam_state.last_frame_np
        prev_alert_active = cam_state.alert_active
        prev_alert_ts = cam_state.alert_ts
        cam_state.last_infer_ts = time.monotonic()

    if frame_np is None:
        return
    # Always encode clean frame for VLM (no YOLO overlay)
    image_b64 = encode_jpeg_b64(frame_np)

    # ── YOLO pre-filter ───────────────────────────────────────
    if cfg.yolo_enabled and cfg.yolo_classes:
        should_infer, yolo_result, yolo_boxes = _yolo_filter(frame_np, cfg.yolo_classes, cfg.yolo_confidence)
        with cam_state.lock:
            cam_state.last_yolo_result = yolo_result
            cam_state.last_yolo_boxes = yolo_boxes
        if not should_infer:
            return
    # ──────────────────────────────────────────────────────────

    # Resolve effective prompt: scenario > per-cam > global
    scenario = FACTORY_SCENARIOS.get(cfg.scenario)
    use_structured = scenario is not None and scenario.id != "custom"
    if use_structured:
        base_prompt = scenario.get_prompt(cfg.response_language)
    elif cam_cfg.prompt:
        base_prompt = cam_cfg.prompt
    else:
        base_prompt = cfg.prompt
    effective_prompt = _apply_language(base_prompt, cfg.response_language)
    effective_snapshot = (
        cam_cfg.snapshot_enabled if cam_cfg.snapshot_enabled is not None else cfg.snapshot_enabled
    )
    snap_dir = _camera_snapshot_dir(cfg.snapshot_dir, cam_id)

    temp_snapshot_path: Optional[Path] = None
    final_snapshot_path: Optional[Path] = None
    temp_snapshot_error = ""
    if effective_snapshot:
        try:
            final_snapshot_path, temp_snapshot_path = _save_temp_snapshot(image_b64, snap_dir)
        except Exception as e:
            temp_snapshot_error = f"Snapshot temp failed: {e}"

    t0 = time.time()
    try:
        reply = ollama_vlm_chat(cfg.ollama_url, cfg.model, effective_prompt, [image_b64])
        infer_ms = int((time.time() - t0) * 1000)

        # Alert detection
        alert_active = False
        alert_reason = ""
        alert_ts = prev_alert_ts if prev_alert_active else None
        danger_level = 0
        event_type = ""

        if use_structured and reply:
            parsed = _parse_structured_output(reply)
            danger_level = int(parsed.get("danger_level", 0))
            thr = cfg.trigger_danger_level or (scenario.trigger_level if scenario else 5)
            if danger_level >= thr:
                alert_active = True
                # Use first 80 chars of reply (strip the DANGER tag) as alert reason
                short = _re.sub(r'\[DANGER:\s*\d+\s*\]', '', reply).strip()[:80]
                alert_reason = f"[L{danger_level}] {short}"
                alert_ts = (alert_ts or time.time())
                if cfg.video_clip_enabled:
                    _start_clip_recording(cam_state, cfg, cam_id)
        elif cfg.version == "1.1":
            alert_active, alert_reason = _check_alert(reply, cfg.alert_keywords)
            alert_ts = (alert_ts or time.time()) if alert_active else None

        # Snapshot handling
        snapshot_path = ""
        snapshot_text_path = ""
        snapshot_ts = None
        snapshot_error = ""
        if effective_snapshot:
            if _match_snapshot(reply, cfg.snapshot_keyword):
                if temp_snapshot_error:
                    snapshot_error = temp_snapshot_error
                else:
                    try:
                        saved = _finalize_snapshot(temp_snapshot_path, final_snapshot_path)
                        snapshot_path = str(saved)
                        snapshot_text_path = str(_save_snapshot_text(saved, reply))
                        snapshot_ts = time.time()
                    except Exception as e:
                        snapshot_error = f"Snapshot finalize failed: {e}"
                        _discard_snapshot(temp_snapshot_path)
            else:
                _discard_snapshot(temp_snapshot_path)

        with cam_state.lock:
            cam_state.last_reply = reply
            cam_state.last_reply_ts = time.time()
            cam_state.last_infer_ms = infer_ms
            cam_state.alert_active = alert_active
            cam_state.alert_reason = alert_reason
            cam_state.alert_ts = alert_ts
            cam_state.last_danger_level = danger_level
            cam_state.last_event_type = event_type
            cam_state.last_snapshot_error = snapshot_error
            if snapshot_ts:
                cam_state.last_snapshot_ts = snapshot_ts
                cam_state.last_snapshot_path = snapshot_path
                cam_state.last_snapshot_text_path = snapshot_text_path
            if cam_state.last_error.startswith("Ollama"):
                cam_state.last_error = ""

        _update_signal_light()

    except Exception as e:
        _discard_snapshot(temp_snapshot_path)
        with cam_state.lock:
            cam_state.last_error = f"Ollama call failed: {e}"


def _infer_zone(zone_name: str, zone_cams: List[CameraConfig]) -> None:
    """Run one multi-view inference for all cameras in a zone."""
    # Gather frames + metadata from all zone cameras
    images: list[str] = []
    cam_labels: list[tuple[str, str]] = []
    cam_states_in_zone: list[tuple[CameraConfig, CameraState]] = []

    for cam_cfg in zone_cams:
        cam_state = st.cameras.get(cam_cfg.camera_id)
        if cam_state is None:
            continue
        with cam_state.lock:
            frame_np = cam_state.last_frame_np  # clean frame (no YOLO overlay)
            cam_state.last_infer_ts = time.monotonic()
        if frame_np is None:
            continue
        images.append(encode_jpeg_b64(frame_np))
        cam_labels.append((cam_cfg.camera_id, cam_cfg.label))
        cam_states_in_zone.append((cam_cfg, cam_state))

    if not images:
        return

    # ── YOLO pre-filter for zone (OR logic: any camera hit → infer) ───
    if cfg.yolo_enabled and cfg.yolo_classes and cam_states_in_zone:
        zone_hit = False
        for cam_cfg_z, cam_state_z in cam_states_in_zone:
            with cam_state_z.lock:
                fnp = cam_state_z.last_frame_np
            hit, yr, yb = _yolo_filter(fnp, cfg.yolo_classes, cfg.yolo_confidence)
            with cam_state_z.lock:
                cam_state_z.last_yolo_result = yr
                cam_state_z.last_yolo_boxes = yb
            if hit:
                zone_hit = True
        if not zone_hit:
            return
    # ──────────────────────────────────────────────────────────────────

    # Resolve prompt: scenario > per-cam > zone template
    first_cam = zone_cams[0]
    scenario = FACTORY_SCENARIOS.get(cfg.scenario)
    use_structured = scenario is not None and scenario.id != "custom"
    if use_structured:
        # For multi-cam scenarios, inject cam_count/cam_labels into prompt if placeholders exist
        base_prompt = scenario.get_prompt(cfg.response_language)
        if "{cam_count}" in base_prompt or "{cam_labels}" in base_prompt:
            label_lines = "\n".join(
                f"- Image[{i}] = Camera \"{cid}\" ({lbl or cid})"
                for i, (cid, lbl) in enumerate(cam_labels)
            )
            base_prompt = base_prompt.replace("{cam_count}", str(len(cam_labels)))
            base_prompt = base_prompt.replace("{cam_labels}", label_lines)
        prompt = _apply_language(base_prompt, cfg.response_language)
    elif first_cam.prompt:
        prompt = _apply_language(first_cam.prompt, cfg.response_language)
    elif cfg.prompt:
        prompt = _apply_language(cfg.prompt, cfg.response_language)
    else:
        prompt = _apply_language(_build_zone_prompt(zone_name, cam_labels), cfg.response_language)

    # Snapshot: use the first camera's snapshot dir for the zone
    effective_snapshot = (
        first_cam.snapshot_enabled if first_cam.snapshot_enabled is not None else cfg.snapshot_enabled
    )
    snap_dir = _camera_snapshot_dir(cfg.snapshot_dir, f"zone_{zone_name}")

    temp_snapshot_path: Optional[Path] = None
    final_snapshot_path: Optional[Path] = None
    temp_snapshot_error = ""
    if effective_snapshot and images:
        try:
            final_snapshot_path, temp_snapshot_path = _save_temp_snapshot(images[0], snap_dir)
        except Exception as e:
            temp_snapshot_error = f"Snapshot temp failed: {e}"

    t0 = time.time()
    try:
        reply = ollama_vlm_chat(cfg.ollama_url, cfg.model, prompt, images)
        infer_ms = int((time.time() - t0) * 1000)

        # Alert / structured output detection
        alert_active = False
        alert_reason = ""
        danger_level = 0
        event_type = ""

        if use_structured and reply:
            parsed = _parse_structured_output(reply)
            danger_level = int(parsed.get("danger_level", 0))
            thr = cfg.trigger_danger_level or (scenario.trigger_level if scenario else 5)
            if danger_level >= thr:
                alert_active = True
                short = _re.sub(r'\[DANGER:\s*\d+\s*\]', '', reply).strip()[:80]
                alert_reason = f"[L{danger_level}] {short}"
                # Trigger video clip for each camera in zone
                if cfg.video_clip_enabled:
                    for cc_z, cs_z in cam_states_in_zone:
                        _start_clip_recording(cs_z, cfg, cc_z.camera_id)
        elif cfg.version == "1.1":
            alert_active, alert_reason = _check_alert(reply, cfg.alert_keywords)

        alert_ts = time.time() if alert_active else None

        # Snapshot
        snapshot_path = ""
        snapshot_text_path = ""
        snapshot_ts = None
        snapshot_error = ""
        if effective_snapshot:
            if _match_snapshot(reply, cfg.snapshot_keyword):
                if temp_snapshot_error:
                    snapshot_error = temp_snapshot_error
                else:
                    try:
                        saved = _finalize_snapshot(temp_snapshot_path, final_snapshot_path)
                        snapshot_path = str(saved)
                        snapshot_text_path = str(_save_snapshot_text(saved, reply))
                        snapshot_ts = time.time()
                    except Exception as e:
                        snapshot_error = f"Snapshot finalize failed: {e}"
                        _discard_snapshot(temp_snapshot_path)
            else:
                _discard_snapshot(temp_snapshot_path)

        # Apply result to ALL cameras in the zone
        for cam_cfg_z, cam_state_z in cam_states_in_zone:
            with cam_state_z.lock:
                cam_state_z.last_reply = f"[Zone:{zone_name}] {reply}"
                cam_state_z.last_reply_ts = time.time()
                cam_state_z.last_infer_ms = infer_ms
                cam_state_z.alert_active = alert_active
                cam_state_z.alert_reason = alert_reason
                cam_state_z.alert_ts = alert_ts
                cam_state_z.last_danger_level = danger_level
                cam_state_z.last_event_type = event_type
                cam_state_z.last_snapshot_error = snapshot_error
                if snapshot_ts:
                    cam_state_z.last_snapshot_ts = snapshot_ts
                    cam_state_z.last_snapshot_path = snapshot_path
                    cam_state_z.last_snapshot_text_path = snapshot_text_path
                if cam_state_z.last_error.startswith("Ollama"):
                    cam_state_z.last_error = ""

        _update_signal_light()

    except Exception as e:
        _discard_snapshot(temp_snapshot_path)
        for _, cam_state_z in cam_states_in_zone:
            with cam_state_z.lock:
                cam_state_z.last_error = f"Zone inference failed: {e}"


def infer_worker() -> None:
    """
    Single shared inference thread.

    Two modes:
    - **Zone mode**: cameras sharing the same `zone` field are grouped; all
      their frames are sent in a single multi-image VLM call.
    - **Solo mode**: cameras without a zone are inferred individually using
      weighted round-robin.
    """
    while not st.stop_event.is_set():
        with st.global_lock:
            active_cams = [c for c in cameras_cfg if c.enabled and c.rtsp_url.strip()]

        if not active_cams:
            time.sleep(0.1)
            continue

        now = time.monotonic()

        # Separate into zones and solo cameras
        zones: Dict[str, List[CameraConfig]] = {}
        solo: List[CameraConfig] = []
        for cam_cfg in active_cams:
            if cam_cfg.zone.strip():
                zones.setdefault(cam_cfg.zone.strip(), []).append(cam_cfg)
            else:
                solo.append(cam_cfg)

        # Check which zones are ready (all cameras in zone past their interval)
        zone_ready: list[tuple[str, List[CameraConfig]]] = []
        for zone_name, zone_cams in zones.items():
            all_ready = True
            for cam_cfg in zone_cams:
                cam_state = st.cameras.get(cam_cfg.camera_id)
                if cam_state is None:
                    all_ready = False
                    break
                last_ts = cam_state.last_infer_ts or 0.0
                if now - last_ts < cam_cfg.interval_sec:
                    all_ready = False
                    break
            if all_ready:
                zone_ready.append((zone_name, zone_cams))

        # Check which solo cameras are ready
        solo_ready: List[CameraConfig] = []
        for cam_cfg in solo:
            cam_state = st.cameras.get(cam_cfg.camera_id)
            if cam_state is None:
                continue
            last_ts = cam_state.last_infer_ts or 0.0
            if now - last_ts >= cam_cfg.interval_sec:
                solo_ready.extend([cam_cfg] * cam_cfg.priority)

        if not zone_ready and not solo_ready:
            time.sleep(0.1)
            continue

        # Prefer zone inference (multi-view is more valuable), then solo
        if zone_ready:
            zone_name, zone_cams = random.choice(zone_ready)
            _infer_zone(zone_name, zone_cams)
        elif solo_ready:
            chosen = random.choice(solo_ready)
            infer_once_for_camera(chosen)


# ──────────────────────────────────────────────────────────────
# Thread lifecycle
# ──────────────────────────────────────────────────────────────

def _launch_capture_thread(cam_cfg: CameraConfig) -> None:
    """
    Start (or restart) the capture thread for a camera.
    If a previous capture thread exists, its stop event is signalled so it exits
    naturally. A new CameraState with a fresh stop event is installed, preserving
    accumulated inference data from the old state.
    """
    cam_id = cam_cfg.camera_id

    # Signal any existing capture thread to stop
    with st.global_lock:
        old_state = st.cameras.get(cam_id)

    if old_state is not None:
        old_state.capture_stop_event.set()

    # Build new CameraState (fresh stop event), carry over inference history
    new_state = CameraState()
    if old_state is not None:
        with old_state.lock:
            new_state.last_reply = old_state.last_reply
            new_state.last_reply_ts = old_state.last_reply_ts
            new_state.last_infer_ms = old_state.last_infer_ms
            new_state.last_infer_ts = old_state.last_infer_ts
            new_state.alert_active = old_state.alert_active
            new_state.alert_reason = old_state.alert_reason
            new_state.alert_ts = old_state.alert_ts

    with st.global_lock:
        st.cameras[cam_id] = new_state

    t = threading.Thread(
        target=capture_worker,
        args=(cam_id, cam_cfg.rtsp_url, new_state),
        daemon=True,
        name=f"capture-{cam_id}",
    )
    new_state.capture_thread = t
    t.start()


def start_threads() -> None:
    if st.running:
        return
    st.stop_event.clear()
    with st.global_lock:
        cams = list(cameras_cfg)
    for cam_cfg in cams:
        if cam_cfg.enabled and cam_cfg.rtsp_url.strip():
            _launch_capture_thread(cam_cfg)
    st.infer_thread = threading.Thread(target=infer_worker, daemon=True, name="infer")
    st.infer_thread.start()
    st.running = True


def stop_threads() -> None:
    if not st.running:
        return
    st.stop_event.set()
    with st.global_lock:
        cam_states = list(st.cameras.values())
    for cam_state in cam_states:
        cam_state.capture_stop_event.set()
    st.running = False


# ──────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────

app = FastAPI(title="VLM-APP", version="2.0")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(
        str(static_dir / "index.html"),
        headers={
            "Cache-Control": "no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


# ─── Global config ───────────────────────────────────────────

@app.post("/config")
def set_config(new_cfg: GlobalConfig):
    global cfg
    cfg = new_cfg
    cfg.ollama_url = _normalize_ollama_url(cfg.ollama_url)
    cfg.version = _normalize_version(cfg.version)
    cfg.alert_keywords = [kw.strip() for kw in cfg.alert_keywords if kw.strip()] \
        or GlobalConfig.model_fields["alert_keywords"].default_factory()
    cfg.snapshot_keyword = (cfg.snapshot_keyword or "").strip() \
        or GlobalConfig.model_fields["snapshot_keyword"].default
    cfg.snapshot_dir = (cfg.snapshot_dir or "").strip() \
        or GlobalConfig.model_fields["snapshot_dir"].default
    # Downgrading to v1.0 clears all camera alerts
    if cfg.version == "1.0":
        with st.global_lock:
            cam_states = list(st.cameras.values())
        for cam_state in cam_states:
            with cam_state.lock:
                cam_state.alert_active = False
                cam_state.alert_reason = ""
                cam_state.alert_ts = None
    _save_config()
    return JSONResponse({"ok": True, "config": cfg.model_dump()})


@app.get("/config")
def get_config():
    return JSONResponse(cfg.model_dump())


@app.patch("/config/language")
async def patch_response_language(request: Request):
    """Lightweight endpoint — only updates response_language (called on UI lang switch)."""
    body = await request.json()
    lang = (body.get("language") or "English").strip()
    cfg.response_language = lang
    return JSONResponse({"ok": True, "response_language": lang})


# ─── System control ──────────────────────────────────────────

@app.post("/start")
def start():
    start_threads()
    return JSONResponse({"ok": True, "running": st.running})


@app.post("/stop")
def stop():
    stop_threads()
    return JSONResponse({"ok": True, "running": st.running})


# ─── Camera CRUD ─────────────────────────────────────────────

@app.get("/cameras")
def list_cameras():
    with st.global_lock:
        cams = [c.model_dump(exclude={"onvif_password"}) for c in cameras_cfg]
    return JSONResponse({"cameras": cams})


@app.post("/cameras")
def add_camera(cam: CameraConfig):
    global cameras_cfg
    cam = cam.model_copy(update={
        "onvif_host": cam.onvif_host.strip(),
        "onvif_username": cam.onvif_username.strip(),
        "onvif_password": cam.onvif_password.strip(),
    })
    with st.global_lock:
        if any(c.camera_id == cam.camera_id for c in cameras_cfg):
            return JSONResponse({"detail": f"camera_id '{cam.camera_id}' already exists"}, status_code=409)
        cameras_cfg.append(cam)
    # Auto-start capture if the system is already running
    if st.running and cam.enabled and cam.rtsp_url.strip():
        _launch_capture_thread(cam)
    _save_config()
    return JSONResponse({"ok": True, "camera": cam.model_dump(exclude={"onvif_password"})})


@app.patch("/cameras/{camera_id}")
def update_camera(camera_id: str, updates: CameraUpdate):
    global cameras_cfg
    update_dict = updates.model_dump(exclude_unset=True)
    if not update_dict:
        return JSONResponse({"detail": "No fields to update"}, status_code=400)

    rtsp_changed = "rtsp_url" in update_dict
    enabled_changed = "enabled" in update_dict

    with st.global_lock:
        for i, cam in enumerate(cameras_cfg):
            if cam.camera_id == camera_id:
                updated = cam.model_copy(update=update_dict)
                cameras_cfg[i] = updated
                break
        else:
            return JSONResponse({"detail": f"Camera '{camera_id}' not found"}, status_code=404)

    # Restart capture thread if stream URL or enabled state changed
    if (rtsp_changed or enabled_changed) and st.running:
        if updated.enabled and updated.rtsp_url.strip():
            _launch_capture_thread(updated)
        else:
            cam_state = st.cameras.get(camera_id)
            if cam_state:
                cam_state.capture_stop_event.set()

    _save_config()
    return JSONResponse({"ok": True, "camera": updated.model_dump(exclude={"onvif_password"})})


@app.delete("/cameras/{camera_id}")
def delete_camera(camera_id: str):
    global cameras_cfg
    with st.global_lock:
        before = len(cameras_cfg)
        cameras_cfg = [c for c in cameras_cfg if c.camera_id != camera_id]
        if len(cameras_cfg) == before:
            return JSONResponse({"detail": f"Camera '{camera_id}' not found"}, status_code=404)
        cam_state = st.cameras.pop(camera_id, None)

    if cam_state:
        cam_state.capture_stop_event.set()
        cam_state.onvif_patrol_stop_event.set()
        cam_state.onvif_patrol_active = False

    _save_config()
    return JSONResponse({"ok": True})


@app.post("/cameras/{camera_id}/enable")
def toggle_camera_enable(camera_id: str, body: EnableBody):
    global cameras_cfg
    with st.global_lock:
        for i, cam in enumerate(cameras_cfg):
            if cam.camera_id == camera_id:
                updated = cam.model_copy(update={"enabled": body.enabled})
                cameras_cfg[i] = updated
                break
        else:
            return JSONResponse({"detail": f"Camera '{camera_id}' not found"}, status_code=404)

    if st.running:
        if body.enabled and updated.rtsp_url.strip():
            _launch_capture_thread(updated)
        else:
            cam_state = st.cameras.get(camera_id)
            if cam_state:
                cam_state.capture_stop_event.set()

    _save_config()
    return JSONResponse({"ok": True, "enabled": body.enabled})


# ─── Per-camera actions ───────────────────────────────────────

@app.post("/cameras/{camera_id}/oneshot")
def camera_oneshot(camera_id: str):
    cam_cfg = _get_camera_cfg(camera_id)
    if cam_cfg is None:
        return JSONResponse({"detail": f"Camera '{camera_id}' not found"}, status_code=404)
    threading.Thread(target=infer_once_for_camera, args=(cam_cfg,), daemon=True).start()
    return JSONResponse({"ok": True})


@app.post("/cameras/{camera_id}/alert/clear")
def camera_clear_alert(camera_id: str):
    cam_state = st.cameras.get(camera_id)
    if cam_state is None:
        return JSONResponse({"detail": f"Camera '{camera_id}' not found"}, status_code=404)
    with cam_state.lock:
        cam_state.alert_active = False
        cam_state.alert_reason = ""
        cam_state.alert_ts = None
    _update_signal_light()
    return JSONResponse({"ok": True})


@app.post("/signal_light/test")
async def test_signal_light(request: Request):
    """Manually test the signal light API (UI test button)."""
    body = await request.json()
    alert_on = bool(body.get("alert_on", True))
    threading.Thread(target=_notify_signal_light, args=(alert_on,), daemon=True).start()
    return JSONResponse({"ok": True, "alert_on": alert_on})


@app.post("/cameras/{camera_id}/onvif/patrol/start")
def camera_onvif_patrol_start(camera_id: str):
    if ONVIFCamera is None:
        return JSONResponse({"detail": "onvif-zeep not installed"}, status_code=500)
    cam_cfg = _get_camera_cfg(camera_id)
    if cam_cfg is None:
        return JSONResponse({"detail": f"Camera '{camera_id}' not found"}, status_code=404)
    if not cam_cfg.onvif_host:
        return JSONResponse({"detail": "ONVIF host not set for this camera"}, status_code=400)
    # Ensure camera state exists even if capture hasn't started
    with st.global_lock:
        if camera_id not in st.cameras:
            st.cameras[camera_id] = CameraState()
    start_onvif_patrol(camera_id)
    return JSONResponse({"ok": True, "patrol": True})


@app.post("/cameras/{camera_id}/onvif/patrol/stop")
def camera_onvif_patrol_stop(camera_id: str):
    stop_onvif_patrol(camera_id)
    return JSONResponse({"ok": True, "patrol": False})


# ─── Legacy / convenience endpoints ──────────────────────────

@app.post("/oneshot")
def oneshot():
    """Trigger immediate inference for all enabled cameras."""
    with st.global_lock:
        enabled = [c for c in cameras_cfg if c.enabled and c.rtsp_url.strip()]
    for cam_cfg in enabled:
        threading.Thread(target=infer_once_for_camera, args=(cam_cfg,), daemon=True).start()
    return JSONResponse({"ok": True, "triggered": len(enabled)})


@app.post("/alert/clear")
def clear_all_alerts():
    """Clear alerts for every camera at once."""
    with st.global_lock:
        cam_states = list(st.cameras.values())
    for cam_state in cam_states:
        with cam_state.lock:
            cam_state.alert_active = False
            cam_state.alert_reason = ""
            cam_state.alert_ts = None
    _update_signal_light()
    return JSONResponse({"ok": True})


# ─── Scenarios ───────────────────────────────────────────────

@app.get("/scenarios")
def list_scenarios():
    """List all available factory safety scenarios."""
    return JSONResponse([
        {"id": s.id, "name": s.name, "name_en": s.name_en, "name_ja": s.name_ja,
         "description": s.description, "desc_en": s.desc_en, "desc_ja": s.desc_ja,
         "prompt": s.prompt, "prompt_zh": s.prompt_zh, "prompt_ja": s.prompt_ja,
         "trigger_level": s.trigger_level, "pre_sec": s.pre_sec, "post_sec": s.post_sec}
        for s in FACTORY_SCENARIOS.values()
    ])


# ─── Video Clips ─────────────────────────────────────────────

@app.get("/video_clips")
def list_video_clips():
    """List recorded video clips, newest first."""
    clip_dir = Path(cfg.video_clip_dir)
    items = []
    if clip_dir.exists():
        for p in sorted(clip_dir.rglob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
            items.append({
                "name": p.name,
                "camera_id": p.parent.name,
                "ts": p.stat().st_mtime,
                "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
            })
    return JSONResponse({"items": items})


@app.get("/video_clips/{cam_id}/{filename}")
def serve_video_clip(cam_id: str, filename: str):
    """Serve a video clip file."""
    p = Path(cfg.video_clip_dir) / cam_id / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(str(p), media_type="video/mp4")


# ─── Snapshots ───────────────────────────────────────────────

@app.get("/snapshots")
def list_snapshots():
    """List all snapshots across all camera subdirectories, newest first."""
    base = _normalize_snapshot_dir(cfg.snapshot_dir)
    items = []
    if base.exists():
        for cam_dir in sorted(base.iterdir()):
            if not cam_dir.is_dir():
                continue
            cam_id = cam_dir.name
            for img_path in cam_dir.glob("snapshot_*.jpg"):
                text_path = img_path.with_suffix(".txt")
                items.append({
                    "camera_id": cam_id,
                    "name": img_path.name,
                    "image_url": f"/snapshots/{cam_id}/{img_path.name}",
                    "text_url": f"/snapshots/{cam_id}/{text_path.name}" if text_path.exists() else "",
                    "ts": img_path.stat().st_mtime,
                })
    items.sort(key=lambda x: x["ts"], reverse=True)
    return JSONResponse({"items": items})


@app.get("/snapshots/{camera_id}/{filename}")
def get_snapshot_file(camera_id: str, filename: str):
    path = _resolve_snapshot_file(camera_id, filename)
    if not path or not path.exists():
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(str(path))


# ─── Status ──────────────────────────────────────────────────

@app.get("/status")
def status():
    """Return per-camera state for all cameras plus global flags."""
    with st.global_lock:
        cam_ids = list(st.cameras.keys())
        cam_cfgs_map = {c.camera_id: c for c in cameras_cfg}

    cameras_status: dict = {}
    global_alert = False

    for cam_id in cam_ids:
        cam_state = st.cameras.get(cam_id)
        if cam_state is None:
            continue
        cam_cfg = cam_cfgs_map.get(cam_id)
        with cam_state.lock:
            if cam_state.alert_active:
                global_alert = True
            cameras_status[cam_id] = {
                "label": cam_cfg.label if cam_cfg else cam_id,
                "enabled": cam_cfg.enabled if cam_cfg else False,
                "rtsp_url": cam_cfg.rtsp_url if cam_cfg else "",
                "last_frame_b64": cam_state.last_frame_b64,
                "last_frame_ts": cam_state.last_frame_ts,
                "last_reply": cam_state.last_reply,
                "last_reply_ts": cam_state.last_reply_ts,
                "last_infer_ms": cam_state.last_infer_ms,
                "last_error": cam_state.last_error,
                "alert_active": cam_state.alert_active,
                "alert_reason": cam_state.alert_reason,
                "alert_ts": cam_state.alert_ts,
                "last_snapshot_path": cam_state.last_snapshot_path,
                "last_snapshot_text_path": cam_state.last_snapshot_text_path,
                "last_snapshot_ts": cam_state.last_snapshot_ts,
                "last_snapshot_error": cam_state.last_snapshot_error,
                "onvif_patrol_active": cam_state.onvif_patrol_active,
                "last_onvif_error": cam_state.last_onvif_error,
                "last_yolo_result": cam_state.last_yolo_result,
                "recording": cam_state.recording,
                "last_clip_path": cam_state.last_clip_path,
                "last_clip_ts": cam_state.last_clip_ts,
                "last_danger_level": cam_state.last_danger_level,
                "last_event_type": cam_state.last_event_type,
            }

    # Aggregate zone results
    zones_status: dict = {}
    with st.global_lock:
        zone_groups: Dict[str, List[str]] = {}
        for cam_cfg in cameras_cfg:
            z = cam_cfg.zone.strip() if cam_cfg.zone else ""
            if z:
                zone_groups.setdefault(z, []).append(cam_cfg.camera_id)

    for zone_name, cam_ids_in_zone in zone_groups.items():
        # Use the first camera's state that has a reply as the zone result
        for cid in cam_ids_in_zone:
            cs = cameras_status.get(cid)
            if cs and cs.get("last_reply"):
                # Strip the [Zone:xxx] prefix for clean display
                raw_reply = cs["last_reply"]
                prefix = f"[Zone:{zone_name}] "
                clean_reply = raw_reply[len(prefix):] if raw_reply.startswith(prefix) else raw_reply
                zones_status[zone_name] = {
                    "camera_ids": cam_ids_in_zone,
                    "last_reply": clean_reply,
                    "last_reply_ts": cs.get("last_reply_ts"),
                    "last_infer_ms": cs.get("last_infer_ms"),
                    "alert_active": cs.get("alert_active", False),
                    "alert_reason": cs.get("alert_reason", ""),
                }
                break
        if zone_name not in zones_status:
            zones_status[zone_name] = {
                "camera_ids": cam_ids_in_zone,
                "last_reply": "",
                "last_reply_ts": None,
                "last_infer_ms": None,
                "alert_active": False,
                "alert_reason": "",
            }

    return JSONResponse({
        "running": st.running,
        "global_alert": global_alert,
        "camera_count": len(cameras_status),
        "cameras": cameras_status,
        "zones": zones_status,
        "config": cfg.model_dump(),
    })
