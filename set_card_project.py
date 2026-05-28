# -*- coding: utf-8 -*-
"""
Set card recognition project.

Run in VSCode:
    python set_card_project.py

Optional command line mode:
    python set_card_project.py "E:/OneDrive - MSFT/DIP/数字图像处理大作业/第一批图片/set纸牌2/IMG_1.png"

Required packages:
    opencv-python, numpy, pillow
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


CARD_SIZE = (240, 360)  # width, height after perspective correction
MAX_PROCESS_SIDE = 1800
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

COLOR_NAMES = {"red": "红色", "green": "绿色", "blue": "蓝色"}
SHAPE_NAMES = {"oval": "椭圆", "diamond": "菱形", "squiggle": "波浪"}
FILL_NAMES = {"solid": "实心", "striped": "阴影", "open": "空心"}
SET_ATTRIBUTES = (
    ("number", "数目"),
    ("shape", "形状"),
    ("color", "颜色"),
    ("fill", "填充"),
)


@dataclass
class CardFeature:
    card_id: str
    number: int
    shape: str
    color: str
    fill: str
    box: List[List[float]]
    confidence: float

    # 将一张牌的属性压缩成“数量+颜色+形状+填充”的中文描述。
    def compact(self) -> str:
        return (
            f"{self.number}"
            f"{COLOR_NAMES.get(self.color, self.color)}"
            f"{SHAPE_NAMES.get(self.shape, self.shape)}"
            f"{FILL_NAMES.get(self.fill, self.fill)}"
        )


@dataclass
class AnalysisResult:
    image_path: str
    cards: List[CardFeature]
    sets: List[Tuple[str, str, str]]
    annotated_bgr: np.ndarray


# 读取包含中文路径的图片文件。
def imread_unicode(path: str | Path) -> np.ndarray:
    """Read images from paths containing Chinese or other non-ASCII text."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


# 保存图片到可能包含中文的路径。
def imwrite_unicode(path: str | Path, image_bgr: np.ndarray) -> None:
    path = Path(path)
    suffix = path.suffix or ".png"
    ok, data = cv2.imencode(suffix, image_bgr)
    if not ok:
        raise ValueError(f"无法保存图片：{path}")
    data.tofile(str(path))


# 按最大边长等比例缩小图片，提升大图处理速度。
def resize_by_max_side(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


# 将卷积核尺寸修正为奇数，保证有明确中心点。
def odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


# 将四个角点排序为左上、右上、右下、左下。
def order_points(points: np.ndarray) -> np.ndarray:
    """Return points in top-left, top-right, bottom-right, bottom-left order."""
    pts = points.astype("float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


# 按比例扩大旋转矩形，避免卡牌边缘被裁掉。
def expand_rotated_rect(rect: Tuple[Tuple[float, float], Tuple[float, float], float], factor: float) -> np.ndarray:
    (cx, cy), (w, h), angle = rect
    rect = ((cx, cy), (w * factor, h * factor), angle)
    return cv2.boxPoints(rect)


# 分割整幅图中的白色卡牌区域。
def build_card_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Separate bright card bodies from the dark tabletop."""
    # 卡牌主体通常是白色亮区域，背景为黑色桌面；先用灰度阈值抓主体。
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 亮度高、饱和度低的区域也属于白色牌面，用 HSV 条件补充 Otsu 的漏检。
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, sat, val = cv2.split(hsv)
    bright_low_saturation = ((val > 80) & (sat < 120)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(otsu, bright_low_saturation)
    # Keep the kernel conservative: some batches have very small black gaps
    # between cards, and a large closing kernel would merge a whole column.
    k = odd_kernel(max(image_bgr.shape[:2]) // 450, minimum=3)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return mask


# 检测整幅图中所有卡牌的四边形定位框。
def detect_card_boxes(image_bgr: np.ndarray) -> List[np.ndarray]:
    """Detect card quadrilaterals, including a repair pass for touching cards."""
    small, scale = resize_by_max_side(image_bgr, MAX_PROCESS_SIDE)
    mask = build_card_mask(small)
    boxes = boxes_from_card_mask(mask, scale, expand_factor=1.025)
    boxes.extend(split_large_card_components(mask, scale, boxes))

    # A second pass after erosion helps split cards that touch through a thin
    # bright bridge in the threshold image.
    split_k = odd_kernel(max(small.shape[:2]) // 120, minimum=9)
    split_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (split_k, split_k))
    split_mask = cv2.erode(mask, split_kernel, iterations=1)
    boxes.extend(boxes_from_card_mask(split_mask, scale, expand_factor=1.09))

    boxes = remove_duplicate_boxes(boxes)
    boxes = prune_oversized_boxes(boxes)
    # 最后按 3x4 的弱网格先验修复少量合并框或缺失框，并保证编号顺序稳定。
    return regularize_grid_boxes(sort_boxes_by_grid(boxes))


# 删除明显大于正常卡牌的异常检测框。
def prune_oversized_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    if len(boxes) <= 12:
        return boxes
    diagonals = np.array([np.linalg.norm(box[0] - box[2]) for box in boxes], dtype=np.float32)
    median_diag = float(np.median(diagonals))
    if median_diag <= 1:
        return boxes
    return [box for box, diag in zip(boxes, diagonals) if diag <= median_diag * 1.42]


# 将卡牌框按从上到下、从左到右排序。
def sort_boxes_by_grid(boxes: List[np.ndarray]) -> List[np.ndarray]:
    """Sort cards row-by-row so labels match visual reading order."""
    if not boxes:
        return []

    heights = []
    for box in boxes:
        ordered = order_points(box)
        top = np.linalg.norm(ordered[1] - ordered[0])
        left = np.linalg.norm(ordered[3] - ordered[0])
        heights.append(max(top, left))
    row_tolerance = float(np.median(heights)) * 0.45

    rows: List[dict] = []
    for box in sorted(boxes, key=lambda item: float(np.mean(item[:, 1]))):
        center_y = float(np.mean(box[:, 1]))
        placed = False
        for row in rows:
            if abs(center_y - row["center_y"]) <= row_tolerance:
                row["boxes"].append(box)
                row["center_y"] = float(np.mean([np.mean(item[:, 1]) for item in row["boxes"]]))
                placed = True
                break
        if not placed:
            rows.append({"center_y": center_y, "boxes": [box]})

    sorted_boxes: List[np.ndarray] = []
    for row in sorted(rows, key=lambda item: item["center_y"]):
        sorted_boxes.extend(sorted(row["boxes"], key=lambda item: float(np.mean(item[:, 0]))))
    return sorted_boxes


# 计算卡牌框的宽、高和中心坐标。
def box_measurements(box: np.ndarray) -> Tuple[float, float, float, float]:
    ordered = order_points(box)
    width = (np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])) / 2
    height = (np.linalg.norm(ordered[3] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[1])) / 2
    center_x = float(np.mean(ordered[:, 0]))
    center_y = float(np.mean(ordered[:, 1]))
    return float(width), float(height), center_x, center_y


# 按指定中心和尺寸重建一个方向与模板框一致的卡牌框。
def rebuild_box_like(box: np.ndarray, center_x: float, center_y: float, width: float, height: float) -> np.ndarray:
    ordered = order_points(box)
    u = ordered[1] - ordered[0]
    v = ordered[3] - ordered[0]
    if np.linalg.norm(u) < 1 or np.linalg.norm(v) < 1:
        u = np.array([1.0, 0.0], dtype=np.float32)
        v = np.array([0.0, 1.0], dtype=np.float32)
    else:
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
    center = np.array([center_x, center_y], dtype=np.float32)
    half_w = u * (width / 2)
    half_h = v * (height / 2)
    return np.array([center - half_w - half_h, center + half_w - half_h, center + half_w + half_h, center - half_w + half_h])


# 根据纵向中心将卡牌框分成多行。
def group_boxes_by_rows(boxes: Sequence[np.ndarray]) -> List[List[np.ndarray]]:
    if not boxes:
        return []
    heights = [box_measurements(box)[1] for box in boxes]
    row_tolerance = float(np.median(heights)) * 0.45
    rows: List[dict] = []
    for box in sorted(boxes, key=lambda item: box_measurements(item)[3]):
        _, _, _cx, center_y = box_measurements(box)
        placed = False
        for row in rows:
            if abs(center_y - row["center_y"]) <= row_tolerance:
                row["boxes"].append(box)
                row["center_y"] = float(np.mean([box_measurements(item)[3] for item in row["boxes"]]))
                placed = True
                break
        if not placed:
            rows.append({"center_y": center_y, "boxes": [box]})
    return [sorted(row["boxes"], key=lambda item: box_measurements(item)[2]) for row in sorted(rows, key=lambda item: item["center_y"])]


# 利用3行4列弱先验修复合并框和缺失框。
def regularize_grid_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    """Repair bottom-row merged/missing boxes using the regular card grid as a weak prior."""
    if len(boxes) < 8:
        return boxes

    rows = group_boxes_by_rows(boxes)
    if len(rows) < 2:
        return boxes

    widths = np.array([box_measurements(box)[0] for box in boxes], dtype=np.float32)
    heights = np.array([box_measurements(box)[1] for box in boxes], dtype=np.float32)
    median_width = float(np.median(widths))
    median_height = float(np.median(heights))
    if median_width <= 1 or median_height <= 1:
        return boxes

    # 判断某个框是否很可能由两张或多张牌粘连形成。
    def is_merged_box(box: np.ndarray) -> bool:
        width, height, _center_x, _center_y = box_measurements(box)
        return width > median_width * 1.45 or height > median_height * 1.35

    # 先剔除明显过宽或过高的合并框，再用正常卡牌估计每一列的中心位置。
    cleaned_rows = [[box for box in row if not is_merged_box(box)] for row in rows]
    column_count = 4 if max(len(row) for row in rows) >= 4 else max(len(row) for row in cleaned_rows)
    if column_count < 3:
        return boxes

    column_centers: List[Optional[float]] = []
    for column in range(column_count):
        centers = []
        for row in cleaned_rows:
            if len(row) != column_count:
                continue
            width, _height, center_x, _center_y = box_measurements(row[column])
            if width >= median_width * 0.88:
                centers.append(center_x)
        column_centers.append(float(np.median(centers)) if centers else None)

    if any(center is None for center in column_centers):
        all_centers = sorted(box_measurements(box)[2] for row in cleaned_rows for box in row)
        if len(all_centers) >= column_count:
            groups = np.array_split(np.array(all_centers, dtype=np.float32), column_count)
            for index, group in enumerate(groups):
                if column_centers[index] is None and len(group):
                    column_centers[index] = float(np.median(group))

    fixed_rows: List[List[np.ndarray]] = []
    for row in cleaned_rows:
        assigned: dict[int, np.ndarray] = {}
        for box in row:
            width, height, center_x, center_y = box_measurements(box)
            available = [(idx, center) for idx, center in enumerate(column_centers) if center is not None]
            if not available:
                continue
            column, target_x = min(available, key=lambda item: abs(center_x - float(item[1])))
            if target_x is None or abs(center_x - target_x) > median_width * 0.72:
                continue
            # 若检测框偏离列中心或尺寸偏小，就按当前行中心和中位宽高重建一个更规整的框。
            should_fix = width < median_width * 0.86 or height < median_height * 0.86
            if abs(center_x - target_x) > median_width * 0.20:
                should_fix = True
            candidate = rebuild_box_like(box, target_x, center_y, median_width, median_height) if should_fix else box
            if column not in assigned:
                assigned[column] = candidate
            else:
                old_width, old_height, old_x, _old_y = box_measurements(assigned[column])
                old_score = abs(old_x - target_x) + max(0.0, median_width * 0.75 - old_width) + max(0.0, median_height * 0.75 - old_height)
                new_score = abs(center_x - target_x) + max(0.0, median_width * 0.75 - width) + max(0.0, median_height * 0.75 - height)
                if new_score < old_score:
                    assigned[column] = candidate

        # 一行只缺一张牌时，用同列/同行几何先验补齐，避免批量时出现 11 张牌。
        if len(assigned) >= column_count - 1:
            row_center_y = float(np.median([box_measurements(box)[3] for box in assigned.values()]))
            for column, target_x in enumerate(column_centers):
                if column in assigned or target_x is None:
                    continue
                template = None
                for other_row in fixed_rows:
                    if len(other_row) > column:
                        template = other_row[column]
                        break
                if template is None:
                    template = next(iter(assigned.values()))
                assigned[column] = rebuild_box_like(template, target_x, row_center_y, median_width, median_height)

        fixed_row = [assigned[index] for index in sorted(assigned)]
        fixed_rows.append(fixed_row)

    regularized: List[np.ndarray] = []
    for row in fixed_rows:
        regularized.extend(row)
    return sort_boxes_by_grid(regularized)


# 将一个大的合并连通块按卡牌尺寸尝试切分成多个候选框。
def split_large_card_components(mask: np.ndarray, scale: float, reference_boxes: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Split a merged card block into a simple row-column grid when possible."""
    if not reference_boxes:
        return []

    widths = []
    heights = []
    for box in reference_boxes:
        b = order_points(box * scale)
        top = np.linalg.norm(b[1] - b[0])
        left = np.linalg.norm(b[3] - b[0])
        widths.append(min(top, left))
        heights.append(max(top, left))

    card_w = float(np.median(widths))
    card_h = float(np.median(heights))
    if card_w <= 1 or card_h <= 1:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = mask.shape[0] * mask.shape[1]
    boxes: List[np.ndarray] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < card_w * 1.35 and h < card_h * 1.35:
            continue

        cols = max(1, int(round(w / card_w)))
        rows = max(1, int(round(h / card_h)))
        if rows * cols < 2 or rows * cols > 12:
            continue

        cell_w = w / cols
        cell_h = h / rows
        cell_aspect = min(cell_w, cell_h) / max(cell_w, cell_h)
        if not (0.34 <= cell_aspect <= 0.78):
            continue

        for row in range(rows):
            for col in range(cols):
                x0 = int(round(x + col * cell_w))
                x1 = int(round(x + (col + 1) * cell_w))
                y0 = int(round(y + row * cell_h))
                y1 = int(round(y + (row + 1) * cell_h))
                roi = mask[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                if np.count_nonzero(roi) / roi.size < 0.42:
                    continue

                pad_x = cell_w * 0.015
                pad_y = cell_h * 0.015
                box = np.array(
                    [
                        [x0 + pad_x, y0 + pad_y],
                        [x1 - pad_x, y0 + pad_y],
                        [x1 - pad_x, y1 - pad_y],
                        [x0 + pad_x, y1 - pad_y],
                    ],
                    dtype=np.float32,
                )
                boxes.append(order_points(box / scale))
    return boxes


# 从卡牌主体二值掩膜中提取候选旋转矩形框。
def boxes_from_card_mask(mask: np.ndarray, scale: float, expand_factor: float) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = mask.shape[0] * mask.shape[1]
    boxes: List[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.012 or area > image_area * 0.28:
            continue

        rect = cv2.minAreaRect(contour)
        (w, h) = rect[1]
        if w <= 1 or h <= 1:
            continue

        short, long = sorted((w, h))
        aspect = short / long
        extent = area / (w * h)
        if not (0.34 <= aspect <= 0.78 and extent >= 0.55):
            continue

        box = expand_rotated_rect(rect, factor=expand_factor) / scale
        boxes.append(order_points(box))
    return boxes


# 合并重复检测到的卡牌框，避免同一张牌被编号两次。
def remove_duplicate_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    if not boxes:
        return []

    kept: List[np.ndarray] = []
    centers: List[np.ndarray] = []
    for box in boxes:
        center = np.mean(box, axis=0)
        diag = np.linalg.norm(box[0] - box[2])
        duplicate = False
        for old_center, old_box in zip(centers, kept):
            old_diag = np.linalg.norm(old_box[0] - old_box[2])
            if np.linalg.norm(center - old_center) < 0.12 * min(diag, old_diag):
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
            centers.append(center)
    return kept


# 将检测到的单张卡牌透视矫正为固定大小。
def warp_card(image_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Normalize one detected card to a fixed front-view image."""
    ordered = order_points(box)
    w, h = CARD_SIZE
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image_bgr, matrix, CARD_SIZE, flags=cv2.INTER_CUBIC)


# 提取单张卡牌内部的彩色图案掩膜。
def symbol_color_mask(card_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the inner card ROI and a binary mask for colored symbols."""
    h, w = card_bgr.shape[:2]
    # 去掉靠近卡牌边框的区域，减少圆角、阴影和黑色背景对图案分割的干扰。
    y0, y1 = int(h * 0.08), int(h * 0.92)
    x0, x1 = int(w * 0.10), int(w * 0.90)
    roi = card_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    white_level = np.percentile(gray, 82)
    # 彩色图案靠饱和度提取；较暗的蓝/紫边线可能饱和度不高，用灰度条件补充。
    colored_symbol = (sat > 26) & (val > 8)
    dark_symbol = (gray < white_level - 55) & (sat > 12) & (val > 8)
    raw = (colored_symbol | dark_symbol).astype(np.uint8) * 255

    # Suppress tiny colored noise while keeping thin stripes.
    raw = cv2.medianBlur(raw, 3)
    raw = remove_tall_edge_components(raw)
    return roi, raw


# 删除靠近ROI边缘的竖向背景/卡牌边缘干扰。
def remove_tall_edge_components(mask: np.ndarray) -> np.ndarray:
    """Remove dark table/card-edge regions that touch the ROI border."""
    cleaned = mask.copy()
    h, w = cleaned.shape[:2]
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats((cleaned > 0).astype(np.uint8), 8)
    for label in range(1, labels_count):
        x, y, bw, bh, area = stats[label]
        touches_edge = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        thin_vertical_edge = touches_edge and bh > h * 0.30 and bw < w * 0.18
        full_height_edge = bh > h * 0.75 and (x <= 2 or x + bw >= w - 2 or area > h * w * 0.02)
        if thin_vertical_edge or full_height_edge:
            cleaned[labels == label] = 0
    return cleaned


# 根据HSV色相投票识别图案颜色。
def classify_color(roi_bgr: np.ndarray, raw_mask: np.ndarray) -> Tuple[str, float]:
    """Classify symbol color by voting in HSV hue intervals."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    reliable = (raw_mask > 0) & (sat > 18) & (val > 8)
    pixels = hue[reliable]
    if pixels.size == 0:
        pixels = hue[raw_mask > 0]
    if pixels.size == 0:
        return "blue", 0.0

    red_low = int(np.count_nonzero(pixels <= 15))
    red_high = int(np.count_nonzero(pixels >= 165))
    green = int(np.count_nonzero((pixels >= 35) & (pixels <= 95)))
    blue = int(np.count_nonzero((pixels >= 96) & (pixels < 165)))
    median_value = float(np.median(val[reliable])) if np.count_nonzero(reliable) else 255.0

    # In these photos the blue ink can become very dark purple, whose hue wraps
    # close to 180 and would otherwise be counted as red. True red cards remain
    # much brighter and cluster around hue 0.
    if median_value < 90 and red_high > red_low * 2:
        blue += red_high
        red = red_low
    else:
        red = red_low + red_high

    counts = {"red": red, "green": green, "blue": blue}
    color = max(counts, key=counts.get)
    confidence = counts[color] / max(1, pixels.size)
    return color, confidence


# 提取图案连通轮廓，并把阴影线碎片合并成单个图案。
def connected_symbol_contours(raw_mask: np.ndarray) -> List[np.ndarray]:
    """Find one contour per printed symbol after merging broken mask fragments."""
    h, w = raw_mask.shape[:2]
    k = odd_kernel(min(h, w) // 28, minimum=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # 阴影牌的竖线会被分成很多细碎部分，闭运算和膨胀可以把同一图案连成整体。
    work = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    work = cv2.dilate(work, kernel, iterations=1)

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = h * w
    candidates: List[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, bw, bh = cv2.boundingRect(contour)
        if is_roi_edge_artifact(contour, raw_mask.shape):
            continue
        if area < roi_area * 0.001:
            continue
        if bw < w * 0.04 or bh < h * 0.025:
            continue
        candidates.append(contour)

    good = merge_symbol_fragments_by_row(candidates, raw_mask.shape)
    good = [
        contour
        for contour in good
        if not is_roi_edge_artifact(contour, raw_mask.shape)
        and cv2.contourArea(contour) >= roi_area * 0.004
        and cv2.boundingRect(contour)[2] >= w * 0.12
        and cv2.boundingRect(contour)[3] >= h * 0.035
    ]

    good.sort(key=lambda contour: cv2.boundingRect(contour)[1])
    return good[:3]


# 判断轮廓是否是ROI边界处的伪目标。
def is_roi_edge_artifact(contour: np.ndarray, mask_shape: Tuple[int, int]) -> bool:
    h, w = mask_shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    touches_outer_roi_edge = (y <= 2 or y + bh >= h - 2) and (x <= 2 or x + bw >= w - 2)
    return bool(touches_outer_roi_edge and bw > w * 0.75 and bh < h * 0.12)


# 将同一行的碎片轮廓合并，解决阴影图案被拆散的问题。
def merge_symbol_fragments_by_row(contours: Sequence[np.ndarray], mask_shape: Tuple[int, int]) -> List[np.ndarray]:
    """Merge broken pieces that belong to the same vertically arranged symbol."""
    if not contours:
        return []

    h, _ = mask_shape[:2]
    groups: List[dict] = []
    for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2):
        x, y, w, bh = cv2.boundingRect(contour)
        center_y = y + bh / 2
        placed = False
        for group in groups:
            tolerance = max(h * 0.12, max(group["height"], bh) * 0.55)
            if abs(center_y - group["center_y"]) <= tolerance:
                group["contours"].append(contour)
                centers = [
                    cv2.boundingRect(item)[1] + cv2.boundingRect(item)[3] / 2
                    for item in group["contours"]
                ]
                group["center_y"] = float(np.mean(centers))
                group["height"] = max(group["height"], bh)
                placed = True
                break
        if not placed:
            groups.append({"center_y": center_y, "height": bh, "contours": [contour]})

    merged: List[np.ndarray] = []
    for group in groups:
        group_contours = group["contours"]
        if len(group_contours) == 1:
            merged.append(group_contours[0])
            continue
        points = np.vstack(group_contours)
        merged.append(cv2.convexHull(points))
    return merged


# 根据轮廓几何特征判断形状类型。
def contour_shape_features(contour: np.ndarray) -> Tuple[str, float]:
    """Use contour geometry to distinguish diamond, squiggle and oval."""
    area = max(cv2.contourArea(contour), 1.0)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = max(w * h, 1)
    hull = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1.0)
    perimeter = max(cv2.arcLength(contour, True), 1.0)
    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)

    extent = area / rect_area
    solidity = area / hull_area
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    vertices = len(approx)

    # 菱形外接矩形利用率低且近似顶点少；波浪图案凸度/圆形度更低。
    if 4 <= vertices <= 6 and solidity > 0.90 and extent < 0.62:
        return "diamond", min(1.0, 0.65 + (0.62 - extent))
    if solidity < 0.91 or circularity < 0.33:
        return "squiggle", min(1.0, 0.70 + (0.91 - solidity))
    return "oval", min(1.0, 0.70 + max(0.0, extent - 0.55))


# 综合密度、纹理和中心区域特征判断填充类型。
def classify_fill(
    raw_mask: np.ndarray,
    contours: Sequence[np.ndarray],
    shape: str = "",
    roi_bgr: Optional[np.ndarray] = None,
) -> Tuple[str, float]:
    """Classify fill by combining density, center blankness and stripe texture."""
    if not contours:
        return "open", 0.0

    # 这里不用单一面积比例判断填充，因为空心轮廓、阴影竖线和实心区域在不同图片中亮度差异较大。
    densities = []
    core_densities = []
    dark_core_densities = []
    texture_scores = []
    stripe_runs = []
    for contour in contours:
        filled = np.zeros_like(raw_mask)
        cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
        inside = filled > 0
        if np.count_nonzero(inside) == 0:
            continue
        densities.append(float(np.count_nonzero((raw_mask > 0) & inside) / np.count_nonzero(inside)))
        core_densities.append(symbol_core_density(raw_mask, contour))
        if roi_bgr is not None:
            dark_core_densities.append(symbol_dark_core_density(roi_bgr, contour))
            texture_scores.append(symbol_vertical_texture_score(roi_bgr, contour))
        stripe_runs.append(count_vertical_projection_runs(raw_mask, contour))

    density = float(np.median(densities)) if densities else 0.0
    core_density = float(np.median(core_densities)) if core_densities else 0.0
    dark_core_density = float(np.median(dark_core_densities)) if dark_core_densities else 0.0
    texture_score = float(np.median(texture_scores)) if texture_scores else 0.0
    run_count = float(np.median(stripe_runs)) if stripe_runs else 0.0

    # 明显空心优先：中心几乎没有颜色且没有 Sobel 条纹时，避免被外轮廓投影误判为阴影。
    if shape == "diamond" and core_density < 0.04 and dark_core_density < 0.06 and texture_score < 80:
        return "open", min(1.0, 0.72 + (0.04 - core_density))
    if shape == "oval" and core_density < 0.04 and dark_core_density < 0.06 and texture_score < 80:
        return "open", min(1.0, 0.72 + (0.04 - core_density))
    # 阴影牌的内部竖线会产生强横向灰度梯度，Sobel X 对这类纹理最敏感。
    if texture_score >= 80 and density < 0.93:
        return "striped", min(1.0, 0.60 + texture_score / 600)
    if shape == "oval" and density < 0.84 and core_density >= 0.65 and dark_core_density < 0.55:
        return "striped", min(1.0, 0.58 + core_density / 3)
    # 实心牌中心区域颜色连续，中心密度或暗色中心密度会明显升高。
    if core_density >= 0.70 or dark_core_density >= 0.68:
        return "solid", min(1.0, max(core_density, dark_core_density))
    if shape == "oval" and core_density >= 0.18 and dark_core_density >= 0.30:
        return "solid", min(1.0, 0.62 + dark_core_density)
    if shape == "squiggle":
        if core_density >= 0.12:
            return "striped", min(1.0, 0.58 + core_density)
        return "open", min(1.0, 1.0 - density)
    if run_count >= 5 or core_density >= 0.08:
        return "striped", min(1.0, 0.48 + core_density + min(run_count, 12) / 45)
    return "open", min(1.0, 1.0 - density)


# 计算图案内部核心区域的前景密度。
def symbol_core_density(raw_mask: np.ndarray, contour: np.ndarray) -> float:
    """Foreground density near the symbol center, ignoring outer outlines."""
    filled = np.zeros_like(raw_mask)
    cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
    # 腐蚀后只保留图案内部区域，这样空心牌的边框不会被当作填充。
    k = odd_kernel(min(raw_mask.shape[:2]) // 8, minimum=9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.erode(filled, kernel, iterations=1) > 0
    core_area = np.count_nonzero(core)
    if core_area == 0:
        return 0.0
    return float(np.count_nonzero((raw_mask > 0) & core) / core_area)


# 计算图案核心区域中较暗墨迹的比例。
def symbol_dark_core_density(roi_bgr: np.ndarray, contour: np.ndarray) -> float:
    """Measure how much dark ink exists inside the symbol core."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    white_level = np.percentile(gray, 82)
    filled = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
    k = odd_kernel(min(gray.shape[:2]) // 8, minimum=9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.erode(filled, kernel, iterations=1) > 0
    core_area = np.count_nonzero(core)
    if core_area == 0:
        return 0.0
    dark = gray < white_level - 55
    return float(np.count_nonzero(dark & core) / core_area)


# 用Sobel横向梯度衡量竖向阴影线纹理强度。
def symbol_vertical_texture_score(roi_bgr: np.ndarray, contour: np.ndarray) -> float:
    """Use Sobel X response to detect dense vertical hatch lines."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    filled = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
    k = odd_kernel(min(gray.shape[:2]) // 18, minimum=7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = cv2.erode(filled, kernel, iterations=1) > 0
    if np.count_nonzero(inner) == 0:
        return 0.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    return float(np.mean(np.abs(grad_x)[inner]))


# 统计图案内部竖向投影的连续条纹段数。
def count_vertical_projection_runs(raw_mask: np.ndarray, contour: np.ndarray) -> int:
    """Estimate whether the symbol contains several vertical hatch strokes."""
    filled = np.zeros_like(raw_mask)
    cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
    k = odd_kernel(min(raw_mask.shape[:2]) // 18, minimum=7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = cv2.erode(filled, kernel, iterations=1)
    inside = ((raw_mask > 0) & (inner > 0)).astype(np.uint8)

    x, y, w, h = cv2.boundingRect(contour)
    crop = inside[y : y + h, x : x + w]
    if crop.size == 0:
        return 0

    projection = crop.sum(axis=0)
    threshold = max(3, int(crop.shape[0] * 0.10))
    active = projection > threshold
    runs = 0
    in_run = False
    for value in active:
        if value and not in_run:
            runs += 1
            in_run = True
        elif not value:
            in_run = False
    return runs


# 对多个图案轮廓的分类结果进行加权投票。
def vote(values: Sequence[Tuple[str, float]], default: str) -> Tuple[str, float]:
    if not values:
        return default, 0.0
    scores = {}
    for value, confidence in values:
        scores[value] = scores.get(value, 0.0) + confidence
    best = max(scores, key=scores.get)
    return best, scores[best] / max(1, len(values))


# 从一张矫正后的卡牌中提取完整属性。
def extract_card_feature(card_id: str, card_bgr: np.ndarray, box: np.ndarray) -> CardFeature:
    """Extract all four Set attributes from one perspective-corrected card."""
    roi, raw_mask = symbol_color_mask(card_bgr)
    contours = connected_symbol_contours(raw_mask)

    # Set 每张牌最多 3 个图案；若少量噪声导致数量异常，最终仍限制在 1~3。
    number = min(3, max(1, len(contours)))
    color, color_conf = classify_color(roi, raw_mask)

    shape_votes = [contour_shape_features(c) for c in contours]
    shape, shape_conf = vote(shape_votes, default="oval")

    fill, fill_conf = classify_fill(raw_mask, contours, shape, roi)
    # 置信度是规则特征的稳定程度估计，不是训练模型概率。
    confidence = float(np.clip(np.mean([color_conf, shape_conf, fill_conf, len(contours) / 3]), 0.0, 1.0))

    return CardFeature(
        card_id=card_id,
        number=number,
        shape=shape,
        color=color,
        fill=fill,
        box=box.astype(float).round(1).tolist(),
        confidence=round(confidence, 3),
    )


# 判断三张牌是否满足标准Set成组规则。
def is_valid_set(cards: Sequence[CardFeature]) -> bool:
    """Standard Set rule: each attribute is either all same or all different."""
    if len(cards) != 3:
        return False
    for attr, _label in SET_ATTRIBUTES:
        # 四个属性逐项独立判断；不要求“全同/全异”的模式在四个属性间一致。
        values = {getattr(card, attr) for card in cards}
        if len(values) not in (1, 3):
            return False
    return True


# 生成人类可读的成组规则说明。
def describe_set_rule(cards: Sequence[CardFeature]) -> str:
    parts = []
    for attr, label in SET_ATTRIBUTES:
        values = {getattr(card, attr) for card in cards}
        parts.append(f"{label}{'全同' if len(values) == 1 else '全异'}")
    return "，".join(parts)


# 遍历所有三张牌组合，找出全部合法Set。
def find_all_sets(cards: Sequence[CardFeature]) -> List[Tuple[str, str, str]]:
    """Enumerate all three-card combinations and keep the valid Set groups."""
    sets: List[Tuple[str, str, str]] = []
    for triple in itertools.combinations(cards, 3):
        if is_valid_set(triple):
            sets.append(tuple(card.card_id for card in triple))
    return sets


# 在原图上绘制卡牌框和编号。
def annotate_image(image_bgr: np.ndarray, cards: Sequence[CardFeature], sets: Sequence[Tuple[str, str, str]]) -> np.ndarray:
    """Draw card boxes and readable labels on the original image."""
    annotated = image_bgr.copy()

    line_width = max(3, int(max(image_bgr.shape[:2]) / 450))
    font_scale = max(0.8, max(image_bgr.shape[:2]) / 1800)
    for card in cards:
        box = np.array(card.box, dtype=np.int32)
        border_color = (255, 190, 0)
        cv2.polylines(annotated, [box], True, border_color, line_width)

        x = int(box[:, 0].min())
        y = int(box[:, 1].min())
        label = card.card_id
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(2, line_width // 2))
        label_w = text_size[0] + 18
        label_h = text_size[1] + baseline + 14
        label_x = max(0, min(x + 6, annotated.shape[1] - label_w - 1))
        label_y = max(0, min(y + 6, annotated.shape[0] - label_h - 1))
        # 白底编号比直接写黑字更清楚，尤其是黑色背景和深色图案附近。
        cv2.rectangle(
            annotated,
            (label_x, label_y),
            (label_x + label_w, label_y + label_h),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            annotated,
            (label_x, label_y),
            (label_x + label_w, label_y + label_h),
            (30, 30, 30),
            max(1, line_width // 3),
        )
        cv2.putText(
            annotated,
            label,
            (label_x + 9, label_y + text_size[1] + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness=max(2, line_width // 2),
            lineType=cv2.LINE_AA,
        )
    return annotated


# 执行单张图片的完整识别流程。
def analyze_image(path: str | Path) -> AnalysisResult:
    """Full recognition pipeline for one image path."""
    image = imread_unicode(path)
    boxes = detect_card_boxes(image)

    cards: List[CardFeature] = []
    for index, box in enumerate(boxes, start=1):
        # 先把每张牌单独透视矫正，再提取四个属性。
        card_bgr = warp_card(image, box)
        card_id = f"C{index:02d}"
        cards.append(extract_card_feature(card_id, card_bgr, box))

    sets = find_all_sets(cards)
    annotated = annotate_image(image, cards, sets)
    return AnalysisResult(str(path), cards, sets, annotated)


# 将识别结果整理成可保存的文本报告。
def result_to_text(result: AnalysisResult) -> str:
    lines = [
        f"图片：{result.image_path}",
        f"检测到卡牌：{len(result.cards)} 张",
        f"成组组合：{len(result.sets)} 组",
        "",
        "卡牌属性：",
    ]
    by_id = {card.card_id: card for card in result.cards}
    for card in result.cards:
        lines.append(f"{card.card_id}: {card.compact()}    置信度 {card.confidence:.2f}")

    lines.append("")
    lines.append("成组结果：")
    if not result.sets:
        lines.append("未发现满足条件的三张组合。")
    for idx, group in enumerate(result.sets, start=1):
        group_cards = [by_id[cid] for cid in group]
        desc = " | ".join(f"{cid}({by_id[cid].compact()})" for cid in group)
        lines.append(f"{idx:02d}. {desc}    [{describe_set_rule(group_cards)}]")
    return "\n".join(lines)


# 获取文件夹中所有支持格式的图片路径。
def image_files_in_folder(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


# 将单张识别结果保存为txt或json文件。
def save_analysis_report(result: AnalysisResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    payload = {
        "image_path": result.image_path,
        "cards": [asdict(card) for card in result.cards],
        "sets": result.sets,
        "text": result_to_text(result),
    }
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        output_path.write_text(result_to_text(result), encoding="utf-8")


class SetCardApp(tk.Tk):
    """Tkinter GUI wrapper for single-image recognition and batch review."""
    # 初始化窗口状态、变量和界面组件。
    def __init__(self) -> None:
        super().__init__()
        self.title("神奇形色牌 Set 识别")
        self.geometry("1240x760")
        self.minsize(1060, 680)

        self.image_path = tk.StringVar()
        self.status_text = tk.StringVar(value="请选择一张图片或一个图片文件夹。")
        self.current_result: Optional[AnalysisResult] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.batch_results: List[AnalysisResult] = []
        self.batch_index = -1
        self.batch_choice = tk.StringVar()

        self._build_style()
        self._build_ui()

    # 设置Tkinter主题、按钮、表格和标题样式。
    def _build_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TButton", padding=(10, 6))
        style.configure("Treeview", rowheight=28)
        style.configure("Header.TLabel", font=("Microsoft YaHei UI", 14, "bold"))

    # 构建GUI主体布局，包括工具栏、预览区、属性表和结果文本框。
    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(root)
        toolbar.pack(fill=tk.X)

        ttk.Label(toolbar, text="图片路径").pack(side=tk.LEFT)
        entry = ttk.Entry(toolbar, textvariable=self.image_path)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(toolbar, text="选择图片", command=self.choose_image).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="选择文件夹", command=self.choose_folder).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="识别当前", command=self.run_current).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="保存结果", command=self.save_current).pack(side=tk.LEFT)

        batchbar = ttk.Frame(root)
        batchbar.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(batchbar, text="批量结果").pack(side=tk.LEFT)
        self.batch_combo = ttk.Combobox(batchbar, textvariable=self.batch_choice, state="readonly", width=46)
        self.batch_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.batch_combo.bind("<<ComboboxSelected>>", self.on_batch_select)
        ttk.Button(batchbar, text="上一张", command=self.show_previous_batch_result).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(batchbar, text="下一张", command=self.show_next_batch_result).pack(side=tk.LEFT, padx=(0, 6))
        self.batch_count_text = tk.StringVar(value="未加载批量结果")
        ttk.Label(batchbar, textvariable=self.batch_count_text, width=18, anchor=tk.E).pack(side=tk.LEFT)

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(12, 8))

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.canvas = tk.Canvas(left, bg="#151515", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.refresh_preview())

        ttk.Label(right, text="识别出的卡牌", style="Header.TLabel").pack(anchor=tk.W)
        columns = ("id", "number", "shape", "color", "fill", "confidence")
        self.tree = ttk.Treeview(right, columns=columns, show="headings", height=10)
        headings = {
            "id": "编号",
            "number": "数目",
            "shape": "形状",
            "color": "颜色",
            "fill": "填充",
            "confidence": "置信度",
        }
        widths = {"id": 56, "number": 58, "shape": 70, "color": 70, "fill": 70, "confidence": 70}
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor=tk.CENTER)
        self.tree.pack(fill=tk.X, pady=(6, 12))

        ttk.Label(right, text="成组结果", style="Header.TLabel").pack(anchor=tk.W)
        self.result_text = tk.Text(right, height=18, wrap=tk.WORD, font=("Consolas", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        status = ttk.Label(root, textvariable=self.status_text, anchor=tk.W)
        status.pack(fill=tk.X)

    # 弹出文件选择框，选择单张图片并立即识别。
    def choose_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择待识别图片",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
        )
        if file_path:
            self.image_path.set(file_path)
            self.clear_batch_navigation()
            self.run_current()

    # 弹出文件夹选择框，进入批量识别流程。
    def choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        self.image_path.set(folder)
        self.run_batch(folder)

    # 根据当前路径判断执行单张识别还是文件夹批量识别。
    def run_current(self) -> None:
        path = self.image_path.get().strip()
        if not path:
            messagebox.showinfo("提示", "请先选择图片。")
            return
        if Path(path).is_dir():
            self.run_batch(path)
            return

        self.clear_batch_navigation()
        self.status_text.set("正在识别，请稍候...")
        threading.Thread(target=self._analyze_one_worker, args=(path,), daemon=True).start()

    # 对文件夹中的全部图片启动后台批量识别。
    def run_batch(self, folder: str) -> None:
        paths = image_files_in_folder(folder)
        if not paths:
            messagebox.showinfo("提示", "该文件夹中没有可识别的图片。")
            return
        self.clear_batch_navigation()
        self.status_text.set(f"批量识别 {len(paths)} 张图片，请稍候...")
        threading.Thread(target=self._batch_worker, args=(paths,), daemon=True).start()

    # 后台执行单张识别，避免GUI界面卡死。
    def _analyze_one_worker(self, path: str) -> None:
        try:
            result = analyze_image(path)
        except Exception as exc:  # GUI boundary
            self.after(0, lambda: self._show_error(exc))
            return
        self.after(0, lambda: self.show_result(result))

    # 后台执行批量识别，并把每张图的结果保存到内存和文本报告。
    def _batch_worker(self, paths: Sequence[Path]) -> None:
        # 批量结果既保存到文本，也保存在内存中，方便 GUI 用下拉框/上一张/下一张回看。
        reports = []
        results: List[AnalysisResult] = []
        started = time.strftime("%Y%m%d_%H%M%S")
        output_dir = paths[0].parent
        report_path = output_dir / f"set_batch_result_{started}.txt"

        try:
            for index, path in enumerate(paths, start=1):
                self.after(0, lambda i=index, n=len(paths), p=path: self.status_text.set(f"正在识别 {i}/{n}: {p.name}"))
                result = analyze_image(path)
                results.append(result)
                reports.append(result_to_text(result))
                reports.append("\n" + "=" * 70 + "\n")

            report_path.write_text("\n".join(reports), encoding="utf-8")
        except Exception as exc:
            self.after(0, lambda: self._show_error(exc))
            return

        # 回到GUI线程后刷新批量结果控件。
        def done() -> None:
            self.load_batch_results(results, report_path)

        self.after(0, done)

    # 清空批量浏览状态，避免单张识别时残留旧结果。
    def clear_batch_navigation(self) -> None:
        self.batch_results = []
        self.batch_index = -1
        self.batch_combo["values"] = []
        self.batch_choice.set("")
        self.batch_count_text.set("未加载批量结果")

    # 加载批量识别结果，并初始化下拉框和第一张预览。
    def load_batch_results(self, results: Sequence[AnalysisResult], report_path: Path) -> None:
        # Combobox 中显示每张图的摘要，选中后同步刷新左侧图像、右侧表格和文本结果。
        self.batch_results = list(results)
        if not self.batch_results:
            self.batch_index = -1
            self.batch_count_text.set("未加载批量结果")
            self.status_text.set(f"批量识别完成，报告已保存：{report_path}")
            return

        values = [
            f"{index:02d}. {Path(result.image_path).name}  |  卡牌 {len(result.cards)}  |  成组 {len(result.sets)}"
            for index, result in enumerate(self.batch_results, start=1)
        ]
        self.batch_combo["values"] = values
        self.show_batch_result(0, update_path=False)
        self.status_text.set(f"批量识别完成，共 {len(self.batch_results)} 张；报告已保存：{report_path}")

    # 显示批量结果中的指定图片。
    def show_batch_result(self, index: int, update_path: bool = False) -> None:
        if not self.batch_results:
            return
        index = max(0, min(index, len(self.batch_results) - 1))
        self.batch_index = index
        self.batch_combo.current(index)
        result = self.batch_results[index]
        if update_path:
            self.image_path.set(result.image_path)
        self.show_result(result, update_status=False)
        self.batch_count_text.set(f"{index + 1}/{len(self.batch_results)}")
        self.status_text.set(
            f"当前显示 {index + 1}/{len(self.batch_results)}：{Path(result.image_path).name}，"
            f"检测到 {len(result.cards)} 张，发现 {len(result.sets)} 组。"
        )

    # 响应批量结果下拉框选择事件。
    def on_batch_select(self, _event: object | None = None) -> None:
        index = self.batch_combo.current()
        if index >= 0:
            self.show_batch_result(index)

    # 在批量结果中切换到上一张图片。
    def show_previous_batch_result(self) -> None:
        if self.batch_results:
            self.show_batch_result(self.batch_index - 1)

    # 在批量结果中切换到下一张图片。
    def show_next_batch_result(self) -> None:
        if self.batch_results:
            self.show_batch_result(self.batch_index + 1)

    # 将识别结果同步显示到表格、文本框和左侧预览图。
    def show_result(self, result: AnalysisResult, update_status: bool = True) -> None:
        self.current_result = result
        self._fill_table(result.cards)
        self._fill_text(result_to_text(result))
        self.refresh_preview()
        if update_status:
            self.status_text.set(f"完成：检测到 {len(result.cards)} 张卡牌，发现 {len(result.sets)} 组成组。")

    # 刷新右侧卡牌属性表。
    def _fill_table(self, cards: Sequence[CardFeature]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for card in cards:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    card.card_id,
                    card.number,
                    SHAPE_NAMES.get(card.shape, card.shape),
                    COLOR_NAMES.get(card.color, card.color),
                    FILL_NAMES.get(card.fill, card.fill),
                    f"{card.confidence:.2f}",
                ),
            )

    # 刷新右下角文本结果区域。
    def _fill_text(self, text: str) -> None:
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)

    # 根据窗口大小刷新左侧标注图预览。
    def refresh_preview(self) -> None:
        if self.current_result is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="请选择图片开始识别",
                fill="#e5e5e5",
                font=("Microsoft YaHei UI", 18),
            )
            return

        image = cv2.cvtColor(self.current_result.annotated_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image)
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        # 按画布尺寸等比例缩放，避免大图溢出或小图被拉伸变形。
        scale = min(cw / pil.width, ch / pil.height)
        new_size = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
        pil = pil.resize(new_size, Image.Resampling.LANCZOS)

        self.preview_photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.preview_photo, anchor=tk.CENTER)

    # 保存当前识别结果，可输出文本、JSON或标注图片。
    def save_current(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("提示", "当前还没有识别结果。")
            return
        out_path = filedialog.asksaveasfilename(
            title="保存识别结果",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("JSON", "*.json"), ("PNG image", "*.png")],
        )
        if not out_path:
            return
        suffix = Path(out_path).suffix.lower()
        try:
            if suffix == ".png":
                imwrite_unicode(out_path, self.current_result.annotated_bgr)
            else:
                save_analysis_report(self.current_result, out_path)
        except Exception as exc:
            self._show_error(exc)
            return
        self.status_text.set(f"结果已保存：{out_path}")

    # 统一显示异常信息，供后台线程回调使用。
    def _show_error(self, exc: Exception) -> None:
        self.status_text.set("识别失败。")
        messagebox.showerror("错误", str(exc))


# 命令行模式下打印单张或文件夹批量识别结果。
def print_cli_result(path: str) -> None:
    if Path(path).is_dir():
        for image_path in image_files_in_folder(path):
            print(result_to_text(analyze_image(image_path)))
            print("=" * 70)
    else:
        print(result_to_text(analyze_image(path)))


# 程序入口：有命令行参数则走CLI，否则启动GUI。
def main() -> None:
    if len(sys.argv) > 1:
        print_cli_result(sys.argv[1])
        return
    app = SetCardApp()
    app.mainloop()


if __name__ == "__main__":
    main()
