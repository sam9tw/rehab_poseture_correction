"""
評估 TinyTFClassifier (v1_F*.pt) 在 test set 上的準確度。

架構、前處理、特徵維度與 yolo11_detect_person_trf.py 完全一致：
  - Backbone : mobilenet_v3_small.features → [N, 576, 7, 7]
  - ROI Align: output_size=(7,7), spatial_scale=7/224, aligned=True
  - 時序聚合 : TSN-style 30 段 max-pool（與訓練 cache 一致，自動沿用 npz）
  - 分類器   : DSConV + Transformer + Attention Pooling → 9 classes

用法（在 Code/ 底下執行）：
  python evaluate_test_set.py \
      --weights v1_F*.pt \
      --test_root data/test \
      --max_frames 30 \
      --batch_size 8 \
      --out_dir test_eval_results
"""

import os
import re
import csv
import json
import math
import hashlib
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.ops import roi_align
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from ultralytics import YOLO


# =========================================================
# 1. 與訓練腳本完全一致的常數
# =========================================================
LABELS = [
    "shoulder_abduction_left", "shoulder_abduction_right",
    "shoulder_flexion_left",   "shoulder_flexion_right",
    "shoulder_backward_left",  "shoulder_backward_right",
    "elbow_flexion_left",      "elbow_flexion_right",
    "shoulder_forward_elevation",
]
label2idx = {label: i for i, label in enumerate(LABELS)}

MBV3_CHANNELS = 576  # mobilenet_v3_small.features 輸出通道數


# =========================================================
# 2. 與訓練一致的前處理 / 池化工具
# =========================================================
def letterbox_224_with_params(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    s = 224.0 / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (224 - nh) // 2
    bottom = 224 - nh - top
    left = (224 - nw) // 2
    right = 224 - nw - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, s, left, top


def make_cache_path(video_path, do_flip, meta: dict, cache_root="cache_feats"):
    os.makedirs(cache_root, exist_ok=True)
    key_str = (
        json.dumps(meta, sort_keys=True)
        + "|" + os.path.abspath(video_path)
        + f"|flip={int(do_flip)}"
    )
    h = hashlib.md5(key_str.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(cache_root, f"{stem}-{h}.npz")


def pool_chunk(chunk: torch.Tensor, mode: str = "max") -> torch.Tensor:
    if mode == "mean":
        return chunk.mean(dim=0)
    if mode == "max":
        return torch.amax(chunk, dim=0)
    if mode == "topk":
        k = max(1, int(0.3 * chunk.size(0)))
        return torch.topk(chunk, k, dim=0).values.mean(dim=0)
    if mode == "lse":
        return torch.logsumexp(chunk, dim=0) - math.log(chunk.size(0))
    raise ValueError(mode)


# =========================================================
# 3. 與訓練一致的模型架構（DSConV + TinyTFClassifier）
# =========================================================
class DSConV(nn.Module):
    def __init__(self, c=MBV3_CHANNELS, mid=256, act=nn.GELU, p=0.1):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)
        self.pw = nn.Conv2d(c, mid, kernel_size=1)
        self.bn = nn.GroupNorm(16, mid)
        self.act = act()
        self.drop = nn.Dropout(p)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):  # x: [B,c,7,7]
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.gap(x)
        return x.squeeze(-1).squeeze(-1)


class TinyTFClassifier(nn.Module):
    def __init__(self, mid=256, d_model=256, nhead=4, num_layers=2,
                 dim_ff=1024, num_classes=9, dropout=0.1,
                 use_delta=True, max_len=300):
        super().__init__()
        self.use_delta = use_delta
        self.roi_head = DSConV(c=MBV3_CHANNELS, mid=mid)
        in_dim = mid * 2 if use_delta else mid
        self.proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.att = nn.Sequential(
            nn.Linear(d_model, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.roi_head(x)
            x = x.view(B, T, -1)
        elif x.ndim != 3:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # 影片內標準化
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.std(dim=1, keepdim=True).clamp_min(1e-6)

        if self.use_delta:
            dx = torch.diff(x, dim=1, prepend=x[:, :1, :])
            x = torch.cat([x, dx], dim=2)

        x = self.proj(x)
        T = x.size(1)
        x = x + self.pos[:, :T, :]
        y = self.tf(x)
        w = torch.softmax(self.att(y).squeeze(-1), dim=1)
        h = (y * w.unsqueeze(-1)).sum(dim=1)
        return self.fc(h)


# =========================================================
# 4. Test set Dataset（不做翻轉、不做 cutout，純評估）
# =========================================================
class TestVideoDataset(Dataset):
    """
    與訓練 Dataset 共用 cache 邏輯，但：
      - 永遠 is_train=False、do_flip=False
      - cache_root 預設 data/video_features_v1（與訓練同 hash），讀過就能秒抽
    """

    def __init__(self, root_dir, transform_norm, yolo_model, feature_extractor,
                 max_frames=30, yolo_device="cpu",
                 cache_root="data/video_features_v1",
                 test_subject: str = "",
                 file_list=None):
        """
        file_list    : 若提供 [(path, label_idx), ...]，直接使用，不掃描目錄。
                       （由 data_split.json 載入時使用）
        test_subject : 舊版相容參數，指定受試者時從 root_dir/P003/label/ 掃檔。
        """
        self.samples: List[Tuple[str, int]] = []
        self.to_tensor_norm = transform_norm
        self.yolo = yolo_model
        self.feature_extractor = feature_extractor
        self.max_frames = max_frames
        self.yolo_device = yolo_device
        self.cache_root = cache_root

        if file_list is not None:
            # 直接使用指定清單（來自 data_split.json）
            self.samples = list(file_list)
            print(f"=== Test set 樣本統計（來源: data_split.json）===")
        else:
            # 舊版目錄掃描（相容原有 test_subject 參數）
            _subdirs = sorted(os.listdir(root_dir)) if os.path.isdir(root_dir) else []
            _subject_dirs = [
                d for d in _subdirs
                if os.path.isdir(os.path.join(root_dir, d)) and re.match(r'^P\d+$', d)
            ]
            if test_subject:
                if test_subject not in _subject_dirs:
                    print(f"[警告] 找不到受試者資料夾：{os.path.join(root_dir, test_subject)}")
                for label in LABELS:
                    label_dir = os.path.join(root_dir, test_subject, label)
                    if not os.path.isdir(label_dir):
                        print(f"[警告] 找不到類別資料夾：{label_dir}（跳過）")
                        continue
                    for fname in sorted(os.listdir(label_dir)):
                        if fname.lower().endswith((".mov", ".mp4", ".mkv", ".avi")):
                            self.samples.append(
                                (os.path.join(label_dir, fname), label2idx[label])
                            )
            elif _subject_dirs:
                for subj in _subject_dirs:
                    for label in LABELS:
                        label_dir = os.path.join(root_dir, subj, label)
                        if not os.path.isdir(label_dir):
                            continue
                        for fname in sorted(os.listdir(label_dir)):
                            if fname.lower().endswith((".mov", ".mp4", ".mkv", ".avi")):
                                self.samples.append(
                                    (os.path.join(label_dir, fname), label2idx[label])
                                )
            else:
                for label in LABELS:
                    d = os.path.join(root_dir, label)
                    if not os.path.isdir(d):
                        print(f"[警告] 找不到類別資料夾：{d}（跳過）")
                        continue
                    for fname in sorted(os.listdir(d)):
                        if fname.lower().endswith((".mov", ".mp4", ".mkv", ".avi")):
                            self.samples.append(
                                (os.path.join(d, fname), label2idx[label])
                            )
            subj_tag = test_subject if test_subject else "all"
            print(f"=== Test set 樣本統計（root={root_dir}, subject={subj_tag}）===")

        cnt_per_label = {i: 0 for i in range(len(LABELS))}
        for _, lb in self.samples:
            cnt_per_label[lb] += 1
        for i, name in enumerate(LABELS):
            print(f"  [{i}] {name:30s}: {cnt_per_label[i]}")
        print(f"  TOTAL = {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        simple_path = os.path.splitext(os.path.basename(video_path))[0]

        meta = {
            "T": self.max_frames,
            "backbone": "mbv3_small.features@224",
            "norm": "imagenet",
            "roi": {"type": "align", "bins": [7, 7],
                    "aligned": True, "sampling": "auto", "scale": "7/224"},
            "sampler": "uniform_ts+segment_max+nearest_fill",
            "cached_shape": [MBV3_CHANNELS, 7, 7],
            "head": "DSConv+GAP",
            "version": "v1",
        }
        cache_path = make_cache_path(
            video_path, do_flip=False, meta=meta, cache_root=self.cache_root
        )

        # 1) 讀 cache
        try:
            with np.load(cache_path, allow_pickle=False, mmap_mode="r") as z:
                arr = z["feat"]
                det_ratio = float(z["det_ratio"][0]) if "det_ratio" in z.files else float("nan")
                ok = (
                    (arr.ndim == 2 and arr.shape[1] in (MBV3_CHANNELS, 1920))
                    or (arr.ndim == 4 and arr.shape[1] == MBV3_CHANNELS and tuple(arr.shape[2:]) == (3, 3))
                    or (arr.ndim == 4 and arr.shape[1] == MBV3_CHANNELS and tuple(arr.shape[2:]) == (7, 7))
                )
                if ok:
                    return (
                        torch.from_numpy(arr.astype(np.float32, copy=False)),
                        int(label), simple_path, det_ratio,
                    )
        except Exception:
            pass

        # 2) 沒 cache → 重新抽特徵
        T = self.max_frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (
                torch.zeros((T, MBV3_CHANNELS, 7, 7), dtype=torch.float32),
                int(label), simple_path, 0.0,
            )

        feats_ts = []
        total = 0
        detected = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        use_fps = fps is not None and fps > 1e-3
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                total += 1
                ts_ms = (frame_idx * 1000.0 / float(fps)) if use_fps \
                    else float(cap.get(cv2.CAP_PROP_POS_MSEC))
                frame_idx += 1

                results = self.yolo(frame, device=self.yolo_device,
                                    verbose=False, classes=[0])
                if not results:
                    continue
                r0 = results[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is None or boxes.cls is None or boxes.xyxy is None:
                    continue
                cls = boxes.cls.detach().cpu().numpy()
                xyxy = boxes.xyxy.detach().cpu().numpy()

                for i in range(len(cls)):
                    if int(cls[i]) != 0:
                        continue
                    x1, y1, x2, y2 = map(float, xyxy[i])
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img224, s224, padL, padT = letterbox_224_with_params(rgb)

                    x1_224 = float(np.clip(x1 * s224 + padL, 0, 224))
                    y1_224 = float(np.clip(y1 * s224 + padT, 0, 224))
                    x2_224 = float(np.clip(x2 * s224 + padL, 0, 224))
                    y2_224 = float(np.clip(y2 * s224 + padT, 0, 224))

                    img_tensor = self.to_tensor_norm(img224).unsqueeze(0).to(
                        next(self.feature_extractor.parameters()).device
                    )
                    with torch.no_grad():
                        fmap = self.feature_extractor(img_tensor)
                    spatial_scale = fmap.shape[-1] / 224.0
                    boxes_224 = torch.tensor(
                        [[x1_224, y1_224, x2_224, y2_224]],
                        dtype=torch.float32, device=fmap.device,
                    )
                    pooled = roi_align(
                        fmap, [boxes_224],
                        output_size=(7, 7),
                        spatial_scale=spatial_scale,
                        sampling_ratio=-1, aligned=True,
                    )
                    feats_ts.append((ts_ms, pooled.squeeze(0).cpu()))
                    detected += 1
                    break
        finally:
            cap.release()

        if len(feats_ts) == 0:
            return (
                torch.zeros((T, MBV3_CHANNELS, 7, 7), dtype=torch.float32),
                int(label), simple_path, 0.0,
            )

        # TSN-style 分段 max-pool（與訓練一致）
        ts_arr = np.array([t for t, _ in feats_ts], dtype=np.float64)
        feats_list = [f for _, f in feats_ts]
        t0, t1 = float(ts_arr[0]), float(ts_arr[-1])
        if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
            pooled = [feats_list[min(i, len(feats_list) - 1)] for i in range(T)]
            video_feature = torch.stack(pooled, dim=0).float()
        else:
            edges = np.linspace(t0, t1, num=T + 1, dtype=np.float64)
            out = []
            for i in range(T):
                l, r = edges[i], edges[i + 1]
                mask = (ts_arr >= l) & (ts_arr < r) if i < T - 1 \
                    else (ts_arr >= l) & (ts_arr <= r)
                idxs = np.where(mask)[0]
                if idxs.size > 0:
                    chunk = torch.stack([feats_list[j] for j in idxs], dim=0)
                    out.append(pool_chunk(chunk, mode="max"))
                else:
                    mid = 0.5 * (l + r)
                    j = np.searchsorted(ts_arr, mid)
                    if j <= 0:
                        out.append(feats_list[0])
                    elif j >= len(feats_list):
                        out.append(feats_list[-1])
                    else:
                        out.append(
                            feats_list[j - 1]
                            if (mid - ts_arr[j - 1]) <= (ts_arr[j] - mid)
                            else feats_list[j]
                        )
            video_feature = torch.stack(out, dim=0).float()

        det_ratio = detected / total if total > 0 else 0.0

        # 寫回 cache（之後重複評估就秒抽）
        try:
            np.savez_compressed(
                cache_path,
                feat=video_feature.numpy().astype(np.float32, copy=False),
                det_ratio=np.array([det_ratio], dtype=np.float32),
            )
        except Exception as e:
            print(f"[警告] 寫 cache 失敗：{cache_path}: {e}")

        return video_feature, int(label), simple_path, float(det_ratio)


# =========================================================
# 5. 主流程
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="v1-F5.pt",
                    help="TinyTFClassifier 權重檔")
    ap.add_argument("--split_json", type=str, default="data_split.json",
                    help="訓練腳本產出的 data_split.json（內含 test 路徑清單）")
    ap.add_argument("--test_root", type=str, default="data",
                    help="舊版相容：資料根目錄（split_json 不存在時使用）")
    ap.add_argument("--test_subject", type=str, default="",
                    help="舊版相容：指定受試者 ID（split_json 不存在時使用）")
    ap.add_argument("--max_frames", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cache_root", type=str, default="data/video_features_v1",
                    help="特徵快取資料夾（與訓練同 hash key）")
    ap.add_argument("--out_dir", type=str, default="logs/test_eval_results",
                    help="評估結果輸出資料夾")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 裝置
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"=== Device: {device} ===")

    # backbone & yolo
    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    mobilenet.eval()
    feature_extractor = mobilenet.features.to(device).eval()

    yolo = YOLO("yolo11n.pt")
    yolo_device = "cuda" if device.type == "cuda" else "cpu"

    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 從 data_split.json 載入 test 路徑（優先），否則走舊版目錄掃描
    test_file_list = None
    if os.path.exists(args.split_json):
        print(f"=== 載入 split: {args.split_json} ===")
        with open(args.split_json, "r", encoding="utf-8") as _f:
            _split = json.load(_f)
        _test_paths  = _split["test"]
        # 從路徑反推 label（路徑最後兩層: .../label_name/video.ext）
        _test_labels = []
        for p in _test_paths:
            _parts = p.replace("\\", "/").split("/")
            # 找到 LABELS 中的類別名稱
            _lbl = next((label2idx[part] for part in reversed(_parts)
                         if part in label2idx), None)
            if _lbl is None:
                raise ValueError(f"無法從路徑推斷類別：{p}")
            _test_labels.append(_lbl)
        test_file_list = list(zip(_test_paths, _test_labels))
        print(f"    test samples: {len(test_file_list)}")
    else:
        print(f"[注意] 找不到 {args.split_json}，改用舊版 --test_root / --test_subject 模式")

    # Dataset / Loader
    ds = TestVideoDataset(
        root_dir=args.test_root,
        transform_norm=transform_norm,
        yolo_model=yolo,
        feature_extractor=feature_extractor,
        max_frames=args.max_frames,
        yolo_device=yolo_device,
        cache_root=args.cache_root,
        test_subject=args.test_subject,
        file_list=test_file_list,
    )
    if len(ds) == 0:
        print("[錯誤] test set 沒有任何樣本，請檢查 --test_root 與類別資料夾名稱")
        return
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    # 載入分類器
    model = TinyTFClassifier(mid=256, use_delta=True,
                             num_classes=len(LABELS)).to(device)
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"找不到權重檔：{args.weights}")
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"=== 載入權重：{args.weights} ===\n")

    # 推論
    y_true_all, y_pred_all = [], []
    fname_all, det_ratio_all = [], []
    all_probs = []

    with torch.no_grad():
        for feats, labels, fnames, det_ratios in tqdm(loader, desc="Evaluating test set"):
            feats = feats.to(device)
            labels = labels.to(device).long()

            logits = model(feats)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            y_true_all.extend(labels.detach().cpu().tolist())
            y_pred_all.extend(preds.detach().cpu().tolist())
            fname_all.extend(list(fnames))
            det_ratio_all.extend(
                [float(x) for x in det_ratios.detach().cpu().numpy().reshape(-1).tolist()]
            )
            all_probs.append(probs.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # [N, num_classes]
    y_true = np.array(y_true_all, dtype=int)
    y_pred = np.array(y_pred_all, dtype=int)

    # === 指標 ===
    overall_acc      = float((y_true == y_pred).mean()) * 100.0
    overall_macro_f1 = f1_score(y_true, y_pred, average="macro",  zero_division=0) * 100.0
    overall_micro_f1 = f1_score(y_true, y_pred, average="micro",  zero_division=0) * 100.0

    # 找出 test set 中實際出現的類別（P005 缺類時會有差異）
    present_classes = sorted(set(y_true.tolist()))
    missing_classes = [i for i in range(len(LABELS)) if i not in present_classes]
    if missing_classes:
        print(f"[注意] Test set 缺少以下類別（共 {len(missing_classes)} 類）：")
        for i in missing_classes:
            print(f"  [{i}] {LABELS[i]}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    cr_text = classification_report(
        y_true, y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS, digits=3, zero_division=0,
    )
    cr_dict = classification_report(
        y_true, y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS, digits=3, zero_division=0,
        output_dict=True,
    )

    # === 主控台輸出 ===
    _split_src = args.split_json if test_file_list is not None else \
                 f"{args.test_root}/{args.test_subject}"
    print("\n======================================")
    print(f"Split source : {_split_src}")
    print(f"Weights      : {args.weights}")
    print(f"Samples      : {len(y_true)}")
    print(f"Overall Acc  : {overall_acc:.2f}%  ({int((y_true == y_pred).sum())}/{len(y_true)})")
    print(f"Macro-F1     : {overall_macro_f1:.2f}%")
    print(f"Micro-F1     : {overall_micro_f1:.2f}%")
    print("======================================\n")

    print("Per-class 準確率：")
    for i, name in enumerate(LABELS):
        mask = (y_true == i)
        n = int(mask.sum())
        acc_i = float((y_pred[mask] == i).mean() * 100.0) if n > 0 else float("nan")
        print(f"  [{i}] {name:30s}: {acc_i:6.2f}%  (N={n})")

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr_text)

    # === 各類別偵測率（YOLO 找到人的比例）===
    print("Per-class YOLO detection ratio:")
    det_by_class = {i: [] for i in range(len(LABELS))}
    for t, r in zip(y_true, det_ratio_all):
        if np.isfinite(r):
            det_by_class[int(t)].append(float(r))
    for i, name in enumerate(LABELS):
        rs = det_by_class[i]
        if rs:
            print(f"  [{i}] {name:30s}: N={len(rs)}, mean={np.mean(rs):.3f}, std={np.std(rs):.3f}")
        else:
            print(f"  [{i}] {name:30s}: N=0")

    # === 輸出檔案 ===
    weight_tag = os.path.splitext(os.path.basename(args.weights))[0]

    # 1) 混淆矩陣 CSV（含表頭）
    cm_path = os.path.join(args.out_dir, f"confusion_matrix_{weight_tag}.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred"] + LABELS)
        for i, row in enumerate(cm):
            w.writerow([LABELS[i]] + row.tolist())
    print(f"\n[輸出] 混淆矩陣 → {cm_path}")

    # 2) 分類報告 TXT
    cr_path = os.path.join(args.out_dir, f"classification_report_{weight_tag}.txt")
    with open(cr_path, "w", encoding="utf-8") as f:
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Test root: {args.test_root}\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}%\n\n")
        f.write(cr_text)
    print(f"[輸出] 分類報告 → {cr_path}")

    # 3) 每筆預測明細 CSV
    pred_csv = os.path.join(args.out_dir, f"predictions_{weight_tag}.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["filename", "true_label", "pred_label",
                  "correct", "pred_prob", "det_ratio"] + [f"p_{n}" for n in LABELS]
        w.writerow(header)
        for i in range(len(y_true)):
            true_name = LABELS[int(y_true[i])]
            pred_name = LABELS[int(y_pred[i])]
            pred_prob = float(all_probs[i, int(y_pred[i])])
            w.writerow(
                [fname_all[i], true_name, pred_name,
                 int(y_true[i] == y_pred[i]),
                 f"{pred_prob:.6f}", f"{det_ratio_all[i]:.4f}"]
                + [f"{all_probs[i, k]:.6f}" for k in range(len(LABELS))]
            )
    print(f"[輸出] 逐筆預測 → {pred_csv}")

    # 4) 錯誤樣本 CSV（只留錯的）
    wrong_csv = os.path.join(args.out_dir, f"wrong_samples_{weight_tag}.csv")
    with open(wrong_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label", "pred_label",
                    "pred_prob", "det_ratio"])
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                w.writerow([
                    fname_all[i],
                    LABELS[int(y_true[i])],
                    LABELS[int(y_pred[i])],
                    f"{all_probs[i, int(y_pred[i])]:.6f}",
                    f"{det_ratio_all[i]:.4f}",
                ])
    print(f"[輸出] 錯誤樣本 → {wrong_csv}")

    # 5) summary JSON（方便程式化讀取）
    summary = {
        "weights": args.weights,
        "test_root": args.test_root,
        "split_source": _split_src,
        "num_samples": int(len(y_true)),
        "overall_accuracy_percent": round(overall_acc, 4),
        "macro_f1_percent": round(overall_macro_f1, 4),
        "micro_f1_percent": round(overall_micro_f1, 4),
        "missing_classes": [LABELS[i] for i in missing_classes],
        "per_class_metrics": cr_dict,
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
    }
    summary_path = os.path.join(args.out_dir, f"summary_{weight_tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[輸出] 評估摘要 → {summary_path}")

    print("\n完成。")


if __name__ == "__main__":
    main()
