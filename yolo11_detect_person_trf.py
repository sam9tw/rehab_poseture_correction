import os
import csv
import cv2
import json
import hashlib
import math
import re
import random
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.ops import roi_align
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

# === 1. 動作類別 ===
LABELS = [
    "shoulder_abduction_left", "shoulder_abduction_right",
    "shoulder_flexion_left", "shoulder_flexion_right",
    "shoulder_backward_left", "shoulder_backward_right",
    # "side_tap_left", "side_tap_right",
    "elbow_flexion_left", "elbow_flexion_right",
    "shoulder_forward_elevation"
]
label2idx = {label: i for i, label in enumerate(LABELS)}

FLIP_MAP = {
    "shoulder_abduction_left":"shoulder_abduction_right",
    "shoulder_abduction_right":"shoulder_abduction_left",
    "shoulder_flexion_left":"shoulder_flexion_right",
    "shoulder_flexion_right":"shoulder_flexion_left",
    "shoulder_backward_left":"shoulder_backward_right",
    "shoulder_backward_right":"shoulder_backward_left",
    #  "side_tap_left":"side_tap_right",
    #  "side_tap_right":"side_tap_left",
    "elbow_flexion_left":"elbow_flexion_right",
    "elbow_flexion_right":"elbow_flexion_left",
}
MBV3_CHANNELS = 576 # mobilenet_v3_small output channels (features layer)

# === 1. Letterbox_resize: 等比例縮放到 224 並回傳縮放與 padding 參數 ===
def letterbox_224_with_params(rgb):
    h, w = rgb.shape[:2]
    s = 224.0 / max(h, w) # 最小縮放比例
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)  # 影像等比例縮放
    top = (224 - nh) // 2
    bottom = 224 - nh - top
    left = (224 - nw) // 2
    right = 224 - nw - left
    padded = cv2.copyMakeBorder( # padding成224x224
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, s, left, top  # 224x224, 縮放係數, 左pad, 上pad

# === 2. 資料集類別（整合 ROI Align） ===
# 特徵檔案快取
def make_cache_path(video_path, do_flip, meta: dict, cache_root="cache_feats"):
    """
    meta 內放會影響特徵的所有超參，避免改參數卻撞舊檔。
    例如: T、roi、scale、backbone、norm、diff、version...
    """
    os.makedirs(cache_root, exist_ok=True)
    # 用「完整路徑 + meta」做 hash，避免重名、確保參數變動就換檔
    key_str = json.dumps(meta, sort_keys=True) + "|" + os.path.abspath(video_path) + f"|flip={int(do_flip)}"
    h = hashlib.md5(key_str.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(video_path))[0]
    fname = f"{stem}-{h}.npz"
    return os.path.join(cache_root, fname)

def temporal_cutout(x, drop_ratio=0.15, train=True):
    # x: [B,T,C]  (已過 ROI head 的序列)
    if not train or drop_ratio <= 0: return x
    B, T, C = x.shape
    L = max(1, int(T * drop_ratio))
    for b in range(B):
        s = np.random.randint(0, T - L + 1)
        x[b, s:s+L, :] = 0
    return x

def pool_chunk(chunk, mode="mean"):
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

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform_norm, yolo_model, feature_extractor,
                 max_frames=30, yolo_device='cpu', show=True, is_train=False,
                 flip_prob=0.5, file_list=None):
        self.samples = []  # (PATH, label index)
        self.to_tensor_norm = transform_norm
        self.yolo = yolo_model
        self.feature_extractor = feature_extractor  # MobileNet.features
        self.max_frames = max_frames
        self.yolo_device = yolo_device
        self.show = show
        self.is_train = is_train
        self.flip_prob = float(flip_prob)
        self.detection_log = []

        if file_list is not None:
            # 直接使用指定的 (path, label_idx) 清單，不掃描目錄
            self.samples = list(file_list)
            return

        # 自動偵測目錄結構：
        #   受試者模式：root_dir/P001/label/video.mp4
        #   平坦模式：  root_dir/label/video.mp4
        _subdirs = sorted(os.listdir(root_dir)) if os.path.isdir(root_dir) else []
        _subject_dirs = [
            d for d in _subdirs
            if os.path.isdir(os.path.join(root_dir, d)) and re.match(r'^P\d+$', d)
        ]

        if _subject_dirs:
            # 受試者子目錄模式
            for subj in _subject_dirs:
                for label in LABELS:
                    label_dir = os.path.join(root_dir, subj, label)
                    if not os.path.isdir(label_dir):
                        continue
                    for fname in sorted(os.listdir(label_dir)):
                        if fname.lower().endswith(('.mov', '.mp4', '.mkv', '.avi')):
                            self.samples.append((os.path.join(label_dir, fname), label2idx[label]))
        else:
            # 平坦模式（原有相容）
            for label in LABELS:
                label_dir = os.path.join(root_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                for fname in sorted(os.listdir(label_dir)):
                    if fname.lower().endswith(('.mov', '.mp4', '.mkv', '.avi')):
                        self.samples.append((os.path.join(label_dir, fname), label2idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        simple_path = os.path.splitext(os.path.basename(video_path))[0]

        do_flip = self.is_train and (np.random.rand() < self.flip_prob)
        if do_flip:
            lbl_name = LABELS[int(label)]
            if lbl_name in FLIP_MAP:
                label = label2idx[FLIP_MAP[lbl_name]]  # 左右互換

        meta = {
            "T": self.max_frames,
            "backbone": "mbv3_small.features@224",
            "norm": "imagenet",
            "roi": {"type": "align", "bins": [7, 7], "aligned": True, "sampling": "auto", "scale": "7/224"},
            "sampler": "uniform_ts+segment_max+nearest_fill",
            "cached_shape": [MBV3_CHANNELS, 7, 7],   # 每幀 shape
            "head": "DSConv+GAP",
            "version": "v1"
        }

        cache_path = make_cache_path(video_path, do_flip, meta, cache_root=os.path.join("data","video_features_v1"))

        # 讀快取
        try:
            with np.load(cache_path, allow_pickle=False, mmap_mode="r") as z:
                arr = z["feat"]
                # 基本驗證
                det_ratio = float(z["det_ratio"][0]) if "det_ratio" in z.files else float('nan')
                ok = (
                    (arr.ndim == 2 and arr.shape[1] in (MBV3_CHANNELS, 1920)) or
                    (arr.ndim == 4 and arr.shape[1] == MBV3_CHANNELS and tuple(arr.shape[2:]) == (3, 3)) or
                    (arr.ndim == 4 and arr.shape[1] == MBV3_CHANNELS and tuple(arr.shape[2:]) == (7, 7))
                )
                if ok:
                    return torch.from_numpy(arr.astype(np.float32, copy=False)), label, simple_path, det_ratio
        except Exception:
            pass  # 檔案不存在或壞掉就走計算路徑

        T = self.max_frames
 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            video_feature = torch.zeros((T, MBV3_CHANNELS, 7, 7), dtype=torch.float32)
            detection_ratio = 0.0
            self.detection_log.append({
                "video": os.path.basename(video_path),
                "detected_frames": 0,
                "total_frames": 0,
                "detection_ratio": float(detection_ratio)
            })
            return video_feature, int(label), simple_path, float(detection_ratio)

        feats_ts = []  # list of (ts_ms, feat[MBV3_CHANNELS])
        person_detected_frames = 0
        total_frames_read = 0

        # 取得 FPS（可能為 0）
        fps = cap.get(cv2.CAP_PROP_FPS)
        use_fps_ts = fps is not None and fps > 1e-3
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                total_frames_read += 1

                # ---- 若此段需要翻轉，對每幀做水平翻轉 ----
                if do_flip:
                    frame = cv2.flip(frame, 1)
                # --- 每幀的時間戳（毫秒） ---
                if use_fps_ts:
                    ts_ms = (frame_idx * 1000.0) / float(fps)
                else:
                    # 有些編碼器在 read() 後取 CAP_PROP_POS_MSEC 也行
                    ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                frame_idx += 1

                # --- YOLO 單幀偵測 ---
                results = self.yolo(frame, device=self.yolo_device, verbose=False, classes=[0])  # 只偵測 person (class_id=0)
                if not results or len(results) == 0:
                    continue
                r0 = results[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is None or boxes.cls is None or boxes.xyxy is None:
                    continue

                cls = boxes.cls.detach().cpu().numpy()
                xyxy = boxes.xyxy.detach().cpu().numpy()

                # 找第一個 person
                got_person = False
                for i in range(len(cls)):
                    if int(cls[i]) != 0:
                        continue
                    x1, y1, x2, y2 = map(float, xyxy[i])

                    # --- 整幀 letterbox 到 224 ---
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img224, scale224, pad_left, pad_top = letterbox_224_with_params(frame_rgb)

                    # 人框座標，後續ROI Align使用
                    x1_224 = float(np.clip(x1 * scale224 + pad_left, 0, 224))
                    y1_224 = float(np.clip(y1 * scale224 + pad_top,  0, 224))
                    x2_224 = float(np.clip(x2 * scale224 + pad_left, 0, 224))
                    y2_224 = float(np.clip(y2 * scale224 + pad_top,  0, 224))

                    # --- 特徵圖 ---
                    img_tensor = self.to_tensor_norm(img224).unsqueeze(0).to(
                        next(self.feature_extractor.parameters()).device
                    )  # [1,3,224,224]
                    with torch.no_grad():
                        fmap = self.feature_extractor(img_tensor)  # [1,MBV3_CHANNELS,7,7]

                    # --- ROI Align（對齊到固定 7x7） ---
                    spatial_scale = fmap.shape[-1] / 224.0  # 7/224
                    boxes_224 = torch.tensor([[x1_224, y1_224, x2_224, y2_224]],
                                            dtype=torch.float32, device=fmap.device)
                    pooled = roi_align(
                        fmap, [boxes_224],
                        output_size=(7, 7),
                        spatial_scale=spatial_scale,
                        sampling_ratio=-1,
                        aligned=True
                    )  # [1,MBV3_CHANNELS,7,7]
                    feat = pooled.squeeze(0)
                    feats_ts.append((ts_ms, feat.cpu()))
                    person_detected_frames += 1
                    got_person = True

                    if self.show:
                        vis = img224.copy()
                        cv2.rectangle(
                            vis,
                            (int(round(x1_224)), int(round(y1_224))),
                            (int(round(x2_224)), int(round(y2_224))),
                            (0, 255, 0), 2
                        )
                        cv2.imshow("YOLO ROIAlign (letterboxed 224)", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    break  # 只取第一個人
                # 若此幀沒抓到人，就不 push feats_ts；之後分段池化會做最近鄰補
        finally:
            cap.release()
            if self.show:
                cv2.destroyAllWindows()

        # --- 若整支影片完全沒有特徵，回傳全零 ---
        if len(feats_ts) == 0:
            video_feature = torch.zeros((T, MBV3_CHANNELS, 3, 3), dtype=torch.float32)
            detection_ratio = 0.0
            self.detection_log.append({
                "video": os.path.basename(video_path),
                "detected_frames": 0,
                "total_frames": int(total_frames_read),
                "detection_ratio": float(detection_ratio)
            })
            return video_feature, int(label), simple_path, float(detection_ratio)

        # === 按時間戳均勻化 + 分段池化（TSN-style）===
        # 在 [t0, t1] 上切成 T 段，段內對多幀做平均；空段用最近鄰補
        ts_arr = np.array([t for t, _ in feats_ts], dtype=np.float64)
        feats_list = [f if isinstance(f, torch.Tensor) else torch.from_numpy(f) for _, f in feats_ts]
        t0, t1 = float(ts_arr[0]), float(ts_arr[-1])  # 第一幀時間/最後一幀時間
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            # 退化情況：所有 ts 一樣或異常 -> 直接均勻複製最近鄰
            pooled = [feats_list[min(i, len(feats_list)-1)] for i in range(T)]
            video_feature = torch.stack(pooled, dim=0).float()
        else:
            edges = np.linspace(t0, t1, num=T+1, dtype=np.float64)  # T 段的邊界
            pooled_feats = []
            for i in range(T):
                l = edges[i]
                r = edges[i+1]
                # 左閉右開，最後一段右端點包含
                if i < T-1:
                    mask = (ts_arr >= l) & (ts_arr < r)
                else:
                    mask = (ts_arr >= l) & (ts_arr <= r)
                idxs = np.where(mask)[0]

                if idxs.size > 0: # 該區段有幀
                    chunk = torch.stack([feats_list[j] for j in idxs], dim=0)  # [K,MBV3_CHANNELS, 3, 3]
                    pooled = pool_chunk(chunk, mode="max") # [MBV3_CHANNELS, 3, 3]
                else: # 該區段無幀
                    # 最近鄰補：取該段中點的最近時間
                    mid = 0.5 * (l + r)
                    j = np.searchsorted(ts_arr, mid)
                    if j <= 0:
                        pooled = feats_list[0]
                    elif j >= len(feats_list):
                        pooled = feats_list[-1]
                    else:
                        pooled = feats_list[j-1] if (mid - ts_arr[j-1]) <= (ts_arr[j] - mid) else feats_list[j]
                pooled_feats.append(pooled)
            video_feature = torch.stack(pooled_feats, dim=0).float()  # [T,MBV3_CHANNELS, 3, 3]

        detection_ratio = (person_detected_frames / total_frames_read) if total_frames_read > 0 else 0.0
        self.detection_log.append({
            "video": os.path.basename(video_path),
            "detected_frames": int(person_detected_frames),
            "total_frames": int(total_frames_read),
            "detection_ratio": float(detection_ratio)
        })
        # 否則照原流程抽 -> 存 cache
        np.savez_compressed(
            cache_path, 
            feat=video_feature.numpy().astype(np.float32, copy=False), 
            det_ratio=np.array([detection_ratio], dtype=np.float32)
        )
        return video_feature, int(label), simple_path, float(detection_ratio)

class DSConV(nn.Module): # Depthwise Separable Convolution
    def __init__(self, c=MBV3_CHANNELS, mid=256, act=nn.GELU, p=0.1):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)   # depthwise 3x3: [B,c,7,7]
        self.pw = nn.Conv2d(c, mid, kernel_size=1)           # pointwise 1x1: [B,mid,7,7]
        self.bn = nn.GroupNorm(16, mid)  # 使用 GroupNorm 代替 BatchNorm
        self.act = act()
        self.drop = nn.Dropout(p)
        self.gap = nn.AdaptiveAvgPool2d(1)  # [B,mid,1,1]

    def forward(self, x):           # x: [B,c,7,7]
        x = self.dw(x)              # [B,c,7,7]
        x = self.pw(x)              # [B,mid,7,7]
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.gap(x)          # [B,mid,1,1]
        return x.squeeze(-1).squeeze(-1)   # [B,mid]

class TinyTFClassifier(nn.Module):
    def __init__(self, mid=256, d_model=256, nhead=4, num_layers=2,
                 dim_ff=1024, num_classes=9, dropout=0.1, use_delta=True, max_len=300):
        super().__init__()
        self.use_delta = use_delta
        # ROI 3x3 → mid（例如 256）
        self.roi_head = DSConV(c=MBV3_CHANNELS, mid=mid)

        in_dim = mid * 2 if use_delta else mid
        self.proj = nn.Linear(in_dim, d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.tf = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))  # [:, :T, :] # 可學式位置編碼

        self.att = nn.Sequential(nn.Linear(d_model, 128), nn.Tanh(), nn.Linear(128, 1))
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B,T,MBV3_CHANNELS,3,3] 或 [B,T,mid]（已過 ROI head）
        if x.ndim == 5:  # [B,T,C,3,3]
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)          # [B*T,MBV3_CHANNELS,3,3]
            x = self.roi_head(x)              # [B*T, mid]
            x = x.view(B, T, -1)              # [B,T, mid]
        elif x.ndim == 3:
            # 已是 [B,T,mid] 的情形
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # 影片內標準化
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.std(dim=1, keepdim=True).clamp_min(1e-6)
        if self.training:
            # x 此時是 [B,T,mid]；若 use_delta=True，cutout 放在 concat Δt 之前
            x = temporal_cutout(x, drop_ratio=0.15, train=True)

        # 一階時間差分(凸顯動作的變化)
        if self.use_delta:
            dx = torch.diff(x, dim=1, prepend=x[:, :1, :])
            x = torch.cat([x, dx], dim=2)     # [B,T, mid*2]

        # Transformer
        x = self.proj(x)                      # [B,T,d_model]
        T = x.size(1)
        x = x + self.pos[:, :T, :]
        y = self.tf(x)                        # [B,T,d_model]

        # Attention 池化
        w = torch.softmax(self.att(y).squeeze(-1), dim=1)  # [B,T] # 學習一個注意力分數看哪個時間步重要
        h = (y * w.unsqueeze(-1)).sum(dim=1)               # [B,d_model] # 加權和
        return self.fc(h)


# === 3. MobileNet 特徵擷取（取 features 層，供 ROI Align） ===
mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
mobilenet.eval()
feature_extractor = mobilenet.features  # [N,MBV3_CHANNELS,7,7]
feature_extractor.eval()

# 抽特徵裝置（可用 GPU 就上；YOLO 先用 CPU 避開 NMS 問題）
extractor_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# extractor_device = 'cpu'
feature_extractor.to(extractor_device)

# === 4. Normalize（整幀 224×224 使用） ===
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 5. YOLOv11 模型 ===
yolo = YOLO('yolo11n.pt')
yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# yolo_device = 'cpu'

# ===================================================================
# 6. 掃描全部影片 -> 各類別平衡取樣 -> 70/15/15 分割
# ===================================================================
print("\n[1/3] 掃描所有影片...")
_scan_ds = VideoDataset(
    root_dir='data',
    transform_norm=transform_norm,
    yolo_model=yolo,
    feature_extractor=feature_extractor,
    max_frames=30,
    yolo_device=yolo_device,
    show=False,
    is_train=False,
    flip_prob=0.0,
)
all_samples = _scan_ds.samples   # list of (path, label_idx)
print(f"      找到 {len(all_samples)} 支影片\n")

# 按類別分組
class_to_samples: dict = defaultdict(list)
for path, label in all_samples:
    class_to_samples[label].append((path, label))

print("各類別影片數：")
for i, name in enumerate(LABELS):
    print(f"  [{i}] {name:30s}: {len(class_to_samples[i])}")

min_count = min(len(v) for v in class_to_samples.values())
print(f"\n最少類別數: {min_count} -> 各類別均取此數量（下採樣平衡）")

# 各類別均衡取樣（固定 seed 保持可重現）
random.seed(42)
balanced_samples = []
for label_idx in range(len(LABELS)):
    sampled = random.sample(class_to_samples[label_idx], min_count)
    balanced_samples.extend(sampled)
print(f"平衡後總樣本: {len(balanced_samples)}  ({len(LABELS)} 類 x {min_count})\n")

# ===================================================================
# 7. 分割 pool / test（85% pool，15% test 封存）
# ===================================================================
_paths  = [p for p, _ in balanced_samples]
_labels = [l for _, l in balanced_samples]

# 85% 進入 5-fold CV pool，15% 封存為 test
pool_paths, test_paths, pool_labels, test_labels = train_test_split(
    _paths, _labels, test_size=0.15, stratify=_labels, random_state=42
)

_total = len(balanced_samples)
print("[2/3] 資料分割結果：")
print(f"  Pool (train+val): {len(pool_paths):4d} samples  ({len(pool_paths)/_total*100:.0f}%)")
print(f"  Test  (封存)    : {len(test_paths):4d} samples  ({len(test_paths)/_total*100:.0f}%)")

# 儲存 split 供評估腳本使用
split_info = {
    "min_class_count": min_count,
    "split_ratio": "85pool/15test + 5-fold CV",
    "pool": pool_paths,
    "test": test_paths,
}
with open("data_split.json", "w", encoding="utf-8") as _f:
    json.dump(split_info, _f, ensure_ascii=False, indent=2)
print("  Split 資訊已儲存至 data_split.json\n")

# ===================================================================
# 8. 準備 5-fold StratifiedKFold
# ===================================================================
pool_paths_arr  = np.array(pool_paths)
pool_labels_arr = np.array(pool_labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ===================================================================
# 9. 訓練主函式（macro-F1 early stopping）
# ===================================================================
def run_training(train_ds, val_ds, fold_num):
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

    model     = TinyTFClassifier(mid=256, use_delta=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_f1   = 0.0
    best_val_acc  = 0.0
    best_epoch    = 0
    best_cr_text  = ""
    best_cm       = None
    patience      = 10
    patience_counter = 0

    # ---- Log 初始化 ----
    log_dir = os.path.join("logs", f"F{fold_num}")
    os.makedirs(log_dir, exist_ok=True)

    epoch_csv_path = os.path.join(log_dir, f"F{fold_num}_epochs.csv")
    wrong_csv_path = os.path.join(log_dir, f"F{fold_num}_wrong_samples.csv")

    with open(epoch_csv_path, "w", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow([
            "fold", "epoch", "train_loss", "val_loss",
            "train_acc", "val_acc", "val_macro_f1", "lr", "is_best"
        ])
    with open(wrong_csv_path, "w", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow(["Epoch", "Filename", "True Label", "Predicted Label"])

    EPOCHS = 128
    for epoch in tqdm(range(EPOCHS), desc=f"[Fold {fold_num}] Epochs"):

        # ---- Train ----
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for feats, labels_b, _, det_ratio in tqdm(
            train_loader, desc=f"  Train Ep {epoch+1}", leave=False
        ):
            feats    = feats.to(device)
            labels_b = labels_b.to(device).long()

            preds = model(feats)
            loss  = criterion(preds, labels_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            correct    += (preds.argmax(1) == labels_b).sum().item()
            total      += labels_b.size(0)

        train_acc = correct / total * 100 if total > 0 else 0.0

        # ---- Validation ----
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        y_true_all, y_pred_all   = [], []
        det_by_class             = {i: [] for i in range(len(LABELS))}
        wrong_samples_ep         = []
        wrong_label_cnt          = {i: 0 for i in range(len(LABELS))}
        label_cnt                = {i: 0 for i in range(len(LABELS))}

        with torch.no_grad():
            for feats, labels_b, path, det_ratio in tqdm(
                val_loader, desc=f"  Val   Ep {epoch+1}", leave=False
            ):
                feats    = feats.to(device)
                labels_b = labels_b.to(device).long()

                preds     = model(feats)
                val_loss += criterion(preds, labels_b).item()
                correct  += (preds.argmax(1) == labels_b).sum().item()
                total    += labels_b.size(0)

                y_true_all.extend(labels_b.detach().cpu().tolist())
                y_pred_all.extend(preds.argmax(1).detach().cpu().tolist())

                if det_ratio is not None:
                    dr = det_ratio.detach().cpu().numpy().reshape(-1)
                    for t, r in zip(labels_b.detach().cpu().numpy(), dr.tolist()):
                        if np.isfinite(r):
                            det_by_class[int(t)].append(float(r))

                for fname, t_lbl, p_lbl in zip(path, labels_b, preds.argmax(1)):
                    t = t_lbl.item() if isinstance(t_lbl, torch.Tensor) else int(t_lbl)
                    p = p_lbl.item() if isinstance(p_lbl, torch.Tensor) else int(p_lbl)
                    if t != p:
                        wrong_samples_ep.append((epoch + 1, fname, LABELS[t], LABELS[p]))
                        wrong_label_cnt[t] += 1
                    label_cnt[t] += 1

        val_acc      = correct / total * 100 if total > 0 else 0.0
        val_macro_f1 = f1_score(y_true_all, y_pred_all,
                                average='macro', zero_division=0) * 100
        cur_lr       = optimizer.param_groups[0]['lr']

        # ---- 寫錯誤樣本 ----
        if wrong_samples_ep:
            with open(wrong_csv_path, "a", newline="", encoding="utf-8") as _f:
                csv.writer(_f).writerows(wrong_samples_ep)
            print(f"  wrong_label_cnt: {wrong_label_cnt}")
            print(f"  label_cnt:       {label_cnt}")

        # ---- LR Scheduler ----
        prev_lr = cur_lr
        scheduler.step(val_macro_f1)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"  LR: {prev_lr:.2e} -> {new_lr:.2e}")

        # ---- Early stopping 判斷 ----
        is_best = val_macro_f1 > best_val_f1
        if is_best:
            best_val_f1  = val_macro_f1
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f"v1-F{fold_num}.pt")
            # 儲存最佳 epoch 的分類報告與混淆矩陣（供事後查閱）
            best_cr_text = classification_report(
                y_true_all, y_pred_all,
                labels=list(range(len(LABELS))), target_names=LABELS,
                digits=3, zero_division=0
            )
            best_cm = confusion_matrix(y_true_all, y_pred_all,
                                       labels=list(range(len(LABELS))))
        else:
            patience_counter += 1

        # ---- 寫 epoch CSV ----
        with open(epoch_csv_path, "a", newline="", encoding="utf-8") as _f:
            csv.writer(_f).writerow([
                fold_num, epoch + 1,
                round(total_loss, 6), round(val_loss, 6),
                round(train_acc, 4),  round(val_acc, 4),
                round(val_macro_f1, 4), round(cur_lr, 8),
                int(is_best)
            ])

        # ---- Console ----
        print(
            f"[F{fold_num}] Ep {epoch+1:03d} | "
            f"TrainLoss={total_loss:.4f} ValLoss={val_loss:.4f} | "
            f"TrainAcc={train_acc:.2f}% ValAcc={val_acc:.2f}% "
            f"Val-macroF1={val_macro_f1:.2f}%"
            + (" ★ best" if is_best else "")
        )

        cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(LABELS))))
        print(f"  Confusion Matrix:\n{cm}")

        os.makedirs(f"confusion_matrix_F{fold_num}", exist_ok=True)
        np.savetxt(
            f"confusion_matrix_F{fold_num}/cm_epoch{epoch+1:03d}.csv",
            cm.astype(int), delimiter=",", fmt="%d"
        )

        cr = classification_report(
            y_true_all, y_pred_all,
            labels=list(range(len(LABELS))), target_names=LABELS,
            digits=3, zero_division=0
        )
        print(f"  Classification Report:\n{cr}")

        print(f"  Detection Ratios by Class:")
        for i, name in enumerate(LABELS):
            ratios = det_by_class[i]
            if ratios:
                print(f"    {i:02d} {name:30s}: N={len(ratios)}, "
                      f"mean={np.mean(ratios):.3f}, std={np.std(ratios):.3f}")
            else:
                print(f"    {i:02d} {name:30s}: N=0")

        if is_best:
            print(f"  -> new best  macro-F1={best_val_f1:.2f}%  acc={best_val_acc:.2f}%  (saved v1-F{fold_num}.pt)")
        elif patience_counter >= patience:
            print(f"  Early stopping (patience={patience})")
            break

    # ---- 儲存此折最佳 epoch 的詳細報告 ----
    if best_cr_text:
        report_path = os.path.join(log_dir, f"F{fold_num}_best_report.txt")
        with open(report_path, "w", encoding="utf-8") as _f:
            _f.write(f"Fold {fold_num}  Best Epoch: {best_epoch}\n")
            _f.write(f"Val macro-F1 : {best_val_f1:.4f}%\n")
            _f.write(f"Val Acc      : {best_val_acc:.4f}%\n\n")
            _f.write("=== Classification Report ===\n")
            _f.write(best_cr_text)
            if best_cm is not None:
                _f.write("\n=== Confusion Matrix ===\n")
                _f.write(str(best_cm) + "\n")

    return best_val_f1, best_val_acc, best_epoch 

# ===================================================================
# 10. 執行 5-fold 訓練
# ===================================================================
print("[3/3] 開始 5-fold 交叉驗證訓練...\n")
fold_results = []  # list of (fold_num, best_f1, best_acc)

for fold_num, (train_idx, val_idx) in enumerate(
    skf.split(pool_paths_arr, pool_labels_arr), start=1
):
    print(f"\n{'='*60}")
    print(f"Fold {fold_num}/5  train={len(train_idx)} samples  val={len(val_idx)} samples")
    print('='*60)

    train_paths_fold  = pool_paths_arr[train_idx].tolist()
    train_labels_fold = pool_labels_arr[train_idx].tolist()
    val_paths_fold    = pool_paths_arr[val_idx].tolist()
    val_labels_fold   = pool_labels_arr[val_idx].tolist()

    train_dataset = VideoDataset(
        root_dir='data',
        transform_norm=transform_norm,
        yolo_model=yolo,
        feature_extractor=feature_extractor,
        max_frames=30,
        yolo_device=yolo_device,
        show=False,
        is_train=True,
        flip_prob=0.5,
        file_list=list(zip(train_paths_fold, train_labels_fold)),
    )
    val_dataset = VideoDataset(
        root_dir='data',
        transform_norm=transform_norm,
        yolo_model=yolo,
        feature_extractor=feature_extractor,
        max_frames=30,
        yolo_device=yolo_device,
        show=False,
        is_train=False,
        flip_prob=0.0,
        file_list=list(zip(val_paths_fold, val_labels_fold)),
    )

    best_f1, best_acc, best_ep = run_training(train_dataset, val_dataset, fold_num)
    fold_results.append((fold_num, best_f1, best_acc, best_ep))
    print(f"[Fold {fold_num}]  best macro-F1={best_f1:.2f}%  best acc={best_acc:.2f}%  best_epoch={best_ep}")

# === 最終摘要 ===
print("\n" + "="*60)
print("5-fold CV 訓練完成")
print("="*60)
for fid, f1, acc, bep in fold_results:
    print(f"  Fold {fid}:  macro-F1={f1:.2f}%  acc={acc:.2f}%  best_epoch={bep}")
if fold_results:
    mean_f1  = float(np.mean([r[1] for r in fold_results]))
    mean_acc = float(np.mean([r[2] for r in fold_results]))
    print(f"  ---- Mean macro-F1 = {mean_f1:.2f}% ----")
    print(f"  ---- Mean Acc      = {mean_acc:.2f}% ----")
print(f"\n  模型已分別儲存至: v1-F1.pt ~ v1-F5.pt")
print(f"  Log 已儲存至: logs/F1/ ~ logs/F5/")
print(f"  Test set 分割資訊: data_split.json")

# 寫整體摘要 JSON
os.makedirs("logs", exist_ok=True)
summary = {
    "n_folds": 5,
    "min_class_count": min_count,
    "split_ratio": "85pool/15test + 5-fold CV",
    "folds": [
        {"fold": fid, "best_val_macro_f1": round(f1, 4),
         "best_val_acc": round(acc, 4), "best_epoch": bep}
        for fid, f1, acc, bep in fold_results
    ],
    "mean_val_macro_f1": round(float(np.mean([r[1] for r in fold_results])), 4) if fold_results else 0,
    "mean_val_acc":      round(float(np.mean([r[2] for r in fold_results])), 4) if fold_results else 0,
}
with open(os.path.join("logs", "summary.json"), "w", encoding="utf-8") as _f:
    json.dump(summary, _f, ensure_ascii=False, indent=2)
print(f"\n  整體訓練摘要已儲存至: logs/summary.json")
print(f"\n[提示] 執行 evaluate_test_set.py 對 test set 做最終評估。")
