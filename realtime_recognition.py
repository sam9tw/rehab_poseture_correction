import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.ops import roi_align
from ultralytics import YOLO

# ====== 1) 與訓練一致的標籤 ======
LABELS = [
    "shoulder_abduction_left", "shoulder_abduction_right",
    "shoulder_flexion_left",   "shoulder_flexion_right",
    "shoulder_backward_left",  "shoulder_backward_right",
    "elbow_flexion_left",      "elbow_flexion_right",
    "shoulder_forward_elevation",
]
MBV3_CHANNELS = 576  # mobilenet_v3_small.features output channels

# ====== 2) 與訓練一致的前處理 ======
def letterbox_224_with_params(rgb):
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
        borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, s, left, top

# ====== 3) 與訓練一致的模型結構 ======
class DSConV(nn.Module):  # ROI 7x7 -> mid
    def __init__(self, c=MBV3_CHANNELS, mid=256, act=nn.GELU, p=0.1):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)
        self.pw = nn.Conv2d(c, mid, kernel_size=1)
        self.bn = nn.GroupNorm(16, mid)
        self.act = act()
        self.drop = nn.Dropout(p)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):           # [B,c,7,7]
        x = self.dw(x)              # [B,c,7,7]
        x = self.pw(x)              # [B,mid,7,7]
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.gap(x)             # [B,mid,1,1]
        return x.squeeze(-1).squeeze(-1)  # [B,mid]

class TinyTFClassifier(nn.Module):
    def __init__(self, mid=256, d_model=256, nhead=4, num_layers=2,
                 dim_ff=1024, num_classes=9, dropout=0.1, use_delta=True, max_len=300):
        super().__init__()
        self.use_delta = use_delta
        self.roi_head = DSConV(c=MBV3_CHANNELS, mid=mid)
        in_dim = mid * 2 if use_delta else mid
        self.proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.tf = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.att = nn.Sequential(nn.Linear(d_model, 128), nn.Tanh(), nn.Linear(128, 1))
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B,T,576,7,7] 或 [B,T,mid]
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            x = self.roi_head(x)         # [B*T, mid]
            x = x.view(B, T, -1)         # [B,T, mid]
        elif x.ndim != 3:
            raise ValueError(f"Unexpected shape: {x.shape}")

        # 影片內標準化（與訓練一致）
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.std(dim=1, keepdim=True).clamp_min(1e-6)

        if self.use_delta:
            dx = torch.diff(x, dim=1, prepend=x[:, :1, :])
            x = torch.cat([x, dx], dim=2)  # [B,T, mid*2]

        x = self.proj(x)                   # [B,T,d_model]
        T = x.size(1)
        x = x + self.pos[:, :T, :]
        y = self.tf(x)                     # [B,T,d_model]

        w = torch.softmax(self.att(y).squeeze(-1), dim=1)  # [B,T]
        h = (y * w.unsqueeze(-1)).sum(dim=1)               # [B,d_model]
        return self.fc(h)                                   # [B,num_classes]

# ====== 4) 線上推論輔助：滑動視窗 Buffer ======
class OnlineBuffer:
    def __init__(self, T=30):
        self.T = T
        self.buf = []

    def push(self, feat_576_7_7):  # torch.Tensor [576,7,7]
        self.buf.append(feat_576_7_7)
        if len(self.buf) > self.T:
            self.buf.pop(0)

    def ready(self):
        return len(self.buf) > 0

    def tensor(self, device):
        if len(self.buf) == 0:
            return torch.zeros((1, self.T, MBV3_CHANNELS, 7, 7), dtype=torch.float32, device=device)
        # 最近鄰補齊到 T
        seq = self.buf[-self.T:]
        if len(seq) < self.T:
            seq = [seq[0]] * (self.T - len(seq)) + seq  # 前補
        x = torch.stack(seq, dim=0)  # [T,576,7,7]
        return x.unsqueeze(0).to(device)  # [1,T,576,7,7]

# ====== 5) 主程式 ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="v1_F4_P006.pt",
                    help="TinyTFClassifier 權重路徑（與訓練輸出一致）")
    ap.add_argument("--yolo", type=str, default="yolo11n.pt", help="YOLO 權重")
    ap.add_argument("--src", type=str, default="0",
                    help="視訊來源：攝影機索引(如 0) 或影片路徑")
    ap.add_argument("--T", type=int, default=30, help="滑動視窗長度（與訓練 max_frames 對齊）")
    ap.add_argument("--ema", type=float, default=0.6, help="logits 指數平滑係數(0~1)")
    ap.add_argument("--conf_thres", type=float, default=0.25, help="YOLO 置信度閾值")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--show_boxes", action="store_true", help="顯示 224 視窗上的人框")
    ap.add_argument("--no_cuda_half", action="store_true", help="禁用半精度")
    ap.add_argument("--conf_enter", type=float, default=0.90, help="進入顯示的機率閾值")
    ap.add_argument("--conf_exit",  type=float, default=0.50, help="退出顯示的機率閾值(防抖, < enter)")
    ap.add_argument("--hold",       type=int,   default=10,   help="最少保持幀數，避免一閃而過")
    ap.add_argument("--margin",     type=float, default=0.30, help="前二名機率差距（確信度）")

    args = ap.parse_args()

    # 裝置
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 讀取 YOLO 與 MobileNet.features
    yolo = YOLO(args.yolo)
    yolo.to(device if device.type == "cuda" else "cpu")
    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    mobilenet.eval()
    feature_extractor = mobilenet.features.to(device).eval()

    # Normalize (224×224)
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 分類器
    clf = TinyTFClassifier(mid=256, use_delta=True, num_classes=len(LABELS))
    ckpt = torch.load(args.weights, map_location="cpu")
    clf.load_state_dict(ckpt)
    clf.to(device).eval()

    use_half = (device.type == "cuda") and (not args.no_cuda_half)
    if use_half:
        feature_extractor.half()
        clf.half()

    # 視訊來源
    src_is_cam = args.src.isdigit()
    cap = cv2.VideoCapture(int(args.src) if src_is_cam else args.src)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟視訊來源：{args.src}")

    # 線上緩衝
    buffer = OnlineBuffer(T=args.T)
    last_logits = None
    fps_t0 = time.time()
    fps_counter = 0
    fps_disp = 0.0
    stable_label = None   # 目前畫面上顯示的類別（None 代表暫不顯示）
    hold_count  = 0       # 還需保持的幀數

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if src_is_cam:
                    continue
                break

            # YOLO 先在原始 BGR frame 上偵測，與 yolo11_detect_person_trf.py 一致；
            # 後續再把人框轉到 letterbox 224 座標做 ROI Align。
            results = yolo.predict(frame, device=(0 if device.type=="cuda" else "cpu"),
                                   classes=[0], conf=args.conf_thres, verbose=False)
            feat_this = None
            show_bgr = frame

            if results and len(results) > 0:
                r0 = results[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is not None and boxes.cls is not None and boxes.xyxy is not None and boxes.xyxy.shape[0] > 0:
                    cls = boxes.cls.detach().cpu().numpy()
                    person_idxs = [i for i, c in enumerate(cls) if int(c) == 0]
                    i = person_idxs[0] if person_idxs else 0
                    xyxy = boxes.xyxy[i].detach().cpu().numpy().tolist()
                    x1, y1, x2, y2 = map(float, xyxy)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img224, scale224, pad_left, pad_top = letterbox_224_with_params(rgb)

                    # 將原始 frame 的 YOLO 座標轉到 224 letterbox 座標。
                    x1_224 = float(np.clip(x1 * scale224 + pad_left, 0, 224))
                    y1_224 = float(np.clip(y1 * scale224 + pad_top, 0, 224))
                    x2_224 = float(np.clip(x2 * scale224 + pad_left, 0, 224))
                    y2_224 = float(np.clip(y2 * scale224 + pad_top, 0, 224))

                    # MobileNet.features
                    ten = transform_norm(img224).unsqueeze(0).to(device)
                    if use_half:
                        ten = ten.half()
                    with torch.no_grad():
                        fmap = feature_extractor(ten)  # [1,576,7,7]
                    spatial_scale = fmap.shape[-1] / 224.0  # 7/224
                    boxes_224 = torch.tensor([[x1_224, y1_224, x2_224, y2_224]],
                                             dtype=torch.float16 if use_half else torch.float32,
                                             device=device)
                    pooled = roi_align(
                        fmap, [boxes_224],
                        output_size=(7, 7),
                        spatial_scale=spatial_scale,
                        sampling_ratio=-1,
                        aligned=True
                    )  # [1,576,7,7]
                    feat_this = pooled.squeeze(0)  # [576,7,7]

                    if args.show_boxes:
                        vis = img224.copy()
                        cv2.rectangle(
                            vis,
                            (int(round(x1_224)), int(round(y1_224))),
                            (int(round(x2_224)), int(round(y2_224))),
                            (0, 255, 0), 2
                        )
                        show_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    else:
                        show_bgr = frame
                else:
                    show_bgr = frame
            else:
                show_bgr = frame

            # 若偵測到人，push 本幀 ROI 特徵；否則沿用上一幀的
            if feat_this is not None:
                buffer.push(feat_this.detach())
            elif buffer.ready():
                buffer.push(buffer.buf[-1])  # hold-last
            else:
                # 還沒任何特徵就先顯示原畫面
                cv2.imshow("Rehab Action (online)", show_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # 組成序列，做分類
            with torch.no_grad():
                seq = buffer.tensor(device)            # [1,T,576,7,7]
                logits = clf(seq)                      # [1,num_classes]
                if use_half: logits = logits.float()   # 後續 softmax 用 float32 較穩

                # 指數平滑（減抖動）
                if last_logits is None:
                    smooth = logits
                else:
                    smooth = args.ema * last_logits + (1 - args.ema) * logits
                last_logits = smooth

                probs = torch.softmax(smooth, dim=-1).squeeze(0).cpu().numpy()
                pred_id = int(np.argmax(probs))
                pred_prob = float(probs[pred_id])
                # ====== 顯示決策：閾值 + 防抖（雙閾值）+ 最小保持幀數 + 前二名差距 ======
                # 前二名差距（提高確信度，避免接近票互搶）
                top2 = float(np.partition(probs, -2)[-2]) if probs.size >= 2 else 0.0
                gap_ok = (pred_prob - top2) >= args.margin

                if stable_label is None:
                    # 尚未顯示任何類別：需滿足「進入閾值」且差距夠大
                    if (pred_prob >= args.conf_enter) and gap_ok:
                        stable_label = pred_id
                        hold_count   = args.hold
                else:
                    if stable_label == pred_id:
                        # 預測與目前顯示相同：若信心仍高，刷新保持；否則遞減保持
                        if pred_prob >= args.conf_exit:
                            hold_count = max(hold_count, 1)  # 保底不讓馬上歸零
                            hold_count -= 1 if hold_count > 0 else 0
                        else:
                            hold_count -= 1
                        if hold_count <= 0:
                            stable_label = None
                    else:
                        # 預測想切換：只有「新類別達進入閾值 + 差距夠」才切換，並重置保持
                        if (pred_prob >= args.conf_enter) and gap_ok:
                            stable_label = pred_id
                            hold_count   = args.hold
                        else:
                            # 不中門檻，維持原顯示並遞減保持
                            hold_count -= 1
                            if hold_count <= 0:
                                stable_label = None

                # ====== 最終決定顯示內容 ======
                if stable_label is not None:
                    show_name = LABELS[stable_label]
                    show_prob = float(probs[stable_label])
                else:
                    show_name = "undetermined"
                    show_prob = 0.0


            # FPS 統計
            fps_counter += 1
            t1 = time.time()
            if t1 - fps_t0 >= 0.5:
                fps_disp = fps_counter / (t1 - fps_t0)
                fps_counter = 0
                fps_t0 = t1

            # 疊字
            text = f"{show_name}  {show_prob*100:.1f}%  |  FPS {fps_disp:.1f}"
            cv2.putText(show_bgr, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 200, 40), 2, cv2.LINE_AA)

            # 簡易條形機率條
            bar_w = 240
            base_y = 60
            for i, name in enumerate(LABELS):
                p = float(probs[i])
                x1, y1 = 12, base_y + i*22
                x2 = x1 + int(bar_w * p)
                cv2.rectangle(show_bgr, (x1, y1), (x1+bar_w, y1+16), (60,60,60), 1)
                cv2.rectangle(show_bgr, (x1, y1), (x2, y1+16), (80,180,250), -1)
                cv2.putText(show_bgr, f"{name[:26]:s}", (x1+bar_w+8, y1+14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)

            cv2.imshow("Rehab Action (online)", show_bgr)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
