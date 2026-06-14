import socket
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.ops import roi_align
from ultralytics import YOLO

# ===== Unity 設定 =====
HOST = '127.0.0.1'
PORT = 5500

def send_command(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(command.encode())
    except ConnectionRefusedError:
        print("Unity 伺服器未啟動")

# ===== 模型設定 =====
LABELS = [
    "shoulder_abduction_left", "shoulder_abduction_right",
    "shoulder_flexion_left",   "shoulder_flexion_right",
    "shoulder_backward_left",  "shoulder_backward_right",
    "elbow_flexion_left",      "elbow_flexion_right",
    "shoulder_forward_elevation",
]

# 只對這些動作發送 Unity 指令 
action_map = {
    "shoulder_abduction_left": "u",    # 轉向左
    "shoulder_abduction_right": "o",   # 轉向右
    "shoulder_flexion_left": "j",              # 左
    "shoulder_flexion_right": "l",             # 右
    "shoulder_forward_elevation": "i", # 前
}

# 冷卻時間 (秒)
COOLDOWN = 4
# ===== 前處理函數 =====
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

# ===== 模型結構 =====
class DSConV(nn.Module): # Depthwise Separable Convolution
    def __init__(self, c=960, mid=256, act=nn.GELU, p=0.1):
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
        self.roi_head = DSConV(c=960, mid=mid)
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
    def forward(self, x):
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            x = self.roi_head(x)
            x = x.view(B, T, -1)
        elif x.ndim != 3:
            raise ValueError(f"Unexpected shape: {x.shape}")
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

# ===== 線上 Buffer =====
class OnlineBuffer:
    def __init__(self, T=30):
        self.T = T
        self.buf = []
    def push(self, feat_960_3_3):
        self.buf.append(feat_960_3_3)
        if len(self.buf) > self.T:
            self.buf.pop(0)
    def ready(self):
        return len(self.buf) > 0
    def tensor(self, device):
        if len(self.buf) == 0:
            return torch.zeros((1, self.T, 960, 3, 3), dtype=torch.float32, device=device)
        seq = self.buf[-self.T:]
        if len(seq) < self.T:
            seq = [seq[0]] * (self.T - len(seq)) + seq
        return torch.stack(seq, dim=0).unsqueeze(0).to(device)

# ===== 主程式 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="v1_F1.pt") 
    parser.add_argument("--yolo", type=str, default="yolo11n.pt")
    parser.add_argument("--src", type=str, default="0")
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device=="auto" else args.device)

    yolo = YOLO(args.yolo)
    yolo.to(device if device.type=="cuda" else "cpu")

    mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    mobilenet.eval()
    feature_extractor = mobilenet.features.to(device).eval()

    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    clf = TinyTFClassifier(num_classes=len(LABELS))
    ckpt = torch.load(args.weights, map_location=device)
    clf.load_state_dict(ckpt)
    clf.to(device).eval()

    cap = cv2.VideoCapture(int(args.src) if args.src.isdigit() else args.src)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟視訊來源：{args.src}")

    buffer = OnlineBuffer(T=args.T)
    last_sent_time = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img224, s224, padL, padT = letterbox_224_with_params(rgb)

            # YOLO 偵測 person
            results = yolo.predict(img224, device=(0 if device.type=="cuda" else "cpu"), classes=[0], conf=0.25, verbose=False)
            feat_this = None
            if results and len(results) > 0:
                r0 = results[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is not None and getattr(boxes, "xyxy", None) is not None and boxes.xyxy.shape[0] > 0:
                    confs = getattr(boxes, "conf", None)
                    i = int(torch.argmax(confs).item()) if confs is not None else 0
                    xyxy = boxes.xyxy[i].detach().cpu().numpy().tolist()
                    x1_224, y1_224, x2_224, y2_224 = xyxy

                    # 計算每個框的中心點
                    cx = (x1_224 + x2_224) / 2
                    cy = (y1_224 + y2_224) / 2

                    # 計算框與影像中心的距離，選擇最中間的人物
                    img_center_x, img_center_y = 224 / 2, 224 / 2
                    distance = np.sqrt((cx - img_center_x) ** 2 + (cy - img_center_y) ** 2)

                    # 如果是第一幀，選擇最接近影像中心的框
                    if not main_track['initialized']:
                        main_track = {'bbox': [x1_224, y1_224, x2_224, y2_224], 'center': [cx, cy], 'dist': distance, 'initialized': True}
                    else:
                        if distance < main_track['dist']:
                            main_track['bbox'] = [x1_224, y1_224, x2_224, y2_224]
                            main_track['center'] = [cx, cy]
                            main_track['dist'] = distance
                     # 基於選定的追蹤框執行後續操作
                    x1_224, y1_224, x2_224, y2_224 = main_track['bbox']
                    ten = transform_norm(img224).unsqueeze(0).to(device)
                    with torch.no_grad():
                        fmap = feature_extractor(ten)
                    spatial_scale = fmap.shape[-1] / 224.0
                    boxes_224 = torch.tensor([[x1_224, y1_224, x2_224, y2_224]], device=device)
                    pooled = roi_align(fmap, [boxes_224], output_size=(7, 7), spatial_scale=spatial_scale, sampling_ratio=-1, aligned=True)
                    feat_this = pooled.squeeze(0)

                    # 顯示選定的追蹤框
                    cv2.rectangle(frame, (int(x1_224), int(y1_224)), (int(x2_224), int(y2_224)), (0, 255, 0), 2)

            if feat_this is not None:
                buffer.push(feat_this)
            elif buffer.ready():
                buffer.push(buffer.buf[-1])
            else:
                continue

            seq = buffer.tensor(device)
            with torch.no_grad():
                logits = clf(seq)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                pred_id = int(np.argmax(probs))
                pred_name = LABELS[pred_id]
                pred_prob = float(probs[pred_id])

            current_time = time.time()
            if pred_name in action_map and pred_prob > 0.9:
                if current_time - last_sent_time >= COOLDOWN:
                    send_command(action_map[pred_name])
                    last_sent_time = current_time

            # 可選：顯示影像與預測
            text = f"{pred_name}  {pred_prob*100:.1f}%"
            cv2.putText(frame, text, (12,32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20,200,40), 2)
            cv2.imshow("Action Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()