"""
deduplicate_dataset.py
======================
掃描 data/ 目錄，找出 MD5 完全相同的影片（真正的重複檔案），
保留路徑字母順序最靠前的一份，刪除其餘重複檔案，
並同時刪除對應的特徵快取（data/video_features_v1/*.npz）。

用法：
  # 【強烈建議先跑這步】預覽，不刪任何檔案
  python deduplicate_dataset.py --dry-run

  # 確認無誤後，實際執行刪除
  python deduplicate_dataset.py

  # 自訂路徑
  python deduplicate_dataset.py --data_root data --cache_root data/video_features_v1

注意：
  - 只刪「位元組完全相同」的重複檔（MD5 碰撞機率極低可忽略）
  - 同場景連拍 5 次但內容不同的影片「不會」被刪，請放心
  - 刪除記錄會存到 dedup_log.txt，可事後查閱
"""

import os
import re
import json
import hashlib
import argparse
from collections import defaultdict
from tqdm import tqdm


# =========================================================
# 與訓練腳本 yolo11_detect_person_trf.py 完全一致的 cache meta
# 若訓練腳本的 meta 有更動，請同步修改此處
# =========================================================
CACHE_META = {
    "T": 30,
    "backbone": "mbv3_small.features@224",
    "norm": "imagenet",
    "roi": {
        "type": "align", "bins": [7, 7],
        "aligned": True, "sampling": "auto", "scale": "7/224"
    },
    "sampler": "uniform_ts+segment_max+nearest_fill",
    "cached_shape": [576, 7, 7],
    "head": "DSConv+GAP",
    "version": "v1"
}

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


# =========================================================
# 工具函式
# =========================================================
def md5_of_file(path: str, chunk_bytes: int = 65536) -> str:
    """逐塊讀取，計算檔案 MD5；大檔也不會爆記憶體。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


def make_cache_path(video_path: str, do_flip: bool,
                    cache_root: str) -> str:
    """還原訓練腳本的 cache 路徑（與 make_cache_path() 邏輯完全相同）。"""
    key_str = (
        json.dumps(CACHE_META, sort_keys=True)
        + "|" + os.path.abspath(video_path)
        + f"|flip={int(do_flip)}"
    )
    h = hashlib.md5(key_str.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(video_path))[0]
    fname = f"{stem}-{h}.npz"
    return os.path.join(cache_root, fname)


def collect_videos(data_root: str):
    """遞迴掃描 data_root，回傳所有影片路徑清單（已排序）。"""
    videos = []
    for root, _, files in os.walk(data_root):
        for fname in sorted(files):
            if fname.lower().endswith(VIDEO_EXTS):
                videos.append(os.path.join(root, fname))
    return sorted(videos)


def subject_of(path: str) -> str:
    """從路徑取出受試者 ID（P001~P006），找不到回傳 'unknown'。"""
    for part in path.replace("\\", "/").split("/"):
        if re.match(r"^P\d+$", part):
            return part
    return "unknown"


# =========================================================
# 主流程
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="找出並刪除資料集中的重複影片及其快取。"
    )
    ap.add_argument("--data_root",   default="data",
                    help="影片根目錄（預設 data）")
    ap.add_argument("--cache_root",  default=os.path.join("data", "video_features_v1"),
                    help="特徵快取資料夾（預設 data/video_features_v1）")
    ap.add_argument("--dry-run",     action="store_true",
                    help="只列出要刪的清單，不實際刪除任何檔案")
    ap.add_argument("--log",         default="dedup_log.txt",
                    help="刪除記錄輸出路徑（預設 dedup_log.txt）")
    args = ap.parse_args()

    dry = args.dry_run
    mode_tag = "[DRY-RUN]" if dry else "[LIVE]"
    print(f"\n{'='*55}")
    print(f"  資料集去重工具  {mode_tag}")
    print(f"  data_root  : {args.data_root}")
    print(f"  cache_root : {args.cache_root}")
    print(f"{'='*55}\n")

    # ---- Step 1：掃描所有影片 ----
    print("[1/3] 掃描影片檔案...")
    videos = collect_videos(args.data_root)
    print(f"      找到 {len(videos)} 支影片\n")
    if not videos:
        print("找不到任何影片，請確認 --data_root 是否正確。")
        return

    # ---- Step 2：計算 MD5，按 hash 分組 ----
    print("[2/3] 計算 MD5（檔案多時需要幾分鐘，請耐心等候）...")
    hash_to_paths: dict = defaultdict(list)
    for path in tqdm(videos, unit="file", ncols=80):
        try:
            h = md5_of_file(path)
            hash_to_paths[h].append(path)
        except Exception as e:
            print(f"\n  [警告] 無法讀取 {path}: {e}")

    # 只留有重複的組
    dup_groups = {
        h: sorted(paths)
        for h, paths in hash_to_paths.items()
        if len(paths) > 1
    }
    total_dup_files = sum(len(v) - 1 for v in dup_groups.values())

    print(f"\n      共找到 {len(dup_groups)} 組重複，需刪除 {total_dup_files} 支影片")

    if not dup_groups:
        print("\n資料集中沒有完全相同的重複影片，無需刪除。")
        # 仍寫一份空 log
        with open(args.log, "w", encoding="utf-8") as f:
            f.write("no_duplicates=True\n")
        return

    # ---- Step 3：逐組處理 ----
    print(f"\n[3/3] {'預覽' if dry else '執行'}刪除...\n")

    log_lines = [
        f"dry_run={dry}",
        f"data_root={os.path.abspath(args.data_root)}",
        f"cache_root={os.path.abspath(args.cache_root)}",
        "",
    ]

    deleted_videos = 0
    deleted_caches = 0
    skipped_caches = 0   # cache 存在但非本腳本的 root，略過

    for grp_idx, (h, paths) in enumerate(sorted(dup_groups.items()), start=1):
        keep      = paths[0]          # 保留第一個（字母序最小）
        to_delete = paths[1:]

        subjects = set(subject_of(p) for p in paths)
        print(f"  組 {grp_idx:02d} [MD5 {h[:10]}...]  "
              f"受試者={sorted(subjects)}")
        print(f"    ✓ 保留: {keep}")

        log_lines.append(f"=== 組 {grp_idx:02d} [MD5={h}] ===")
        log_lines.append(f"KEEP   {keep}")

        for p in to_delete:
            print(f"    ✗ 刪除: {p}")
            log_lines.append(f"DELETE {p}")

            # 尋找對應 cache（flip=False 與 flip=True 各一份）
            for flip in (False, True):
                cp = make_cache_path(p, flip, args.cache_root)
                if os.path.exists(cp):
                    flip_tag = "flip" if flip else "orig"
                    print(f"      - cache({flip_tag}): {os.path.basename(cp)}")
                    log_lines.append(f"  CACHE {cp}")
                    if not dry:
                        try:
                            os.remove(cp)
                            deleted_caches += 1
                        except Exception as e:
                            print(f"      [警告] 刪除 cache 失敗: {e}")
                else:
                    skipped_caches += 1   # 該 cache 不存在（正常，可能未產生）

            if not dry:
                try:
                    os.remove(p)
                    deleted_videos += 1
                except Exception as e:
                    print(f"    [警告] 刪除影片失敗: {e}")
                    log_lines.append(f"  ERROR {e}")

        log_lines.append("")

    # ---- 摘要 ----
    print(f"\n{'='*55}")
    if dry:
        print(f"  [DRY-RUN] 預計刪除影片 {total_dup_files} 支")
        print(f"  實際未刪除任何檔案，確認後去掉 --dry-run 再執行")
    else:
        print(f"  完成：刪除影片 {deleted_videos} 支、"
              f"cache {deleted_caches} 份")
    print(f"{'='*55}\n")

    # ---- 寫 log ----
    log_lines += [
        "",
        f"=== 摘要 ===",
        f"dry_run         = {dry}",
        f"dup_groups      = {len(dup_groups)}",
        f"deleted_videos  = {deleted_videos}",
        f"deleted_caches  = {deleted_caches}",
        f"skipped_caches  = {skipped_caches} (cache 不存在，無需刪除)",
    ]
    with open(args.log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"  記錄已寫入：{args.log}")


if __name__ == "__main__":
    main()
