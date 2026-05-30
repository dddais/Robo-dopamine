"""
从 MP4 视频中提取指定帧并保存为 PNG 图片。

直接修改下方配置区即可使用，无需命令行参数。
"""

import os
import sys

import cv2


# ===================== 配置区 =====================
# VIDEO_PATH = "/home/dais/workspace/bag2video/pick_3_suc@MASTER_SLAVE_MODE@2026_05_12_23_05_32/cam_high.mp4"
VIDEO_PATH = "/mnt/public1/dais/data_xuzhexuan/2025-challenge-demos/episode_1/cam_high.mp4"
OUTPUT_DIR = "/home/dais/workspace/Robo-Dopamine/examples/"

# 选择模式（取消注释你想用的那一行，其余保持注释）
MODE = "indices"       # 按帧索引提取
# MODE = "timestamps"  # 按时间戳（秒）提取
# MODE = "range"       # 按范围提取
# MODE = "all"         # 提取所有帧

# --- 模式参数（按需填写） ---
FRAME_INDICES = [1400]        # MODE="indices" 时生效
TIMESTAMPS = [9]                # MODE="timestamps" 时生效，单位：秒
RANGE = (0, 600, 10)                         # MODE="range" 时生效: (起始帧, 结束帧, 步长)
# ================================================


def extract_by_indices(video_path, output_dir, frame_indices):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {total} 帧, {fps:.2f} FPS, 时长 {total / fps:.2f}s")

    frame_indices = sorted(set(i for i in frame_indices if 0 <= i < total))
    if not frame_indices:
        print("警告: 没有有效的帧索引")
        cap.release()
        return

    saved = 0
    current = 0
    for target in frame_indices:
        if target > current:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 无法读取第 {target} 帧")
            current = target + 1
            continue
        out_path = os.path.join(output_dir, f"frame_{target:06d}.png")
        cv2.imwrite(out_path, frame)
        saved += 1
        current = target + 1

    cap.release()
    print(f"已保存 {saved} 帧到 {output_dir}")


def extract_by_timestamps(video_path, output_dir, timestamps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    print(f"视频信息: {total} 帧, {fps:.2f} FPS, 时长 {duration:.2f}s")

    saved = 0
    for ts in sorted(timestamps):
        if ts < 0 or ts > duration:
            print(f"警告: 时间戳 {ts}s 超出范围 [0, {duration:.2f}]，跳过")
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 无法读取 {ts}s 处的帧")
            continue
        out_path = os.path.join(output_dir, f"frame_{ts:.2f}s.png")
        cv2.imwrite(out_path, frame)
        saved += 1

    cap.release()
    print(f"已保存 {saved} 帧到 {output_dir}")


def extract_by_range(video_path, output_dir, start, end, step):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {total} 帧, {fps:.2f} FPS, 时长 {total / fps:.2f}s")

    end = min(end, total - 1) if end >= 0 else total - 1
    indices = list(range(start, end + 1, step))

    saved = 0
    current = -1
    for target in indices:
        if target > current:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 无法读取第 {target} 帧")
            current = target + 1
            continue
        out_path = os.path.join(output_dir, f"frame_{target:06d}.png")
        cv2.imwrite(out_path, frame)
        saved += 1
        current = target + 1
        if saved % 50 == 0:
            print(f"  已处理 {saved}/{len(indices)} 帧...")

    cap.release()
    print(f"已保存 {saved} 帧到 {output_dir}")


def extract_all(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {total} 帧, {fps:.2f} FPS, 时长 {total / fps:.2f}s")

    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"frame_{saved:06d}.png")
        cv2.imwrite(out_path, frame)
        saved += 1
        if saved % 100 == 0:
            print(f"  已处理 {saved}/{total} 帧...")

    cap.release()
    print(f"已保存 {saved} 帧到 {output_dir}")


def main():
    if not os.path.isfile(VIDEO_PATH):
        print(f"错误: 文件不存在 {VIDEO_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if MODE == "indices":
        extract_by_indices(VIDEO_PATH, OUTPUT_DIR, FRAME_INDICES)
    elif MODE == "timestamps":
        extract_by_timestamps(VIDEO_PATH, OUTPUT_DIR, TIMESTAMPS)
    elif MODE == "range":
        extract_by_range(VIDEO_PATH, OUTPUT_DIR, *RANGE)
    elif MODE == "all":
        extract_all(VIDEO_PATH, OUTPUT_DIR)
    else:
        print(f"错误: 未知模式 '{MODE}'，可选: indices / timestamps / range / all")


if __name__ == "__main__":
    main()
