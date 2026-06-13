"""
视频帧数对齐工具：自动读取目标文件夹中三个相机视频的帧数，
取最小帧数截齐，输出到指定目录。

用法:
    python clip_video.py --src <源目录> --dst <输出目录>

示例:
    python clip_video.py \
        --src /home/dais/workspace/bag2video/pick_3_suc@MASTER_SLAVE_MODE@2026_05_12_23_05_32 \
        --dst ./aligned_data_suc
"""
import argparse
import os

import cv2

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def get_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def clip_video(src: str, dst: str, target_frames: int) -> int:
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst, fourcc, fps, (w, h))

    count = 0
    while count < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        count += 1

    cap.release()
    writer.release()
    return count


def main():
    # parser = argparse.ArgumentParser(description="对齐三个相机视频的帧数")
    # parser.add_argument("--src", required=True, help="源目录，包含 cam_high.mp4 / cam_left_wrist.mp4 / cam_right_wrist.mp4")
    # parser.add_argument("--dst", required=True, help="输出目录")
    # args = parser.parse_args()

    SRC = "/mnt/public1/dais/data_xuzhexuan/2025-challenge-demos/episode_2"
    DST = "/home/dais/workspace/Robo-Dopamine/aligned_data/xzx_episode_2"
    os.makedirs(DST, exist_ok=True)


    # 1. 读取三个视频的帧数
    frame_counts = {}
    for cam in CAMERAS:
        path = os.path.join(SRC, f"{cam}.mp4")
        if not os.path.exists(path):
            print(f"[ERROR] 找不到 {path}")
            return
        frame_counts[cam] = get_frame_count(path)

    min_frames = min(frame_counts.values())
    # min_frames = 250

    print(f"源目录: {SRC}")
    print(f"帧数统计:")
    for cam, cnt in frame_counts.items():
        tag = " <-- 最短" if cnt == min_frames else ""
        print(f"  {cam}: {cnt} 帧{tag}")
    print(f"对齐目标: {min_frames} 帧")
    print()

    # 2. 截齐到最短帧数
    for cam in CAMERAS:
        src = os.path.join(SRC, f"{cam}.mp4")
        dst = os.path.join(DST, f"{cam}.mp4")
        wrote = clip_video(src, dst, min_frames)
        print(f"  {cam}: {wrote} 帧 -> {dst}")

    print(f"\n完成! 对齐后的视频已保存到 {DST}")


if __name__ == "__main__":
    main()
