"""probe_xvisio.py — 逐个分辨率测试，找出普通 RGB."""
import cv2
import numpy as np

SERIAL = "/dev/v4l/by-id/usb-XVisio_Technology_XVisio_vSLAM_250801DR48FB26001173-video-index0"

# 逐个测试不同配置
tests = [
    ("NV12",  "640x1600",  cv2.VideoWriter_fourcc(*"NV12"), 640,  1600),
    ("YU12",  "640x480",   cv2.VideoWriter_fourcc(*"YU12"), 640,  480),
    ("YU12",  "1280x720",  cv2.VideoWriter_fourcc(*"YU12"), 1280, 720),
    ("YU12",  "1280x1280", cv2.VideoWriter_fourcc(*"YU12"), 1280, 1280),
]

for fmt_name, res, fourcc, w, h in tests:
    print(f"\n--- Testing {fmt_name} {res} ---")
    cap = cv2.VideoCapture(SERIAL, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # 不自动转换，看原始数据
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  请求: {w}x{h}, 实际: {actual_w}x{actual_h}")

    ok, raw = cap.read()
    if ok and raw is not None:
        print(f"  raw shape: {raw.shape}, dtype: {raw.dtype}")
        # YU12/I420: 手动转 BGR
        if fmt_name == "YU12":
            try:
                yuv = np.ascontiguousarray(raw).reshape(h * 3 // 2, w)
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                out_path = f"/tmp/xvisio_{fmt_name}_{res}.png"
                cv2.imwrite(out_path, bgr)
                print(f"  已保存: {out_path}")
            except Exception as e:
                print(f"  I420转换失败: {e}")
        elif fmt_name == "NV12":
            try:
                yuv = np.ascontiguousarray(raw).reshape(h * 3 // 2, w)
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                out_path = f"/tmp/xvisio_{fmt_name}_{res}.png"
                cv2.imwrite(out_path, bgr)
                print(f"  已保存: {out_path}")
            except Exception as e:
                print(f"  NV12转换失败: {e}")
    else:
        print(f"  读取失败")
    cap.release()


# 尝试 YU12 格式的 RGB 分辨率
for idx in [6, 8]:
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YU12'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ret, frame = cap.read()
    if ret:
        print(f"video{idx}: Got frame {frame.shape}")
        cv2.imwrite(f"test_video{idx}.png", frame)
    cap.release()

print("\n完成！检查 /tmp/xvisio_*.png 看哪个是普通 RGB。")