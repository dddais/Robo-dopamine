"""
用预训练 GRM 模型对用户自己的数据（抓取胡萝卜放在黄色盘子上）进行推理测试。
测试三种模式：forward, incremental, backward
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from examples.inference import GRMInference

# ============================
# 配置
# ============================
MODEL_PATH = "./pretrained_models/GRM-2.0-4B-Preview"
DATA_DIR = "./aligned_data"   # 帧对齐后的数据
OUTPUT_ROOT = "./results/carrot_pick"

TASK_INSTRUCTION = "pick the carrot and place it on the yellow plate"
GOAL_IMAGE = "./examples/blank_goal.png"

# ============================
# 加载模型
# ============================
print(f"Loading model: {MODEL_PATH} ...")
model = GRMInference(MODEL_PATH)
print("Model loaded successfully!")

# ============================
# 运行三种模式的推理
# ============================
results = {}
for mode in ["forward", "incremental", "backward"]:
    print(f"\n{'='*60}")
    print(f"Running {mode}-mode inference...")
    print(f"{'='*60}")

    try:
        output_dir = model.run_pipeline(
            cam_high_path  = os.path.join(DATA_DIR, "cam_high.mp4"),
            cam_left_path  = os.path.join(DATA_DIR, "cam_left_wrist.mp4"),
            cam_right_path = os.path.join(DATA_DIR, "cam_right_wrist.mp4"),
            out_root       = OUTPUT_ROOT,
            task           = TASK_INSTRUCTION,
            frame_interval = 10,
            batch_size     = 1,
            goal_image     = GOAL_IMAGE,
            eval_mode      = mode,
            visualize      = True
        )
        results[mode] = output_dir
        print(f"[{mode}] Output at: {output_dir}")
    except Exception as e:
        print(f"[{mode}] FAILED: {e}")
        import traceback
        traceback.print_exc()

# ============================
# 汇总结果
# ============================
print(f"\n{'='*60}")
print("All inference results:")
for mode, path in results.items():
    print(f"  {mode}: {path}")

# 尝试读取并打印 reward 结果
for mode, path in results.items():
    reward_file = os.path.join(path, "rewards.json")
    if os.path.exists(reward_file):
        with open(reward_file) as f:
            rewards = json.load(f)
        print(f"\n[{mode}] Reward data: {json.dumps(rewards, indent=2)[:500]}")

print(f"{'='*60}")
