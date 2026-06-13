"""Remote observation viewer for the dual-Franka data collector.

Connects to a running Ray cluster, locates the ``DataCollector`` actor by
name (``EnvGroup:0``), and calls its ``get_obs()`` method to pull the
current raw observation from the underlying ``DualFrankaJointEnv``.

Usage:
    # On any machine that can reach the Ray head node:
    python get_obs_remote.py --address 192.168.120.143:6379

    # If running on the head node itself:
    python get_obs_remote.py

    # Continuous polling at 2 Hz:
    python get_obs_remote.py --loop --freq 2

    # Save the latest frames as PNG images:
    python get_obs_remote.py --save-dir /tmp/franka_frames
"""

import argparse
import os
import time

import numpy as np


ADDRESS = "192.168.120.143:6379"
ACTOR_NAME = "EnvGroup:0"
LOOP = False
FREQ = 1.0
SAVE_DIR = "/home/ubuntu/dais/Robo-dopamine/examples/results/ray_img"

def get_obs_once(collector_handle) -> dict:
    """Call ``get_obs.remote()`` and return the raw observation dict."""
    obs_list = collector_handle.get_obs.remote()
    obs = ray.get(obs_list)
    return obs


def print_obs_summary(obs: dict) -> None:
    """Print a human-readable summary of the observation."""
    frames = obs.get("frames", {})
    state = obs.get("state", {})

    print("--- Frames ---")
    for name, arr in frames.items():
        print(f"  {name:25s}  shape={arr.shape}  dtype={arr.dtype}")

    print("--- State ---")
    for name, arr in state.items():
        if isinstance(arr, np.ndarray):
            print(
                f"  {name:25s}  shape={arr.shape}  "
                f"min={arr.min():.4f}  max={arr.max():.4f}"
            )
        else:
            print(f"  {name:25s}  {arr}")


def save_frames(obs: dict, save_dir: str) -> None:
    """Save each camera frame as a PNG image."""
    os.makedirs(save_dir, exist_ok=True)
    try:
        import cv2
    except ImportError:
        print("cv2 not available; skipping frame save.")
        return

    for name, arr in obs.get("frames", {}).items():
        # arr is RGB, cv2 expects BGR
        bgr = arr[..., ::-1]
        path = os.path.join(save_dir, f"{name}.png")
        cv2.imwrite(path, bgr)
        print(f"  Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch observations from a running dual-Franka DataCollector."
    )
    parser.add_argument(
        "--address",
        default=ADDRESS,
        help="Ray cluster address (e.g. 192.168.120.143:6379). "
        "Omit to connect to a local Ray instance.",
    )
    parser.add_argument(
        "--actor-name",
        default="EnvGroup:0",
        help="Name of the DataCollector Ray actor (default: EnvGroup:0).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously poll for observations instead of fetching once.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=1.0,
        help="Polling frequency in Hz when --loop is set (default: 1.0).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save camera frames as PNG images.",
    )
    args = parser.parse_args()

    import ray

    # init_kwargs = {}
    # if args.address:
    #     init_kwargs["address"] = args.address
    #     # init_kwargs["_temp_dir"] = "/tmp/ray-ubuntu"
    #     # init_kwargs["_node_ip_address"] = args.address.split(":")[0]

    # # ray.init(**init_kwargs)
    # ray.init(address=args.address, _temp_dir="/tmp/ray-ubuntu",namespace="RLinf")

    if args.address:
        host = args.address.split(":")[0]
        # Ray Client mode: uses TCP, bypasses Unix socket permission issues
        # when running on the same machine as the Ray head node under a different user.
        ray.init(f"ray://{host}:10001",namespace="RLinf")
    else:
        ray.init()

    try:
        collector = ray.get_actor(args.actor_name,namespace="RLinf")
    except ValueError:
        print(
            f"Actor '{args.actor_name}' not found. "
            f"Is the DataCollector running? "
            f"Available actors: {ray.util.list_named_actors()}"
        )
        ray.shutdown()
        raise SystemExit(1)

    print(f"Connected to actor '{args.actor_name}'.\n")

    period = 1.0 / args.freq if args.freq > 0 else 1.0

    try:
        while True:
            t0 = time.perf_counter()
            obs = get_obs_once(collector)
            elapsed = time.perf_counter() - t0

            print(f"[{time.strftime('%H:%M:%S')}] get_obs() took {elapsed*1000:.1f} ms")
            print_obs_summary(obs)

            if args.save_dir:
                save_frames(obs, args.save_dir)

            if not args.loop:
                break

            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            print()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ray.shutdown()
