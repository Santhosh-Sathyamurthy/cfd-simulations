from pathlib import Path
import re

# === CONFIGURATION ===
ver = 5
Re_value = 600
base_path = Path(f"/home/santhosh/projects/julia/julia-cfd-simulations/python/flow_over_cylinder (Fischer)/v{str(ver)}_re_{str(Re_value)}/velocity_frames")
duration_s = 30.0                # Total simulation time in seconds
fps_to_keep = 1                  # How many frames per second to retain
prefixes = ["velocity_frame", "vorticity_frame"]

# === Main Logic Per Prefix ===
total_deleted = 0
total_kept = 0

for prefix in prefixes:
    print(f"\n--- 📁 Processing prefix: {prefix}_*.png ---")
    
    # Collect all matching PNGs for this prefix
    files = sorted(base_path.glob(f"{prefix}_*.png"))
    frame_files = []
    frame_numbers = []

    for file in files:
        match = re.search(r"_(\d{6})\.png$", file.name)
        if match:
            frame = int(match.group(1))
            frame_files.append((frame, file))
            frame_numbers.append(frame)

    if not frame_files:
        print("🚫 No PNGs found for this prefix.")
        continue

    frame_numbers.sort()
    min_frame = frame_numbers[0]
    max_frame = frame_numbers[-1]
    total_pngs = len(frame_files)

    # Calculate time step based on the number of PNGs and duration
    estimated_dt = duration_s / total_pngs  # Time per PNG
    frames_per_second = 1 / estimated_dt    # Original FPS
    target_frames = int(duration_s * fps_to_keep)  # Total frames to keep
    keep_every = max(1, total_pngs // target_frames) if target_frames > 0 else total_pngs

    # Select frames to keep at even intervals
    to_keep = []
    to_delete = []

    # Calculate ideal frame indices to keep
    ideal_indices = [round(i * total_pngs / target_frames) for i in range(target_frames)]
    ideal_frames = [frame_numbers[min(idx, total_pngs - 1)] for idx in ideal_indices]

    for frame, file in frame_files:
        # Keep the frame if it's close to an ideal frame number
        closest_ideal = min(ideal_frames, key=lambda x: abs(x - frame))
        if frame == closest_ideal and file not in to_keep:
            to_keep.append(file)
        else:
            to_delete.append(file)

    print(f"📈 Total PNGs: {total_pngs}")
    print(f"🧮 Frame range: {min_frame} to {max_frame}")
    print(f"⏱️ Estimated dt: {estimated_dt:.6f} s/iteration")
    print(f"🎞️ Original FPS: {frames_per_second:.2f}")
    print(f"🎯 Target FPS: {fps_to_keep}, keeping every ~{keep_every:.1f} frames")
    print(f"✅ To keep: {len(to_keep)}")
    print(f"🗑️ To delete: {len(to_delete)}")

    # Ask confirmation
    confirm = input(f"\n❓ Proceed with deleting {len(to_delete)} {prefix} PNGs? (yes/no): ").strip().lower()
    if confirm == "yes":
        for f in to_delete:
            try:
                f.unlink()
                print(f"🗑️ Deleted: {f}")
            except Exception as e:
                print(f"⚠️ Error deleting {f}: {e}")
        print(f"✅ Deleted {len(to_delete)} files for {prefix}")
        total_deleted += len(to_delete)
        total_kept += len(to_keep)
    else:
        print("⚠️ Skipping deletion for this prefix.")

# === Summary ===
print(f"\n=== ✅ DONE ===")
print(f"📂 Folder: {base_path}")
print(f"🖼️ Total kept: {total_kept}")
print(f"🗑️ Total deleted: {total_deleted}")