from pathlib import Path
import re

# === CONFIGURATION ===
base_path = Path("/home/santhosh/projects/julia/julia-cfd-simulations/python/flow_over_cylinder (Fischer)/v3_re_200")
duration_s = 30.0                # Total simulation time in seconds
fps_to_keep = 1                  # How many frames per second to retain
prefixes = ["velocity", "vorticity", "turbulent_frame"]

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

    # Estimate dt and keep stride
    total_iters = max_frame - min_frame
    estimated_dt = duration_s / (max_frame - min_frame + 1)
    seconds_per_png = duration_s / total_pngs
    frames_per_second = 1 / seconds_per_png
    keep_every = max(1, round(frames_per_second / fps_to_keep))

    # Select frames to keep/delete
    to_keep = []
    to_delete = []

    for frame, file in frame_files:
        if (frame - min_frame) % keep_every == 0:
            to_keep.append(file)
        else:
            to_delete.append(file)

    print(f"📈 Total PNGs: {total_pngs}")
    print(f"🧮 Frame stride (est): {(max_frame - min_frame) / total_pngs:.1f}")
    print(f"⏱️ Estimated dt: {estimated_dt:.6f} s/iteration")
    print(f"🎯 Keeping {fps_to_keep} frame(s)/s => keep every {keep_every} PNGs")
    print(f"✅ To keep: {len(to_keep)}")
    print(f"🗑️ To delete: {len(to_delete)}")

    # Ask confirmation
    confirm = input(f"\n❓ Proceed with deleting {len(to_delete)} {prefix} PNGs? (yes/no): ").strip().lower()
    if confirm == "yes":
        for f in to_delete:
            f.unlink()
        print(f"✅ Deleted {len(to_delete)}")
        total_deleted += len(to_delete)
        total_kept += len(to_keep)
    else:
        print("⚠️ Skipping deletion.")

# === Summary ===
print(f"\n=== ✅ DONE ===")
print(f"📂 Folder: {base_path}")
print(f"🖼️ Total kept: {total_kept}")
print(f"🗑️ Total deleted: {total_deleted}")
