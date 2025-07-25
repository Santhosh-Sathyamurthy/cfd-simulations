# MIT License
# Copyright (c) 2025 Santhosh S
# See LICENSE file for full license text.

from pathlib import Path

# === Configuration ===
base_path = Path("/home/santhosh/projects/julia/julia-cfd-simulations/julia/youtube_kolmogorov_turbulence/kolmogorov_icefire/")
keep_every = 50  # Keep one frame every 20 (e.g., 0, 20, 40, ..., 200)

# === Collect files ===
to_delete = []
to_keep = []
deleted_frames = []

for prefix in ["velocity", "vorticity", "kolmogorov"]:
    for file in sorted(base_path.glob(f"{prefix}_*.png")):
        try:
            frame = int(file.stem.split("_")[-1])
            if frame % keep_every == 0:
                to_keep.append(file)
            else:
                to_delete.append(file)
                deleted_frames.append((prefix, frame))
        except ValueError:
            continue

# === Report ===
print("\n📁 Folder:", base_path)
print(f"🖼️ Total PNGs found: {len(to_keep) + len(to_delete)}")
print(f"✅ PNGs to keep (every {keep_every} frames): {len(to_keep)}")
print(f"🗑️ PNGs to delete: {len(to_delete)}")

# # === Preview deletions ===
# if deleted_frames:
#     print("\n🧾 Frame numbers to be deleted:")
#     for prefix, frame in deleted_frames:
#         print(f"  - {prefix}_{frame:06d}.png")

# === Ask for Confirmation ===
confirm = input("\n❓ Proceed with deletion? (yes/no): ").strip().lower()
if confirm == "yes":
    for f in to_delete:
        f.unlink()
    print(f"\n✅ Deleted {len(to_delete)} PNGs. Kept {len(to_keep)}.")
else:
    print("\n⚠️ Deletion canceled. No files were removed.")
