import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import numpy as np
from PIL import Image
import os

# Configuration
output_dir = "turbulent_cylinder_results_re_80"  # Directory containing PNG frames
output_file = "vorticity_re_80.mp4"     # Output animation file
dpi = 150                                      # Resolution of output
fps = 10                                       # Frames per second for animation

def main():
    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    ax.set_axis_off()  # Hide axes for cleaner visualization

    # Get list of frame files
    frame_path = Path(output_dir)
    frame_files = sorted(frame_path.glob("vorticity_frame_*.png"))  # Sort by filename
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {output_dir}")

    print(f"Found {len(frame_files)} frames in {output_dir}")

    # Load first frame to initialize
    img = Image.open(frame_files[0])
    img_array = np.array(img)
    im = ax.imshow(img_array, animated=True)

    def update(frame_file):
        """Update function for animation."""
        print(f"Processing {frame_file}")
        img = Image.open(frame_file)
        img_array = np.array(img)
        im.set_array(img_array)
        return [im]

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_files,
        interval=1000 / fps,  # Convert fps to milliseconds
        blit=True
    )

    # Save animation
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(output_file, writer=writer, dpi=dpi)
    print(f"Animation saved as {output_file}")

    plt.close(fig)

if __name__ == "__main__":
    main()