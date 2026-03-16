import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_geometry_orthogonal(file_path, output_path=None):
    """
    Plots Axial, Coronal, and Sagittal views for Binary Grid and SDF.
    """
    # 1. Load Data
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Loading {file_path}...")
    data = np.load(file_path)
    
    if 'binary' not in data or 'sdf' not in data:
        print("Error: File keys missing. Expected 'binary' and 'sdf'.")
        return

    binary_vol = data['binary'] # Shape: (D, H, W) -> (Z, Y, X)
    sdf_vol = data['sdf']
    
    # 2. Determine Mid-Slices
    D, H, W = binary_vol.shape
    mid_z = D // 2  # Axial
    mid_y = H // 2  # Coronal
    mid_x = W // 2  # Sagittal
    
    # 3. Setup Plot (2 Rows x 3 Cols)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row Definitions
    rows = ['Binary Mask', 'SDF']
    # Column Definitions
    cols = ['Axial (X-Y)', 'Coronal (X-Z)', 'Sagittal (Y-Z)']
    
    # --- Helper to plot consistent subplots ---
    def plot_slice(ax, slice_data, title, is_sdf=False):
        if is_sdf:
            # Centered colormap for SDF (Red=Out, Blue=In)
            limit = max(abs(slice_data.min()), abs(slice_data.max()))
            im = ax.imshow(slice_data, origin='lower', cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
            # Add contour at 0
            ax.contour(slice_data, levels=[0], colors='black', linewidths=0.5, origin='lower')
            return im
        else:
            # Grayscale for Binary
            im = ax.imshow(slice_data, origin='lower', cmap='gray', vmin=0, vmax=1, aspect='auto')
            return im

    # --- ROW 1: BINARY MASKS ---
    
    # 1. Axial (Z-slice) -> Shows X vs Y
    # Slice: [z, :, :]
    ax = axes[0, 0]
    im = plot_slice(ax, binary_vol[mid_z, :, :].T, f"Axial (z={mid_z})", is_sdf=False)
    ax.set_ylabel("Z (Height)")
    ax.set_xlabel("Y (Depth)")

    # 2. Coronal (Y-slice) -> Shows X vs Z
    # Slice: [:, y, :] -> Z is row index (y-axis of plot), X is col index
    ax = axes[0, 1]
    im = plot_slice(ax, binary_vol[:, mid_y, :].T, f"Coronal (y={mid_y})", is_sdf=False)
    ax.set_ylabel("Z (Height)")
    ax.set_xlabel("X (Width)")

    # 3. Sagittal (X-slice) -> Shows Y vs Z
    # Slice: [:, :, x] -> Z is row index, Y is col index
    ax = axes[0, 2]
    im = plot_slice(ax, binary_vol[:, :, mid_x].T, f"Sagittal (x={mid_x})", is_sdf=False)
    ax.set_ylabel("Y (Depth)")
    ax.set_xlabel("X (Width)")

    # --- ROW 2: SDF ---

    # 4. Axial SDF
    ax = axes[1, 0]
    im_sdf = plot_slice(ax, sdf_vol[mid_z, :, :].T, f"Axial SDF", is_sdf=True)
    ax.set_ylabel("Z (Height)")
    ax.set_xlabel("Y (Depth)")
    plt.colorbar(im_sdf, ax=ax, fraction=0.046, pad=0.04, label='Dist')

    # 5. Coronal SDF
    ax = axes[1, 1]
    im_sdf = plot_slice(ax, sdf_vol[:, mid_y, :].T, f"Coronal SDF", is_sdf=True)
    ax.set_ylabel("Z (Height)")
    ax.set_xlabel("X (Width)")
    plt.colorbar(im_sdf, ax=ax, fraction=0.046, pad=0.04, label='Dist')

    # 6. Sagittal SDF
    ax = axes[1, 2]
    im_sdf = plot_slice(ax, sdf_vol[:, :, mid_x].T, f"Sagittal SDF", is_sdf=True)
    ax.set_ylabel("Y (Depth)")
    ax.set_xlabel("X (Width)")
    plt.colorbar(im_sdf, ax=ax, fraction=0.046, pad=0.04, label='Dist')

    # Formatting
    # plt.suptitle(f"Geometry Orthogonal Views: {os.path.basename(file_path)}", fontsize=16)
    plt.suptitle(f"Geometry Orthogonal Views", fontsize=16)
    plt.tight_layout()

    # 4. Save or Show
    if output_path:
        plt.savefig(output_path, dpi=300)
        # Save in PDF
        plt.savefig(output_path.replace(".png", ".pdf"))
        print(f"Comparison plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Axial, Sagittal, and Coronal views of processed geometry.")
    parser.add_argument("file_path", help="Path to the .npz geometry file.")
    parser.add_argument("--output", "-o", default=None, help="Path to save the image (optional).")
    
    args = parser.parse_args()
    
    plot_geometry_orthogonal(args.file_path, args.output)