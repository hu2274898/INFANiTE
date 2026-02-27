import os
import numpy as np
import nibabel as nib

# ---------------------------------------------------------------------
# What are mask1 / mask2 / mask3 / img?
#
# mask1 (mask1_path):
#   A voxel-identifier / index map A defined in the *original* stack space.
#   Each voxel location is assigned a unique integer ID (0 = background).
#   This is used to distinguish truly observed voxels from interpolated ones.
#
# mask2 (mask2_path):
#   The transformed identifier map A after registration / resampling into
#   the template space. Because resampling introduces interpolation, the
#   values in mask2 may become non-integer or slightly shifted, but they
#   still encode where each original voxel ID ends up in the aligned space.
#
# mask3 (mask3_path, generated):
#   A sparse "observed-voxel locator" map built from (mask1, mask2).
#   For every non-zero ID present in mask1, we find the voxel in mask2 whose
#   value is closest to that ID, and mark that voxel with the ID in mask3.
#   This yields (approximately) one voxel per original observed location in
#   the template space, resolving collisions by keeping the closest match.
#   In other words, mask3 provides the template-space positions that are
#   treated as truly observed (high-confidence) supervision targets.
#
# img (img_path):
#   The aligned isotropic image volume in template space (e.g., registered
#   thick-slice reconstruction / stack). We use mask3 to keep only voxels
#   corresponding to observed locations (mask3 != 0) and zero-out all others
#   (mask3 == 0), producing a masked image for spatially weighted training.
# ---------------------------------------------------------------------

mask1_path = "./T2_reg_voxel_index/0015/602_voxel.nii.gz"
mask2_path = "./T2_reg_voxel_index/0015/602_permute_reg_template_voxel.nii.gz"
img_path   = "./T2_reg_voxel_index/0015/602_permute_reg_template.nii.gz"

out_dir = os.path.dirname(mask2_path)
mask3_path = os.path.join(out_dir, "602_permute_reg_template_voxel_mask3_nearest.nii.gz")
masked_img_path = os.path.join(out_dir, "602_permute_reg_template_masked_by_mask3_nearest.nii.gz")

def load_nii(path):
    nii = nib.load(path)
    data = nii.get_fdata()  # float64
    return nii, data

def main():
    m1_nii, m1 = load_nii(mask1_path)
    m2_nii, m2 = load_nii(mask2_path)

    # mask1: extract non-zero labels (mask1 itself is an integer label map)
    labels = np.unique(m1.astype(np.int64))
    labels = labels[labels != 0]
    labels = np.sort(labels)

    print(f"mask1 labels: {labels.size} (non-zero)")

    # mask2: flatten and sort once (key optimization)
    flat = m2.reshape(-1)
    order = np.argsort(flat)               # sorted indices
    flat_sorted = flat[order]              # sorted values

    # output mask3: int label map (each label occupies only one voxel)
    mask3_flat = np.zeros_like(flat, dtype=np.int32)

    # to avoid multiple labels claiming the same voxel: use best_dist to store the current assigned "distance" for each voxel
    best_dist = np.full_like(flat, np.inf, dtype=np.float64)

    # for each label, use binary search to find the nearest value in mask2
    for l in labels:
        pos = np.searchsorted(flat_sorted, l)

        cand = []
        if pos < flat_sorted.size:
            cand.append(pos)
        if pos > 0:
            cand.append(pos - 1)

        # pick the candidate with the smallest distance
        best_pos = None
        best_d = None
        for p in cand:
            d = abs(flat_sorted[p] - l)
            if best_d is None or d < best_d:
                best_d = d
                best_pos = p

        chosen_flat_idx = order[best_pos]  # map back to the original flattened index

        # conflict resolution: if this voxel has been assigned before, keep the label with the smaller "distance"
        if best_d < best_dist[chosen_flat_idx]:
            # if a label was previously assigned, clear it (optional; the line below will overwrite it anyway)
            mask3_flat[chosen_flat_idx] = int(l)
            best_dist[chosen_flat_idx] = best_d

    mask3 = mask3_flat.reshape(m2.shape)

    # save mask3
    m3_nii = nib.Nifti1Image(mask3.astype(np.int32), affine=m2_nii.affine, header=m2_nii.header)
    m3_nii.set_data_dtype(np.int32)
    nib.save(m3_nii, mask3_path)
    print(f"mask3 saved: {mask3_path}")
    print(f"mask3 nonzero voxels: {np.count_nonzero(mask3)}")

    # apply to the template image: set voxels to 0 where mask3==0
    img_nii, img = load_nii(img_path)
    if img.shape != mask3.shape:
        raise ValueError(f"Shape mismatch: img {img.shape} vs mask3 {img.shape}")

    masked = img.copy()
    masked[mask3 == 0] = 0

    out_nii = nib.Nifti1Image(masked.astype(np.float32), affine=img_nii.affine, header=img_nii.header)
    out_nii.set_data_dtype(np.float32)
    nib.save(out_nii, masked_img_path)
    print(f"save masked image saved: {masked_img_path}")

if __name__ == "__main__":
    main()
