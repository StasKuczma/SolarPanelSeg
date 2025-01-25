import os

def synchronize_images_and_masks(img_dir, mask_dir):
    # Get sorted lists of image and mask filenames
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Extract base filenames (without extensions)
    img_base = {os.path.splitext(f)[0] for f in img_files}
    mask_base = {os.path.splitext(f)[0] for f in mask_files}

    # Find unmatched files
    unmatched_imgs = [f for f in img_files if os.path.splitext(f)[0] not in mask_base]
    unmatched_masks = [f for f in mask_files if os.path.splitext(f)[0] not in img_base]

    # Delete unmatched images
    for file in unmatched_imgs:
        os.remove(os.path.join(img_dir, file))
        print(f"Deleted unmatched image: {file}")

    # Delete unmatched masks
    for file in unmatched_masks:
        os.remove(os.path.join(mask_dir, file))
        print(f"Deleted unmatched mask: {file}")

    print("Synchronization complete!")

# Define paths to img and mask directories
img_dir = "/workspace/data/bdappv/google/img"
mask_dir = "/workspace/data/bdappv/google/mask"

# Synchronize the directories
synchronize_images_and_masks(img_dir, mask_dir)

