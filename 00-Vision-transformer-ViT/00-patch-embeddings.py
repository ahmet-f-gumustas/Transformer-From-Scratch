import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Open the image and resize it
img = Image.open('/home/ahmet/workspace-L-T/Vision-transformer-ViT/demo02.jpg')
img = img.resize((224, 224))  # Ensure the image is resized to 224x224
array_img = np.array(img)
print(array_img.shape)  # Print the shape of the image array for verification

img_size = 224
patch_size = 16
num_patches = img_size // patch_size  # Calculate the number of patches
assert img_size % patch_size == 0  # Ensure that the image size is divisible by the patch size

# Print patch information
print(f"Number of patches per row: {num_patches}\
\nNumber of patches per column: {num_patches}\
\nTotal patches: {num_patches*num_patches}\
\nPatch size: {patch_size} pixels x {patch_size} pixels")

# Create a subplot grid for displaying patches
fig, axs = plt.subplots(nrows=num_patches,
                        ncols=num_patches,
                        figsize=(num_patches, num_patches),
                        sharex=True,  # Share x-axis
                        sharey=True)  # Share y-axis

# Loop through image and display each patch
for i, patch_height in enumerate(range(0, img_size, patch_size)):  # Loop through rows
    for j, patch_width in enumerate(range(0, img_size, patch_size)):  # Loop through columns
        if patch_height + patch_size <= array_img.shape[0] and patch_width + patch_size <= array_img.shape[1]:
            # Display the current patch (slice of the image)
            axs[i, j].imshow(array_img[patch_height:patch_height+patch_size, patch_width:patch_width+patch_size, :])
            
            # Set labels for clarity (optional, just showing row/column indices)
            axs[i, j].set_ylabel(i+1,
                                 rotation="horizontal",
                                 horizontalalignment="right",
                                 verticalalignment="center")
            axs[i, j].set_xlabel(j+1)
            
            # Remove tick marks for a cleaner view
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()  # Hide inner labels

# Display the plot
plt.show()
