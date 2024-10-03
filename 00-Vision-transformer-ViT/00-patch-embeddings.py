import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Görüntüyü aç ve boyutlandır
img = Image.open('/home/ahmet/workspace-L-T/Vision-transformer-ViT/demo02.jpg')
img = img.resize((224, 224))  # Kesin olarak 224x224 boyutunda
array_img = np.array(img)
print(array_img.shape)  # Boyut kontrolü

img_size = 224
patch_size = 16
num_patches = img_size // patch_size
assert img_size % patch_size == 0

print(f"Number of patches per row: {num_patches}\
\nNumber of patches per column: {num_patches}\
\nTotal patches: {num_patches*num_patches}\
\nPatch size: {patch_size} pixels x {patch_size} pixels")

fig, axs = plt.subplots(nrows=num_patches,
                        ncols=num_patches,
                        figsize=(num_patches, num_patches),
                        sharex=True,
                        sharey=True)

# Dilimleme işlemi ve görüntüleme
for i, patch_height in enumerate(range(0, img_size, patch_size)):
    for j, patch_width in enumerate(range(0, img_size, patch_size)):
        if patch_height + patch_size <= array_img.shape[0] and patch_width + patch_size <= array_img.shape[1]:
            axs[i, j].imshow(array_img[patch_height:patch_height+patch_size, patch_width:patch_width+patch_size, :])
            axs[i, j].set_ylabel(i+1,
                                 rotation="horizontal",
                                 horizontalalignment="right",
                                 verticalalignment="center")
            axs[i, j].set_xlabel(j+1)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()

plt.show()
