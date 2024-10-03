# Vision Transformer (ViT) Example

This project demonstrates two key functionalities of Vision Transformers (ViT):

1. **Patch Visualization of an Image**: Splitting an image into patches and visualizing these patches.
2. **Image Classification Using a Pretrained ViT Model**: Using a pretrained Vision Transformer to classify an image into one of the ImageNet categories.

## Installation

To get started with the project, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/ahmet-f-gumustas/Transformer-From-Scratch.git
cd vision-transformer-example
pip install -r requirements.txt
python3 patch_visualization.py  # Patch visualization of an Image
python3 image_classification.py # Image Classification Usin Pretrained ViT Model
```
#### Outout Example

tabby, 0.7253
lynx, 0.1124
Egyptian cat, 0.0583
tiger cat, 0.0329
cougar, 0.0128

## File Structure

.
├── patch_visualization.py      # Script for patch visualization
├── image_classification.py     # Script for image classification using pretrained ViT model
├── README.md                   # Project documentation
├── requirements.txt            # Required dependencies
└── demo02.jpg                  # Example image for testing

