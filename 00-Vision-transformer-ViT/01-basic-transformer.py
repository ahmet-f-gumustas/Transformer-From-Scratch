from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import timm
import torch
import urllib

# Create a Vision Transformer (ViT) model with pre-trained weights
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()  # Set the model to evaluation mode

# Print the model architecture
print(model)
print('##################################################################################')

# Resolve data configuration for the model (input size, normalization, etc.)
config = resolve_data_config({}, model=model)
# Create the transform pipeline based on the model's data configuration
transform = create_transform(**config)

# Open and convert the image to RGB
img = Image.open('/home/ahmet/workspace-L-T/Vision-transformer-ViT/demo02.jpg').convert('RGB')

# Transform the image into a tensor and add a batch dimension
tensor = transform(img).unsqueeze(0)

# Run the model inference without tracking gradients (evaluation mode)
with torch.no_grad():
    out = model(tensor)

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)

# Download ImageNet class labels
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 

# Read ImageNet class labels from the file
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Get the top 5 probabilities and their corresponding categories
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    # Print the class names and the corresponding probabilities
    print(categories[top5_catid[i]], top5_prob[i].item())
