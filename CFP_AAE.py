import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from models import newAECBAM4
import yaml

file="./config.yaml"

config = yaml.load(open(file, "r"), Loader=yaml.FullLoader)

# Set device and model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = (3, 224, 224)  # Adjust as needed
latent_dim = 256  # Adjust as needed

# Instantiate your model
model = newAECBAM4(input_dim, latent_dim).to(device)

# Load the checkpoint (adjust the path and key names as needed)
checkpoint_path = "/home/dnanexus/CFP_encoder.pt"
#checkpoint = torch.load(checkpoint_path, map_location=device)
#model.load_state_dict(checkpoint['state_dict'])  # Or simply: model.load_state_dict(checkpoint

def load_encoder_only_into_full(model, enc_ckpt_path, strict=False):
    ckpt = torch.load(enc_ckpt_path, map_location="cpu")
    enc_sd = ckpt["encoder_state_dict"]
    res = model.load_state_dict(enc_sd, strict=strict)  # returns an object
    print("Missing keys:", res.missing_keys)
    print("Unexpected keys:", res.unexpected_keys)

load_encoder_only_into_full(model, "/home/dnanexus/CFP_encoder.pt", strict=False)


model.eval()

# Define the image transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Path to the directory containing images
images_dir = config['image_dir']
out_dir = config['out_dir']

# List to store the table rows
rows = []

# Iterate through each file in the directory
for image_name in os.listdir(images_dir):
    # Check if the file has an image extension
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        image_path = os.path.join(images_dir, image_name)
        img = Image.open(image_path).convert("RGB")
        input_image = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            latent_vector = model.encode(input_image)
            latent_vector = latent_vector.detach().cpu().squeeze(0)
        latent_list = latent_vector.tolist()
        row = {"image_name": image_name}
        for i, value in enumerate(latent_list):
            row[f"embedding_{i}"] = value
        rows.append(row)

# Create a DataFrame from the collected rows
df = pd.DataFrame(rows)

# Optionally, save the DataFrame to a CSV file
df.to_csv(f'"{out_dir}CFP_embeddings.csv", index=False)

# Display the first few rows of the table
print(df.head())