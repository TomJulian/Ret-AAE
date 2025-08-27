import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from models import newAECBAM3
import yaml

file="./config.yaml"

config = yaml.load(open(file, "r"), Loader=yaml.FullLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = (1, 224, 224)
latent_dim = 256

model = newAECBAM3(input_dim, latent_dim).to(device)

def load_encoder_only_into_full(model, enc_ckpt_path, strict=False):
    ckpt = torch.load(enc_ckpt_path, map_location="cpu")
    enc_sd = ckpt["encoder_state_dict"]
    res = model.load_state_dict(enc_sd, strict=strict)
    print("Missing keys:", res.missing_keys)
    print("Unexpected keys:", res.unexpected_keys)

load_encoder_only_into_full(model, "./v1_OCT_encoder.pt", strict=False)


model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

images_dir = config['image_dir']
out_dir = config['out_dir']
csv_path = os.path.join(out_dir, "OCT_embeddings.csv")

rows = []

for image_name in os.listdir(images_dir):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        image_path = os.path.join(images_dir, image_name)
        img = Image.open(image_path).convert("L")
        input_image = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            latent_vector = model.encode(input_image)
            latent_vector = latent_vector.detach().cpu().squeeze(0)
        latent_list = latent_vector.tolist()
        row = {"image_name": image_name}
        for i, value in enumerate(latent_list):
            row[f"embedding_{i}"] = value
        rows.append(row)

df = pd.DataFrame(rows)

df.to_csv(csv_path, index=False)

print(df.head())
