# Ret-AAE

This is the public repo for the Ret-AAE models (adversarial autoencoders for optical coherence tomography and colour fundus photographs). The encoder is shared, but the decoder cannot be shared, given UK Biobank’s policies on generative AI trained using their data. The encoder can be utilised to produce normally distributed vector embeddings to represent ophthalmic images. These vector embeddings can be used to explore associations between ophthalmic features and other traits. 

The pre-print paper related to this repo is available at:  
https://www.medrxiv.org/content/10.1101/2025.08.04.25332962v1.full-text

---

## Before running this model...
These models were trained using UK Biobank data. Therefore, they have been validated on images captured using the **TOPCON 3D OCT 1000 Mk2** device.

- The **OCT model** is validated on **slice 64** (the middle slice).  
- The **CFP model** was trained on images pre-processed using [AutoMorph](https://github.com/rmaphoh/AutoMorph).  
  Please pre-process your CFPs with AutoMorph before running Ret-AAE.  

---

## Downloading checkpoints
Download the model checkpoints from the **Releases** section:  
- `v1_OCT_encoder.pt`  
- `v1_CFP_encoder.pt`  

The checkpoints can be downloaded directly using the commands:
```
# OCT encoder
wget https://github.com/TomJulian/Ret-AAE/releases/download/Checkpoints/v1_OCT_encoder.pt

# CFP encoder
wget https://github.com/TomJulian/Ret-AAE/releases/download/Checkpoints/v1_CFP_encoder.pt
```
---

## Downloading repo
Clone this repository:
```git clone https://github.com/TomJulian/Ret-AAE.git```

## Set configs
Alter the 'config.yaml' file to provide your image directory (containing your CFPs or OCTs, but not a mix of the two) and output directory (where you would like your text file of derived vector embedding values to be saved to). 

## Install requirements
Install the required packages with this code:

```
cd ./Ret-AAE

python3 -m venv retaae

source retaae/bin/activate

pip3 install -r ./requirements.txt
```

## Run model
### For OCTs (slice 64 only):
```python3 OCT_AAE.py```

### For CFPs:
```python3 CFP_AAE.py```

## The outputs
The 256 vector embedding files will be stored in your output directory. Each row will contain (1) the image name (i.e. the original name of each image file), and (2) 256 columns containing each indvidual vector embedding.
