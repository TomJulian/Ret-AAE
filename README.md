# Ret-AAE
This is the public repo for the Ret-AAE models (adversarial autoencoders for optical coherence tomography and colour fundus photographs). The encoder is shared, but the decoder cannot be shared, given UK Biobanks policies on generative AI trained using their data. 

The pre-print paper related to this repo is available at: https://www.medrxiv.org/content/10.1101/2025.08.04.25332962v1.full-text

# Before running this model...
These models were trained using UK Biobank data. Therefore, it has been validated on images captured using the TOPCON 3D OCT 1000 Mk2 device,

The OCT model is validated on slice 64 (the middle slice) OCT. 

The CFP model was trained on images pre-processed using Automorph (i.e. the cropped images). Please pre-process your CFPs using Automorph prior to running Ret-AAE (https://github.com/rmaphoh/AutoMorph)

# Downloading checkpoints 
Please begin by downloading the model checkpoints, which are available in the 'releases' section as 'OCT_encoder.pt' and 'CFP_encoder.pt'

# Downloading repo
Download the repo with command:
\git clone https://github.com/TomJulian/Ret-AAE.git\

# Set configs
Alter the 'config.yaml' file to provide your image directory (containing your CFPs or OCTs, but not a mix of the two) and output directory (where you would like your text file of derived vector embedding values to be saved to). 

# Install requirements
Install the required packages with this code:

\python3 -m venve retaae

source retaae/bin/activate

pip3 install -r ./requirements.txt\

# Run model
For OCTs (slice 64 only, Topcon e.g. the UK Biobank data:
\python3 OCT_AAE.py\

For CFPs:
\python3 CFP_AAE.py\

# The outputs
The 256 vector embedding files will be stored in your output directory. Each row will contain (1) the image name (i.e. the original name of each image file), and (2) 256 columns containing each indvidual vector embedding
