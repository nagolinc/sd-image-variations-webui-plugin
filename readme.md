

Trying to get image variations working in stable-diffsuion-webui

https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/5505


to use:

in requirements_verions.txt,

replace diffusers and transformers with the latest github version

git+https://github.com/huggingface/diffusers.git
git+https://github.com/huggingface/transformers.git

copy sd_image_vartiations.py to your {webui}/scripts folder

choose input image in img2img tab and select "sd-image-variations" from the dropdown

