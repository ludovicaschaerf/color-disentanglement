import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import os
import json
from tqdm import tqdm
import pickle
    
os.makedirs('images', exist_ok=True)

# Specify the path to your JSONL file
file_path = '/home/ludosc/common/textile_images/stable_diffusion_textiles/test/metadata.jsonl'

# List to store the extracted 'additional_feature' values
additional_features = []

# Open and read the file line by line
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)  # Parse the JSON data from each line
            additional_features.append(data['additional_feature'])  # Append the value to the list

    print("Additional features extracted successfully:")
except FileNotFoundError:
    print(f"Error: The file at {file_path} does not exist.")
except json.JSONDecodeError:
    print("Error: There was an issue decoding the JSON data.")
except KeyError:
    print("Error: The expected key 'additional_feature' is missing in some JSON entries.")
    
print(additional_features[:10])
    
model_path = "/home/ludosc/ludosc/repos/diffusers/examples/text_to_image/sd-textile"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-15000/unet", torch_dtype=torch.float16)

print(unet)

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", unet=unet, torch_dtype=torch.float16)
pipe.to("cuda")

def save_input_bottleneck_latents(module, input, output):
    # Save the output tensor of the bottleneck layer
    global input_bottleneck_latent
    input_bottleneck_latent = output[0].detach().cpu()
    
def save_bottleneck_latents(module, input, output):
    # Save the output tensor of the bottleneck layer
    global bottleneck_latents
    bottleneck_latents = output.detach().cpu()

input_bottleneck_latents = None

# Define a directory to save images and their latent spaces
save_dir = 'latents'
os.makedirs(save_dir, exist_ok=True)
hook = pipe.unet.down_blocks[0].register_forward_hook(save_input_bottleneck_latents)
hook_2 = pipe.unet.mid_block.register_forward_hook(save_bottleneck_latents)
# Dictionary to hold latent space and filename
latent_dict = {'unet_space':[], 'unet_orig_space': [], 'fname':[]}

for additional_feature in tqdm(additional_features):
    image = pipe(prompt=additional_feature).images[0]
    image_filename = 'images/' + additional_feature.replace(' ', '_').replace('/', '_')[:1000] + ".png"
    image.save(image_filename)
    

    # Add data to the dictionary
    latent_dict['unet_space'].append(bottleneck_latents.numpy())
    latent_dict['unet_orig_space'].append(input_bottleneck_latent.numpy())
    latent_dict['fname'].append(image_filename)

    # Remove the hook after use to prevent memory leaks
    hook.remove()
    hook_2.remove()

with open(os.path.join(save_dir, 'latent_data.pkl'), 'wb') as f:
    pickle.dump(latent_dict, f)

print("Latent space and image filename saved successfully.")