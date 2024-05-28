import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel
import os
import json
from tqdm import tqdm
import pickle
from torchsummary import summary
from PIL import Image

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
    
# print(additional_features[:10])
    
model_path = "/home/ludosc/ludosc/repos/diffusers/examples/text_to_image/sd-textile"
# Load the modified UNet model
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-15000/unet", torch_dtype=torch.float16)

# print(unet)

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", unet=unet, torch_dtype=torch.float16)
pipe.to("cuda")

# # Initialize DDIM scheduler
scheduler = DDIMScheduler.from_config('CompVis/stable-diffusion-v1-4', subfolder="scheduler")
pipe.scheduler = scheduler
pipe.scheduler.set_timesteps(100, device='cuda')

# Hook to capture the mid_block output
mid_block_output = None

def hook(module, input, output):
    global mid_block_output
    mid_block_output = output[0].detach().cpu()

pipe.unet.mid_block.register_forward_hook(hook)

for name, submodule in pipe.unet.named_children():
    print(name, '---------------')
    for subname, subsubmodule in submodule.named_children():
        print(name, subname)

def store_latent_spaces(model, prompt, produce_final_image=False):
    h_spaces = []
    timesteps = []
    
    text_inputs = model.tokenizer(prompt, return_tensors="pt").to('cuda')
    text_embeddings = model.text_encoder(text_inputs.input_ids).last_hidden_state.to(torch.float16)
    #generator = torch.manual_seed(0, device='cuda')    # Seed generator to create the inital latent noise

    # Sample latents using DDIM
    latents = torch.randn((1, model.unet.in_channels, 64, 64), device='cuda', dtype=torch.float16)  # Example dimensions
    latents = latents * scheduler.init_noise_sigma

    for t in model.scheduler.timesteps:
        # t = t.to('cuda').type(torch.float16)
        latent_model_input = model.scheduler.scale_model_input(latents, t)
        
        with torch.no_grad():
            noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
            noise_pred = noise_pred.sample
        # noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        latents = model.scheduler.step(noise_pred, t, latents).prev_sample

        # h_space = model.unet.mid_block(latent_model_input)  # Extract h-space
        h_space = mid_block_output  # Extract h-space from the hook output
        h_spaces.append(h_space.cpu())
        timesteps.append(t.cpu())
    
    final_image = None
    if produce_final_image:
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        images = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        final_image = [Image.fromarray(image) for image in images]
        print(len(final_image))


    return h_spaces, timesteps, final_image

# Define a directory to save images and their latent spaces
save_dir = '../data'
os.makedirs(save_dir, exist_ok=True)

latent_dict = {'h_space':[], 'timestep': [], 'fname':[], 'prompt':[]}

for i, additional_feature in tqdm(enumerate(additional_features)):
    h_spaces, timesteps, final_image = store_latent_spaces(pipe, additional_feature, True)
    image_filename = 'images/' + additional_feature.replace(' ', '_').replace('/', '_')[:1000] + ".png"
    final_image[0].save(image_filename)
    
    print(len(h_spaces), h_spaces[-1].shape)
    # Add data to the dictionary
    latent_dict['h_space'].append(h_spaces)
    latent_dict['timestep'].append(timesteps)
    latent_dict['fname'].append(image_filename)
    latent_dict['prompt'].append(additional_feature)
    

    if i % 50 == 0:
        with open(os.path.join(save_dir, f'seeds_asyrp_0000_{len(additional_features)}.pkl'), 'wb') as f:
            pickle.dump(latent_dict, f)

with open(os.path.join(save_dir, f'seeds_asyrp_0000_{len(additional_features)}.pkl'), 'wb') as f:
    pickle.dump(latent_dict, f)

print("Latent space and image filename saved successfully.")