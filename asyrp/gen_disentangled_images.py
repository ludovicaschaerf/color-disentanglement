

import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel

with open('/home/ludosc/ludosc/color-disentanglement/data/seeds_asyrp_0000_4319_c.pkl', 'rb') as infile:
    pkl_file = pickle.load(infile)
    
X = np.array(pkl_file['h_space']).reshape((len(pkl_file['h_space'])*100,1280*8*8))
y = np.array([color for color in pkl_file['color'] for i in range(100)])
print(y.shape, X.shape)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(random_state=0, C=0.001)
clf.fit(x_train, y_train)
print(clf.classes_)
performance = np.round(clf.score(x_val, y_val), 2)
print(performance)

disent_dir = clf.coef_ / np.linalg.norm(clf.coef_)[0]
disent_dir = disent_dir.reshape((1, 1280, 8, 8))

print(disent_dir)

# Custom DDIMScheduler that inherits from DDIMScheduler
class CustomDDIMScheduler(DDIMScheduler):
    def __init__(self, alpha, sigma, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.sigma = sigma

    def step(self, unet, h_varied, h_original, t, latents):
        alpha_t_1 = torch.sqrt(self.alpha[t-1]).to(latents.device)
        noise = self.sigma[t] * torch.randn_like(latents).to(latents.device)

        # Compute the epsilon values using the UNet model
        epsilon_h_varied = unet(h_varied, t).sample
        epsilon_h_original = unet(h_original, t).sample

        # Apply the formula
        x_t_1 = alpha_t_1 * epsilon_h_varied + epsilon_h_original + noise
        return x_t_1

# Function to generate varied images using the custom scheduler
def generate_varied_image(model, prompt, produce_final_image=False):
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

    for i, (h, t) in enumerate(zip(h_spaces.repeat(timesteps), timesteps)):
        t_encoding = torch.tensor([t], device=pipe.device).float().unsqueeze(0)
        h_varied = h.unsqueeze(0) + torch.tensor(disent_dir).float().to('cuda')
        
        # Use the custom scheduler to step through the reverse diffusion process
        latents = h_varied
        for t_back in reversed(timesteps[:i+1]):
            latents = scheduler.step(pipe.unet, h_varied, h, t_back, latents)
        
        image = pipe.decode(latents)
        varied_images.append(image)
    
    return varied_images

# Example usage
prompt = pkl_file['prompt'][0]

timesteps = 100
# Example values for alpha and sigma
alpha = torch.linspace(0.9, 0.1, steps=len(timesteps))
sigma = torch.linspace(0.1, 0.9, steps=len(timesteps))
scheduler = CustomDDIMScheduler(alpha=alpha, sigma=sigma,).from_config('CompVis/stable-diffusion-v1-4', subfolder="scheduler")
scheduler.set_timesteps(100, device='cuda')

varied_images = generate_varied_image(pipe, prompt)
varied_images[0].save("varied_output_image.png")
