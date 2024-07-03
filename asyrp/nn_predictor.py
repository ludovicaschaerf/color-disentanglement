
# Transformer-based HSpacePredictor
class HSpacePredictor(nn.Module):
    def __init__(self, h_dim, t_dim, hidden_dim):
        super(HSpacePredictor, self).__init__()
        self.h_dim = h_dim
        self.t_dim = t_dim
        self.hidden_dim = hidden_dim
        
        self.transformer = nn.Transformer(d_model=h_dim, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
        self.fc = nn.Sequential(
            nn.Linear(h_dim + t_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, h_dim)
        )
    
    def forward(self, h, t):
        # Transformer expects input of shape (sequence length, batch size, embedding dimension)
        h = h.permute(1, 0, 2)  # Permute to match expected input shape for transformer
        h_transformed = self.transformer(h, h)  # Apply transformer
        
        # Combine h-space and timestep embedding
        t = t.unsqueeze(1).repeat(1, h.size(1), 1)  # Repeat timestep embedding
        combined = torch.cat((h_transformed.permute(1, 0, 2), t), dim=-1)  # Combine along the last dimension
        
        return self.fc(combined)
