import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import PoseDataset, getFrameBoundaries
from torch.utils.data import DataLoader

# Encoder Model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4_mean = nn.Linear(hidden_size, latent_dim)
        self.fc4_logvar = nn.Linear(hidden_size, latent_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        h = self.dropout(self.relu(self.fc3(h)))
        return self.fc4_mean(h), self.fc4_logvar(h)

# Decoder Model
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        h = self.dropout(self.relu(self.fc3(h)))
        return self.fc4(h)

# VAE Model combining Encoder and Decoder
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, output_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_size, output_size)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Loss function with adjustable weight for KLD
def loss_function(epoch, recon_x, x, mu, logvar, beta=1.0):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if epoch>10:
        return MSE + KLD
    else:
        return MSE

def train(model, dataloader, optimizer, epoch, scheduler=None):
    model.train()
    train_loss = 0
    for batch_idx, (input_frame, output_frame, idx) in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(input_frame)
        loss = loss_function(epoch,recon_batch, output_frame, mu, logvar, beta=0.1)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')
    
    if scheduler:
        scheduler.step()

# Main training script
if __name__ == "__main__":
    # Define the path to the CSV file
    csv_file_path = 'D:/Claire/CMUsmplx/CMU/01/merged_poses_with_frames_normalized.csv'

    # Frame boundaries for each file, inclusive
    npz_files = [
        'D:/Claire/CMUsmplx/CMU/01/01_01_stageii.npz',
        'D:/Claire/CMUsmplx/CMU/01/01_02_stageii.npz',
        'D:/Claire/CMUsmplx/CMU/01/01_03_stageii.npz',
        'D:/Claire/CMUsmplx/CMU/01/01_05_stageii.npz',
        'D:/Claire/CMUsmplx/CMU/01/01_06_stageii.npz',
        'D:/Claire/CMUsmplx/CMU/01/01_07_stageii.npz',
        'D:/Claire/CMUsmplx/CMU/01/01_08_stageii.npz',
        'D:/Claire/CMUsmplx/CMU/01/01_09_stageii.npz',
        'D:/Claire/CMUsmplx/CMU/01/01_10_stageii.npz',
        'D:/Claire/CMUsmplx/CMU/01/01_11_stageii.npz',
    ]
    frame_boundaries = getFrameBoundaries(npz_files)

    # Create the dataset and dataloader
    n_frames = 2  # Set the number of frames to concatenate as input
    pose_dataset = PoseDataset(csv_file_path, frame_boundaries, n_frames)
    pose_dataloader = DataLoader(pose_dataset, batch_size=64, shuffle=True)


    print(pose_dataset.getDim()[1])
    # Define the model, optimizer, and other training parameters
    frame_dim = pose_dataset.getDim()[1]  # Each frame has 165 or 166 dimensions
    input_dim = frame_dim * n_frames  # Concatenated n frames
    hidden_dim = 512  # hidden layer size
    latent_dim = 10  # latent space dimension
    output_dim = frame_dim  # Single frame output

    model = VAE(input_dim, hidden_dim, latent_dim, output_dim)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Adjust learning rate over time

    # Train the VAE
    num_epochs = 50  
    for epoch in range(1, num_epochs + 1):
        train(model, pose_dataloader, optimizer, epoch, scheduler)
