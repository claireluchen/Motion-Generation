import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import PoseDataset, getFrameBoundaries
from torch.utils.data import DataLoader

#Loss function implemented with velocity component in addition to position
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # t is included in input_Size
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

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, output_size, n):
        super(VAE, self).__init__()
        self.n = n
        self.encoder = Encoder(input_size, hidden_size, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_size, output_size)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, t):
        x = torch.cat([t, x], dim=1)  # concatenate t with x here
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z

def new_loss(output, t, X, velocity):
    # Jacobian of position output with respect to t to get velocity
    velocity_hat = torch.autograd.functional.jacobian(lambda x: output, t).squeeze(0)
    # print(output.shape)
    # print(velocity.shape) #torch.Size([32, 3])
    # print("V", velocity_hat.shape) #torch.Size([32, 3, 32, 1])

    # Loss incorporating velocity
    loss = ((X - output) ** 2).sum() + 0.01 * ((velocity - velocity_hat) ** 2).sum() 
    return loss

def train(model, dataloader, optimizer, epoch, scheduler=None):
    model.train()
    train_loss = 0
    for batch_idx, (input_frame, output_frame, input_velocity, idx) in enumerate(dataloader):
        
        optimizer.zero_grad()

        # Extract the time values from input_frame
        t = input_frame[:, :1]  # Extract the time/frame number (first dimension of each frame)
        t.requires_grad_()
        
        # Exclude time values from input_frame and reshape
        x = input_frame[:, model.n:].reshape(input_frame.size(0), -1)

        recon_batch, mu, logvar, z = model(x, t)
        loss = new_loss(recon_batch, t, output_frame[:, 1:], input_velocity)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')
    
    if scheduler:
        scheduler.step()

if __name__ == "__main__":
    csv_file_path = 'D:/Claire/CMUsmplx/CMU/01/merged_poses_with_frames_normalized.csv'

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
    n = 2
    pose_dataset = PoseDataset(csv_file_path, frame_boundaries, n)
    pose_dataloader = DataLoader(pose_dataset, batch_size=32, shuffle=True)

    input_size = 165 * n + 1  # Joint positions and time/frame number
    hidden_size = 256
    latent_dim = 5
    output_size = 165  # Joint positions

    model = VAE(input_size, hidden_size, latent_dim, output_size, n)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    epochs = 50
    for epoch in range(1, epochs + 1):
        train(model, pose_dataloader, optimizer, epoch, scheduler)
