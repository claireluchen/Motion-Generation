#TAYLOR EXPANSION
import torch
import torch.nn as nn
import torch.optim as optim
from dataloaderTaylor import PoseDataset, getFrameBoundaries
from torch.utils.data import DataLoader
import numpy as np
import time

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4_mean = nn.Linear(hidden_size, latent_dim)
        self.fc4_logvar = nn.Linear(hidden_size, latent_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten from [batch_size, 165, 1] to [batch_size, 165]
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
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        h = self.dropout(self.relu(self.fc3(h)))
        output = self.fc4(h)
        return output.view(output.size(0), 165, 1)  # Reshape back to [batch_size, 165, 1]


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
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z


def getJacobianTerm(model, data_input, pi1):
    batch_size, d, n = data_input.shape

    jacobian = torch.autograd.functional.jacobian(lambda x: model(x)[0], data_input)  # [32, 165, 1, 32, 165, 3]
    # print("jacobian 1", jacobian.shape)
    # print("jacobian 1", jacobian)
    # print("==" * 10)
    jacobian = jacobian.sum(dim=3).squeeze()  # [32, 165, 165, 3]
    # print("jacobian 2", jacobian.shape)
    # print("jacobian 2", jacobian)
    # print("==" * 10)


    # Reshape (pi1 - data_input) to match the required dimensions for matrix multiplication
    difference = (pi1 - data_input).unsqueeze(-1)  # [32, 165, 3, 1]
    # print("difference", difference.shape)
    # print("difference", difference)
    # print("==" * 10)

    # Compute the Taylor expansion term
    result = torch.matmul(jacobian, difference)  # [32, 165, 165, 1]
    # print("RESULT", result)
    # print("==" * 10)

    # Sum over the third dimension (165) to reduce to the desired output shape [32, 165, 1]
    result = result.sum(dim=2)
    # print("result", result.shape)
    # print(result)

    return result


def getTaylor(model, model_output, data_input, pi1):
    #x needs to be f(pi)
    # x_this_frame = model_output#.squeeze().unsqueeze(-1)
    # print("HERE", x_this_frame.shape)
    ans = model_output + getJacobianTerm(model, data_input, pi1)
    # print("TAYLOR", ans.shape)
    return ans

#MODIFY TO TAYLOR
def new_loss(model, model_output, data_input, pi1, data_output, data_output2):
    # print("DATA OUTPUT 2", data_output2.shape)

    taylor = getTaylor(model, model_output, data_input, pi1)
    loss = torch.mean((model_output - data_output) ** 2 + 0.1* (taylor - data_output2) ** 2)
    return loss


def train(model, dataloader, optimizer, n, epoch, scheduler=None):
    model.train()
    train_loss = 0
    
    for batch_idx, (input_frame, output_frame, output_frame_2, output_velocity, idx) in enumerate(dataloader):
        
        optimizer.zero_grad()

        # Transpose input_frame to [batch_size, 165, 3]
        input_frame = input_frame.transpose(1, 2)
        # print(input_frame.shape) #32, 165, 3
        # print("INPUT FRAME", input_frame)
        # print("=" * 20)
        
        # Ensure output_frame has shape [batch_size, 165, 1]
        output_frame = output_frame.view(output_frame.size(0), 165, 1)
        output_frame_2 = output_frame_2.view(output_frame_2.size(0), 165, 1)

        # print(output_frame.shape) #32, 165, 1
        # print("OUTPUT FRAME", output_frame)
        # print("=" * 20)

        # print(output_frame_2.shape)
        # print("OUTPUT FRAME 2", output_frame_2)
        # print("=" * 20)


        # Construct pi1 by concatenating the last two frames from input_frame with output_frame
        pi1 = torch.cat((input_frame[:, :, 1:], output_frame), dim=2)  # pi1 has shape [batch_size, 165, 3]

        # print("Pi+1", pi1.shape) #32, 165, 3
        # print("Pi+1", pi1)
        # Pass input_frame through the model
        model_output, mu, logvar, z = model(input_frame)


        # Compute the loss
        loss = new_loss(model, model_output, input_frame, pi1, output_frame, output_frame_2)
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')


    if scheduler:
        scheduler.step()

if __name__ == "__main__":
    start_time = time.time()
    print("start time", start_time)

    csv_file_path = 'D:/Claire/CMUsmplx/CMU/01/merged_poses.csv'

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
    n = 3
    pose_dataset = PoseDataset(csv_file_path, frame_boundaries, n)
    pose_dataloader = DataLoader(pose_dataset, batch_size=32, shuffle=False)


    input_size = 165 * n
    hidden_size = 256
    latent_dim = 10
    output_size = 165

    model = VAE(input_size, hidden_size, latent_dim, output_size, n)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight)
    #         nn.init.constant_(m.bias, 0)

    # model.apply(init_weights)

    epochs = 50
    for epoch in range(1, epochs + 1):
        train(model, pose_dataloader, optimizer, n, epoch, scheduler)
        print("time passed", time.time() - start_time)

    print("Process took", (time.time() - start_time))

    # torch.save(model.state_dict(), 'vae_model.pth')

    # model.load_state_dict(torch.load('vae_model.pth'))
    # model.eval()

    # input_frame = torch.tensor([[-1.0000,  0.1434, -0.3468,  0.4427, -0.9993,  0.1382, -0.4469,  0.4424]], dtype=torch.float32)
    # t = input_frame[:, :1]
    # t.requires_grad_()
    # input_frame_reshaped = input_frame.view(input_frame.size(0), model.n, -1)
    # x = input_frame_reshaped[:, :, 1:].reshape(input_frame.size(0), -1)

    # with torch.no_grad():
    #     recon_batch, mu, logvar, z = model(x, t)

    # denormalized_output = denormalize_data(recon_batch, position_mean, position_std)
    # print("Reconstructed Output:", denormalized_output)
