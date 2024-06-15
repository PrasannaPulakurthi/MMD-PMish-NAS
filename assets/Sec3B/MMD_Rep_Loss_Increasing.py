import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

class MMD_loss(nn.Module):
    def __init__(self, bu=4, bl=1/4):
        super(MMD_loss, self).__init__()
        self.fix_sigma = 1
        self.bl = bl
        self.bu = bu

    def phi(self, x, y):
        total0 = x.unsqueeze(0).expand(int(x.size(0)), int(x.size(0)), int(x.size(1)))
        total1 = y.unsqueeze(1).expand(int(y.size(0)), int(y.size(0)), int(y.size(1)))
        return (((total0 - total1) ** 2).sum(2))

    def forward(self, source, target, type):
        M = source.size(dim=0)
        N = target.size(dim=0)
        if M != N:
            target = target[:M, :]  # Truncate target samples to match the number of source samples
        L2_XX = self.phi(source, source)
        L2_YY = self.phi(target, target)
        L2_XY = self.phi(source, target)
        bu = self.bu * torch.ones(L2_XX.size()).type(torch.cuda.FloatTensor)
        bl = self.bl * torch.ones(L2_YY.size()).type(torch.cuda.FloatTensor)
        alpha = (1 / (2 * self.fix_sigma)) * torch.ones(1).type(torch.cuda.FloatTensor)
        m = M * torch.ones(1).type(torch.cuda.FloatTensor)
        if type == "critic":
            XX_u = torch.exp(-alpha * torch.min(L2_XX, bu))
            YY_l = torch.exp(-alpha * torch.max(L2_YY, bl))
            XX = (1 / (m * (m - 1))) * (torch.sum(XX_u) - torch.sum(torch.diagonal(XX_u, 0)))
            YY = torch.mean(YY_l)
            lossD = XX - YY
            return lossD
        elif type == "gen":
            XX_u = torch.exp(-alpha * L2_XX)
            YY_u = torch.exp(-alpha * L2_YY)
            XY_l = torch.exp(-alpha * L2_XY)
            XX = (1 / (m * (m - 1))) * (torch.sum(XX_u) - torch.sum(torch.diagonal(XX_u, 0)))
            YY = (1 / (m * (m - 1))) * (torch.sum(YY_u) - torch.sum(torch.diagonal(YY_u, 0)))
            XY = torch.mean(XY_l)
            lossG = XX + YY - 2 * XY
            return lossG

# Parameters
num_points = 10
num_clusters = 10
initial_radius = 2
final_radius = 8
num_steps = 5001  # Number of optimization steps

# Generate initial cluster centers
cluster_angles = np.linspace(0, 2 * np.pi, num_clusters, endpoint=False)
initial_cluster_centers = np.stack([initial_radius * np.cos(cluster_angles), initial_radius * np.sin(cluster_angles)], axis=-1)

# Generate final cluster centers
final_cluster_centers = np.stack([final_radius * np.cos(cluster_angles), final_radius * np.sin(cluster_angles)], axis=-1)

# Function to interpolate cluster centers
def interpolate_clusters(step, num_steps, initial_centers, final_centers):
    factor = step / num_steps
    return initial_centers * (1 - factor) + final_centers * factor

# Prepare for saving frames
frames_data = []

# Generate initial samples (e.g., random samples)
initial_samples = torch.randn(num_points, 2, requires_grad=True, device='cuda')

# MMD loss instance
mmd_loss = MMD_loss(bu=4, bl=1/4).cuda()

# Optimizer
optimizer = torch.optim.Adam([initial_samples], lr=0.01)

# Define samples_per_cluster
samples_per_cluster = num_points // num_clusters

# Training loop
for step in range(num_steps):
    optimizer.zero_grad()
    
    # Interpolate cluster centers and update target samples
    current_cluster_centers = interpolate_clusters(step, num_steps, initial_cluster_centers, final_cluster_centers)
    target_samples = []
    for cx, cy in current_cluster_centers:
        cluster_samples = np.array([[cx, cy]] * samples_per_cluster)
        target_samples.append(cluster_samples)
    target_samples = np.vstack(target_samples)

    # If there are any leftover samples, add them to the first cluster
    leftover_samples = num_points % num_clusters
    if leftover_samples > 0:
        additional_samples = np.array([[current_cluster_centers[0][0], current_cluster_centers[0][1]]] * leftover_samples)
        target_samples = np.vstack([target_samples, additional_samples])

    target_samples = torch.tensor(target_samples, dtype=torch.float32).cuda()
    
    loss = mmd_loss(initial_samples, target_samples, type='gen')
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f'Step {step}, MMD Loss: {loss.item()}')
        # Save the current state
        frames_data.append((initial_samples.detach().cpu().numpy(), target_samples.cpu().numpy(), loss.item(), step))

# Create a directory for saving frames
output_dir = 'animation_frames_R2_8'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create an animation
fig, ax = plt.subplots(figsize=(6, 6))

def animate(i):
    gen_data, real_data, loss, step = frames_data[i]
    ax.clear()
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.scatter(real_data[:, 0], real_data[:, 1], c='red', alpha=0.5, s=300, label='Real Samples')
    ax.scatter(gen_data[:, 0], gen_data[:, 1], c='blue', alpha=0.5, s=300, label='Generated Samples')
    # ax.legend()
    ax.set_title(f'Step {step}')
    plt.savefig(os.path.join(output_dir, f'frame_{i:04d}.png'))

ani = animation.FuncAnimation(fig, animate, frames=len(frames_data), interval=100)

# Save the animation as a GIF
ani.save('Images/training_animation_R2_8.gif', writer='imagemagick')

# Display the plot
# plt.show()
