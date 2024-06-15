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
radius = 8  # Adjust this as needed for your problem
num_steps = 5001  # Number of optimization steps

# Generate cluster centers
cluster_angles = np.linspace(0, 2 * np.pi, num_clusters, endpoint=False)
cluster_centers = np.stack([radius * np.cos(cluster_angles), radius * np.sin(cluster_angles)], axis=-1)

# Generate target samples directly at cluster centers without randomness
target_samples = []
samples_per_cluster = num_points // num_clusters
for cx, cy in cluster_centers:
    cluster_samples = np.array([[cx, cy]] * samples_per_cluster)
    target_samples.append(cluster_samples)
target_samples = np.vstack(target_samples)

# If there are any leftover samples, add them to the first cluster
leftover_samples = num_points % num_clusters
if leftover_samples > 0:
    additional_samples = np.array([[cluster_centers[0][0], cluster_centers[0][1]]] * leftover_samples)
    target_samples = np.vstack([target_samples, additional_samples])

# Convert target samples to PyTorch tensors
target_samples = torch.tensor(target_samples, dtype=torch.float32).cuda()

# Generate initial samples (e.g., random samples)
initial_samples = torch.randn(num_points, 2, requires_grad=True, device='cuda')

# MMD loss instance
mmd_loss = MMD_loss(bu=4, bl=1/4).cuda()

# Optimizer
optimizer = torch.optim.Adam([initial_samples], lr=0.02)

# Prepare for saving frames
frames_data = []

# Training loop
for step in range(num_steps):
    optimizer.zero_grad()
    loss = mmd_loss(initial_samples, target_samples, type='gen')
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f'Step {step}, MMD Loss: {loss.item()}')
        # Save the current state
        frames_data.append((initial_samples.detach().cpu().numpy(), loss.item(), step))


# Create a directory for saving frames
output_dir = 'animation_frames_R'+str(radius)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create an animation
fig, ax = plt.subplots(figsize=(6, 6))

def animate(i):
    data, loss, step = frames_data[i]
    ax.clear()
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.scatter(target_samples.cpu().numpy()[:, 0], target_samples.cpu().numpy()[:, 1], c='red', alpha=0.5, s=300, label='Real Samples')
    ax.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.5, s=300, label='Generated Samples')
    # ax.legend(fontsize="15", loc ="upper left") 
    ax.set_title(f'Step {step}')
    plt.savefig(os.path.join(output_dir, f'frame_{i:04d}.png'))

ani = animation.FuncAnimation(fig, animate, frames=len(frames_data), interval=100)

# Save the animation as a GIF
ani.save('Images/training_animation_R'+str(radius)+'.gif', writer='imagemagick')

# Display the plot
# plt.show()
