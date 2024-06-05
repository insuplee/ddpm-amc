import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim):
        super(ConditionalDiffusionModel, self).__init__()
        self.condition_dim = condition_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Condition embedding layer
        self.condition_embedding = nn.Linear(condition_dim, hidden_dim)

        # Initial transformation layer
        self.initial_transform = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Attention layer
        # Note: Removed batch_first=True, as nn.MultiheadAttention does not accept this argument
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, condition):
        # Embed condition
        condition_embedded = F.relu(self.condition_embedding(condition))

        # Expand condition_embedded to match x's dimensions
        condition_embedded = condition_embedded.unsqueeze(1) # 이 줄은 이전과 동일합니다

        # Adjust x's shape to add an extra dimension
        x = x.unsqueeze(2) # x의 차원을 (batch_size, input_dim, 1)로 조정

        # Concatenate input with condition
        # 이제 x와 condition_embedded 모두 3차원이므로, dim=-1 대신 dim=1을 사용하여 연결합니다.
        x_conditioned = torch.cat((x, condition_embedded), dim=2)

        # Initial transformation
        hidden = F.relu(self.initial_transform(x_conditioned))

        # Adjusting hidden's shape for attention.
        hidden = hidden.transpose(0, 1)

        # Attention mechanism
        hidden, _ = self.attention(hidden, hidden, hidden)

        # After the attention mechanism, convert back from (L, N, E) to (N, L, E)
        hidden = hidden.transpose(0, 1)

        # Generate output
        output = torch.sigmoid(self.output_layer(hidden))
        return output

# Hyperparameters
input_dim = 28*28  # MNIST images are 28x28
condition_dim = 10  # 10 classes in MNIST
hidden_dim = 256

model = ConditionalDiffusionModel(input_dim, condition_dim, hidden_dim)

# Example: Prepare MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)

# Example training loop
for epoch in range(5):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1)  # Flatten the images and ensure it keeps the batch size as its first dimension
        condition = F.one_hot(target, num_classes=condition_dim).float()  # One-hot encode the labels

        # Forward pass
        output = model(data, condition)
        # Here you would typically calculate loss and perform an optimizer step