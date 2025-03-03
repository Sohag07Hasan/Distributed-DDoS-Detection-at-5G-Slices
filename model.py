import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_FEATURES, NUM_CLASSES



class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], latent_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print(f"Debug: Input Shape to AutoEncoder: {x.shape}")
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



#Features 4~41
class Net(nn.Module):
    def __init__(self, input_size: int = NUM_FEATURES, num_classes: int = NUM_CLASSES) -> None:
        super(Net, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3)  # Kernel size 3
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Calculate output size after first conv and pooling
        conv1_output_size = input_size - 3 + 1  # After first convolution layer
        pooled1_size = conv1_output_size // 2   # After first pooling layer

        # Apply the second convolution only if the size is large enough
        if pooled1_size > 2:
            self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3)  # Kernel size 3
            conv2_output_size = pooled1_size - 3 + 1
            if conv2_output_size > 2:
                self.use_second_pool = True
                pooled2_size = conv2_output_size // 2
            else:
                self.use_second_pool = False
                pooled2_size = conv2_output_size
        else:
            self.conv2 = None
            pooled2_size = pooled1_size

        # Dynamically calculate the input size for the fully connected layers
        fc_input_size = 16 * pooled2_size if self.conv2 else 6 * pooled1_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_size)
        x = self.pool(F.relu(self.conv1(x)))  # First conv + pool

        if self.conv2 is not None:
            x = F.relu(self.conv2(x))  # Second conv if it exists
            if self.use_second_pool:
                x = self.pool(x)  # Pool only if the size allows

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten for FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


