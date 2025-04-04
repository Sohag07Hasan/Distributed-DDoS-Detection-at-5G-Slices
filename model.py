import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_FEATURES, NUM_CLASSES


##Input is 30 not in use
# class Net(nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)  # 1D Convolution
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)  # 1D Convolution
#         self.fc1 = nn.Linear(16 * 4, 120)  # Adjust the input size based on the output from conv layers
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, 30)
#         x = self.pool(F.relu(self.conv1(x)))  # Output shape: (batch_size, 6, 26)
#         x = self.pool(F.relu(self.conv2(x)))  # Output shape: (batch_size, 16, 9)
#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers: (batch_size, 16 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    

#input is 40, 41 not in use

# class Net(nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)  # 1D Convolution
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)  # 1D Convolution
        
#         # Recalculate the input size for the fully connected layer based on new input size
#         # New input is 40, conv1 reduces it to (40 - 5 + 1) = 36, pooling reduces it to 18
#         # conv2 reduces it to (18 - 5 + 1) = 14, pooling reduces it to 7
#         self.fc1 = nn.Linear(16 * 7, 120)  # Adjust the input size based on the output from conv layers
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, 40)
#         x = self.pool(F.relu(self.conv1(x)))  # Output shape: (batch_size, 6, 36), after pooling: (batch_size, 6, 18)
#         x = self.pool(F.relu(self.conv2(x)))  # Output shape: (batch_size, 16, 14), after pooling: (batch_size, 16, 7)
#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers: (batch_size, 16 * 7)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# import torch
# import torch.nn as nn


#input size is 35 not in use
# class Net(nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)  # 1D Convolution
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)  # 1D Convolution
        
#         # Recalculate the input size for the fully connected layer based on new input size (35)
#         # Input: 35, conv1 reduces to (35 - 5 + 1) = 31, pooling reduces to 15
#         # conv2 reduces to (15 - 5 + 1) = 11, pooling reduces to 5
#         self.fc1 = nn.Linear(16 * 5, 120)  # Adjust the input size based on the output from conv layers
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, 35)
#         x = self.pool(F.relu(self.conv1(x)))  # Output shape: (batch_size, 6, 31), after pooling: (batch_size, 6, 15)
#         x = self.pool(F.relu(self.conv2(x)))  # Output shape: (batch_size, 16, 11), after pooling: (batch_size, 16, 5)
#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers: (batch_size, 16 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


#Dynamic Input size (big feature numbers like 30) not in use
# class Net(nn.Module):
#     def __init__(self, input_size: int = NUM_FEATURES, num_classes: int = NUM_CLASSES) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)  # 1D Convolution
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)  # 1D Convolution
        
#         # Dynamically calculate the size after convolutions and pooling
#         # conv1 reduces input to (input_size - 5 + 1)
#         # pool1 reduces to half
#         # conv2 further reduces the size
#         conv1_output_size = input_size - 5 + 1  # After first conv layer
#         pooled1_size = conv1_output_size // 2   # After first pooling layer
#         conv2_output_size = pooled1_size - 5 + 1  # After second conv layer
#         pooled2_size = conv2_output_size // 2    # After second pooling layer
        
#         # Fully connected layer based on dynamic input size
#         self.fc1 = nn.Linear(16 * pooled2_size, 120)  # Dynamically set the size
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)  # Output size remains fixed to 2

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, input_size)
#         x = self.pool(F.relu(self.conv1(x)))  # First conv + pool
#         x = self.pool(F.relu(self.conv2(x)))  # Second conv + pool
#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # 26~30,41 features (rest are unused)
# # 29 features (initial desigin)
# class Net(nn.Module):
#     def __init__(self, input_size: int = NUM_FEATURES, num_classes: int = NUM_CLASSES) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)  # 1D Convolution
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)  # 1D Convolution
        
#         # Dynamically calculate the size after convolutions and pooling
#         # For input_size = 29:
#         # conv1 reduces input to (input_size - kernel_size + 1)
#         # pool1 reduces to half
#         # conv2 further reduces the size
#         conv1_output_size = input_size - 5 + 1  # After first conv layer
#         pooled1_size = conv1_output_size // 2   # After first pooling layer
#         conv2_output_size = pooled1_size - 5 + 1  # After second conv layer
#         pooled2_size = conv2_output_size // 2    # After second pooling layer
        
#         # Fully connected layer based on dynamic input size
#         self.fc1 = nn.Linear(16 * pooled2_size, 120)  # Dynamically set the size
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)  # Output size remains fixed

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, input_size)
#         x = self.pool(F.relu(self.conv1(x)))  # First conv + pool
#         x = self.pool(F.relu(self.conv2(x)))  # Second conv + pool
#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


## 28 Features
# class Net(nn.Module):
#     def __init__(self, input_size: int = NUM_FEATURES, num_classes: int = NUM_CLASSES) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)  # 1D Convolution
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)  # 1D Convolution
        
#         # Dynamically calculate the size after convolutions and pooling
#         # For input size 28:
#         # conv1 reduces input to (28 - 5 + 1) = 24
#         # pool1 reduces to half: 24 // 2 = 12
#         # conv2 further reduces to (12 - 5 + 1) = 8
#         # pool2 reduces to half: 8 // 2 = 4
#         conv1_output_size = input_size - 5 + 1  # After first conv layer
#         pooled1_size = conv1_output_size // 2   # After first pooling layer
#         conv2_output_size = pooled1_size - 5 + 1  # After second conv layer
#         pooled2_size = conv2_output_size // 2    # After second pooling layer
        
#         # Fully connected layer based on dynamic input size
#         self.fc1 = nn.Linear(16 * pooled2_size, 120)  # Adjusted size for 28 input features
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)  # Output size remains fixed for binary classification

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, input_size)
#         x = self.pool(F.relu(self.conv1(x)))  # First conv + pool
#         x = self.pool(F.relu(self.conv2(x)))  # Second conv + pool
#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x



# ## 26, 27
# class Net(nn.Module):
#     def __init__(self, input_size: int = 27, num_classes: int = 2) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)  # 1D Convolution
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)  # 1D Convolution
        
#         # Dynamically calculate the size after convolutions and pooling
#         # For input size 27:
#         # conv1 reduces input to (27 - 5 + 1) = 23
#         # pool1 reduces to half: 23 // 2 = 11
#         # conv2 further reduces to (11 - 5 + 1) = 7
#         # pool2 reduces to half: 7 // 2 = 3
#         conv1_output_size = input_size - 5 + 1  # After first conv layer
#         pooled1_size = conv1_output_size // 2   # After first pooling layer
#         conv2_output_size = pooled1_size - 5 + 1  # After second conv layer
#         pooled2_size = conv2_output_size // 2    # After second pooling layer
        
#         # Fully connected layer based on dynamic input size
#         self.fc1 = nn.Linear(16 * pooled2_size, 120)  # Size remains 16 * 3 for 27 features
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)  # Output size remains fixed for binary classification

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, input_size)
#         x = self.pool(F.relu(self.conv1(x)))  # First conv + pool
#         x = self.pool(F.relu(self.conv2(x)))  # Second conv + pool
#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# ## For Small features like 8
# class Net(nn.Module):
#     def __init__(self, input_size: int = 8, num_classes: int = 2) -> None:
#         super(Net, self).__init__()
#         # Adjust the kernel sizes and layers for smaller inputs
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3)  # Reduced kernel size to 3
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # Dynamically calculate the size after convolutions and pooling
#         conv1_output_size = input_size - 3 + 1  # After first conv layer
#         pooled1_size = conv1_output_size // 2   # After first pooling layer
        
#         # If the pooled size is very small, we avoid further convolutions
#         if pooled1_size > 2:
#             self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3)  # Reduced kernel size to 3
#             conv2_output_size = pooled1_size - 3 + 1
#             if conv2_output_size > 2:
#                 self.use_second_pool = True
#                 pooled2_size = conv2_output_size // 2
#             else:
#                 self.use_second_pool = False
#                 pooled2_size = conv2_output_size
#         else:
#             self.conv2 = None
#             pooled2_size = pooled1_size

#         # Fully connected layer based on dynamic input size
#         self.fc1 = nn.Linear(16 * pooled2_size if self.conv2 else 6 * pooled1_size, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, input_size)
#         x = self.pool(F.relu(self.conv1(x)))  # First conv + pool

#         if self.conv2 is not None:
#             x = F.relu(self.conv2(x))  # Second conv (if exists)
#             if self.use_second_pool:
#                 x = self.pool(x)  # Only pool if the size allows it

#         x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x



# #For 11 features
# class Net(nn.Module):
#     def __init__(self, input_size: int = 11, num_classes: int = 2) -> None:
#         super(Net, self).__init__()
#         # First convolution layer
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3)  # Kernel size 3
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # Calculate output size after first conv and pooling
#         conv1_output_size = input_size - 3 + 1  # After first convolution layer
#         pooled1_size = conv1_output_size // 2   # After first pooling layer
        
#         # If the size is large enough, we apply the second convolution and pooling
#         if pooled1_size > 2:
#             self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3)  # Kernel size 3
#             conv2_output_size = pooled1_size - 3 + 1
#             if conv2_output_size > 2:
#                 self.use_second_pool = True
#                 pooled2_size = conv2_output_size // 2
#             else:
#                 self.use_second_pool = False
#                 pooled2_size = conv2_output_size
#         else:
#             self.conv2 = None
#             pooled2_size = pooled1_size

#         # Fully connected layer based on dynamic input size
#         self.fc1 = nn.Linear(16 * pooled2_size if self.conv2 else 6 * pooled1_size, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_size)
#         x = self.pool(F.relu(self.conv1(x)))  # First conv + pool

#         if self.conv2 is not None:
#             x = F.relu(self.conv2(x))  # Second conv if it exists
#             if self.use_second_pool:
#                 x = self.pool(x)  # Pool only if size allows

#         # Flatten the tensor for the fully connected layers
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


#for 25 features

# class Net(nn.Module):
#     def __init__(self, input_size: int = 25, num_classes: int = 2) -> None:
#         super(Net, self).__init__()
        
#         # First convolution layer
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3)  # Kernel size 3
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # Calculate output size after first conv and pooling
#         conv1_output_size = input_size - 3 + 1  # After first convolution layer
#         pooled1_size = conv1_output_size // 2   # After first pooling layer

#         # Apply the second convolution only if the size is large enough
#         if pooled1_size > 2:
#             self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3)  # Kernel size 3
#             conv2_output_size = pooled1_size - 3 + 1
#             if conv2_output_size > 2:
#                 self.use_second_pool = True
#                 pooled2_size = conv2_output_size // 2
#             else:
#                 self.use_second_pool = False
#                 pooled2_size = conv2_output_size
#         else:
#             self.conv2 = None
#             pooled2_size = pooled1_size

#         # Dynamically calculate the input size for the fully connected layers
#         fc_input_size = 16 * pooled2_size if self.conv2 else 6 * pooled1_size
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(fc_input_size, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_size)
#         x = self.pool(F.relu(self.conv1(x)))  # First conv + pool

#         if self.conv2 is not None:
#             x = F.relu(self.conv2(x))  # Second conv if it exists
#             if self.use_second_pool:
#                 x = self.pool(x)  # Pool only if the size allows

#         # Flatten the tensor for the fully connected layers
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


## 17 features

# class Net(nn.Module):
#     def __init__(self, input_size: int = 17, num_classes: int = 2) -> None:
#         super(Net, self).__init__()
        
#         # First convolution layer
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3)  # Kernel size 3
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # Calculate output size after first conv and pooling
#         conv1_output_size = input_size - 3 + 1  # After first convolution layer
#         pooled1_size = conv1_output_size // 2   # After first pooling layer

#         # Apply the second convolution only if the size is large enough
#         if pooled1_size > 2:
#             self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3)  # Kernel size 3
#             conv2_output_size = pooled1_size - 3 + 1
#             if conv2_output_size > 2:
#                 self.use_second_pool = True
#                 pooled2_size = conv2_output_size // 2
#             else:
#                 self.use_second_pool = False
#                 pooled2_size = conv2_output_size
#         else:
#             self.conv2 = None
#             pooled2_size = pooled1_size

#         # Dynamically calculate the input size for the fully connected layers
#         fc_input_size = 16 * pooled2_size if self.conv2 else 6 * pooled1_size
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(fc_input_size, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_size)
#         x = self.pool(F.relu(self.conv1(x)))  # First conv + pool

#         if self.conv2 is not None:
#             x = F.relu(self.conv2(x))  # Second conv if it exists
#             if self.use_second_pool:
#                 x = self.pool(x)  # Pool only if the size allows

#         # Flatten the tensor for the fully connected layers
#         x = x.view(x.size(0), -1)  # Flatten for FC layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# # Features like 16, 15, 14, 12, 13
# class Net(nn.Module):
#     def __init__(self, input_size: int = NUM_FEATURES, num_classes: int = NUM_CLASSES) -> None:
#         super(Net, self).__init__()
        
#         # First convolution layer
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3)  # Kernel size 3
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # Calculate output size after first conv and pooling
#         conv1_output_size = input_size - 3 + 1  # After first convolution layer
#         pooled1_size = conv1_output_size // 2   # After first pooling layer

#         # Apply the second convolution only if the size is large enough
#         if pooled1_size > 2:
#             self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3)  # Kernel size 3
#             conv2_output_size = pooled1_size - 3 + 1
#             if conv2_output_size > 2:
#                 self.use_second_pool = True
#                 pooled2_size = conv2_output_size // 2
#             else:
#                 self.use_second_pool = False
#                 pooled2_size = conv2_output_size
#         else:
#             self.conv2 = None
#             pooled2_size = pooled1_size

#         # Dynamically calculate the input size for the fully connected layers
#         fc_input_size = 16 * pooled2_size if self.conv2 else 6 * pooled1_size
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(fc_input_size, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_size)
#         x = self.pool(F.relu(self.conv1(x)))  # First conv + pool

#         if self.conv2 is not None:
#             x = F.relu(self.conv2(x))  # Second conv if it exists
#             if self.use_second_pool:
#                 x = self.pool(x)  # Pool only if the size allows

#         # Flatten the tensor for the fully connected layers
#         x = x.view(x.size(0), -1)  # Flatten for FC layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


#Features 18~24
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