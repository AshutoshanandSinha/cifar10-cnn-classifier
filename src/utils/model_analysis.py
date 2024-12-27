import torch
from torchsummary import summary
from ..utils.config import config
import torch.nn as nn

def calculate_rf(model):
    """Calculate and print receptive field progression through the network"""
    rf = 1
    stride_accumulated = 1
    
    print("\nReceptive Field Analysis:")
    print("-" * 50)
    print(f"{'Layer':<20} {'RF':<10} {'Stride':<10} {'Image Size':<15}")
    print("-" * 50)
    
    # Input
    print(f"{'Input':<20} {rf:<10} {stride_accumulated:<10} {'32x32':<15}")
    
    # Track image size
    image_size = 32
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Calculate new RF
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]
            dilation = module.dilation[0]
            padding = module.padding[0]
            
            # Update accumulated stride
            stride_accumulated *= stride
            
            # Calculate RF increment
            rf_increment = (kernel_size - 1) * dilation * stride_accumulated
            rf += rf_increment
            
            # Calculate new image size
            image_size = ((image_size + 2*padding - dilation*(kernel_size-1) - 1) // stride) + 1
            
            print(f"{name:<20} {rf:<10} {stride_accumulated:<10} {f'{image_size}x{image_size}':<15}")
    
    print("-" * 50)
    return rf

def analyze_model(model, input_size=(3, 32, 32)):
    """
    Prints detailed analysis of the model including:
    - Model summary
    - Parameter counts by layer
    - Total parameters
    - Model size in MB
    """
    # Move model to the correct device
    model = model.to(config.DEVICE)
    
    print("\nModel Architecture Analysis")
    print("=" * 50)
    
    # Print model summary
    print("\nModel Summary:")
    print("-" * 50)
    summary(model, input_size=input_size, device=config.DEVICE)
    
    # Add RF analysis
    final_rf = calculate_rf(model)
    
    # Print parameter counts by layer
    print("\nParameter Count by Layer:")
    print("-" * 50)
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        print(f"{name}: {param_count:,} parameters")
    
    # Calculate model size in MB
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    print("\nModel Statistics:")
    print("-" * 50)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size:           {model_size:.2f} MB")
    print("=" * 50 + "\n")
    
    return total_params, trainable_params, model_size 