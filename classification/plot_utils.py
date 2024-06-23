import argparse
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from models.vmamba import SS2D



def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max

def name_to_multiple_line(name, line_length=50):
    new_name = ""
    for i in range(len(name) // line_length):
        new_name += name[i*line_length:i*line_length + line_length] + '\n'
    new_name += name[(len(name) // line_length) * line_length:]
    print(new_name)
    return new_name

def plot_weight_distribution(model, model_name, bitwidth=32, plot_type='histogram'):
    # bins = (1 << bitwidth) if bitwidth <= 8 else 256
    output_dir = f"{model_name}_weight_distribution"
    os.makedirs(output_dir, exist_ok=True)
    if bitwidth <= 8:
        qmin, qmax = get_quantized_range(bitwidth)
        bins = np.arange(qmin, qmax + 2)
        align = 'left'
    else:
        bins = 256
        align = 'mid'
    plot_index = 0
    for name, param in model.named_parameters():
        if not ('conv2d' in name or 'fc' in name):
            continue
        if param.dim() > 1:
            parmas_to_plot = param.detach().view(-1).cpu()
            if 'histogram' in plot_type:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(parmas_to_plot, bins=bins, density=True,
                        align=align, color = 'blue', alpha = 0.5,
                        edgecolor='black' if bitwidth <= 4 else None)
                if bitwidth <= 4:
                    quantized_min, quantized_max = get_quantized_range(bitwidth)
                    ax.set_xticks(np.arange(start=quantized_min, stop=quantized_max+1))
                ax.set_xlabel(name_to_multiple_line(name))
                ax.set_ylabel('density')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'layer_{name}_histogram.png'))
                plt.close(fig)  # Close the figure to free up memory

            if 'box' in plot_type:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.boxplot(parmas_to_plot, vert=False)
                ax.set_xlabel(name_to_multiple_line(name))
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'layer_{name}_box.png'))
                plt.close(fig)  # Close the figure to free up memory
            
            # Plot channel-wise 3D weight distribution
            if 'per_channel' in plot_type:
                # if 'conv2d' in name:
                #     print("here is conv2d")
                #     torch.save(param, os.path.join(output_dir, f'{name}.pt'))
                fig = plt.figure()
                input_channel, output_channel = param.size()[:2]
                x = np.linspace(0, input_channel, input_channel)
                y = np.linspace(0, output_channel, output_channel)
                x, y = np.meshgrid(y, x)
                parmas_to_plot = param.detach().view(input_channel, output_channel, -1).mean(dim=2).abs().cpu()
                
                ax = fig.add_subplot(projection='3d')
                ax.plot_surface(x, y, parmas_to_plot, cmap='coolwarm')
                ax.set_title(name_to_multiple_line(name))
                ax.set_xlabel('In Channel')
                ax.set_ylabel('Out Channel')
                ax.set_zlabel('Absolute Value')
          
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'layer_{name}_per_channel.png'))
                plt.close(fig)  # Close the figure to free up memory
            

    SS2D_weight_list = ["x_proj_weight", "dt_projs_weight", "dt_projs_bias", "A_logs", "Ds"]
    for module_name, module in model.named_modules():
        if isinstance(module, SS2D):
            for weight_name in SS2D_weight_list:
                param = getattr(module, weight_name)
                if "histogram" in plot_type:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                            align=align, color = 'blue', alpha = 0.5,
                            edgecolor='black' if bitwidth <= 4 else None)
                    if bitwidth <= 4:
                        quantized_min, quantized_max = get_quantized_range(bitwidth)
                        ax.set_xticks(np.arange(start=quantized_min, stop=quantized_max+1))
                    ax.set_xlabel(name_to_multiple_line(f"{module_name}.{weight_name}"))
                    ax.set_ylabel('density')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'layer_{module_name}.{weight_name}_histogram.png'))
                    plt.close(fig)  # Close the figure to free up memory
                
                if "box" in plot_type:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.boxplot(param.detach().view(-1).cpu(), vert=False)
                    ax.set_xlabel(name_to_multiple_line(f"{module_name}.{weight_name}"))
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'layer_{module_name}.{weight_name}_box.png'))
                    plt.close(fig)  # Close the figure to free up memory
                
                if "per_channel" in plot_type:
                    fig = plt.figure()
                    print("name", name, "param size", param.size())
                    if len(param.size()) < 2:
                        continue
                    input_channel, output_channel = param.size()[:2]
                    x = np.linspace(0, input_channel, input_channel)
                    y = np.linspace(0, output_channel, output_channel)
                    x, y = np.meshgrid(y, x)
                    parmas_to_plot = param.detach().view(input_channel, output_channel, -1).mean(dim=2).abs().cpu()
                    
                    ax = fig.add_subplot(projection='3d')
                    ax.plot_surface(x, y, parmas_to_plot, cmap='coolwarm')
                    ax.set_title(name_to_multiple_line(f"{module_name}.{weight_name}"))
                    ax.set_xlabel('In Channel')
                    ax.set_ylabel('Out Channel')
                    ax.set_zlabel('Absolute Value')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'layer_{module_name}.{weight_name}_per_channel.png'))
                    plt.close(fig)  # Close the figure to free up memory
                plot_index += 1
