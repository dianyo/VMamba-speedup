import argparse
import torch
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

def plot_weight_distribution(model, model_name, bitwidth=32, plot_type='histogram',  per_channel=False):
    # bins = (1 << bitwidth) if bitwidth <= 8 else 256
    if bitwidth <= 8:
        qmin, qmax = get_quantized_range(bitwidth)
        bins = np.arange(qmin, qmax + 2)
        align = 'left'
    else:
        bins = 256
        align = 'mid'
    fig, axes = plt.subplots(21,10, figsize=(40, 24))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if not ('conv2d' in name or 'fc' in name):
            continue
        if param.dim() > 1:
            ax = axes[plot_index]
            if plot_type == 'histogram':
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        align=align, color = 'blue', alpha = 0.5,
                        edgecolor='black' if bitwidth <= 4 else None)
                if bitwidth <= 4:
                    quantized_min, quantized_max = get_quantized_range(bitwidth)
                    ax.set_xticks(np.arange(start=quantized_min, stop=quantized_max+1))
                ax.set_xlabel(name_to_multiple_line(name))
                ax.set_ylabel('density')
            else:
                ax.boxplot(param.detach().view(-1).cpu(), vert=False)
                ax.set_xlabel(name_to_multiple_line(name))
            plot_index += 1
            

    SS2D_weight_list = ["x_proj_weight", "dt_projs_weight", "dt_projs_bias", "A_logs", "Ds"]
    for module_name, module in model.named_modules():
        if isinstance(module, SS2D):
            for weight_name in SS2D_weight_list:
                param = getattr(module, weight_name)
                ax = axes[plot_index]
                if plot_type == 'histogram':
                    ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                            align=align, color = 'blue', alpha = 0.5,
                            edgecolor='black' if bitwidth <= 4 else None)
                    if bitwidth <= 4:
                        quantized_min, quantized_max = get_quantized_range(bitwidth)
                        ax.set_xticks(np.arange(start=quantized_min, stop=quantized_max+1))
                    ax.set_xlabel(name_to_multiple_line(f"{module_name}.{weight_name}"))
                    ax.set_ylabel('density')
                else:
                    ax.boxplot(param.detach().view(-1).cpu(), vert=False)
                    ax.set_xlabel(name_to_multiple_line(f"{module_name}.{weight_name}"))
                plot_index += 1
            
    fig.suptitle(f'Weights distribution (histogram)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(f'./{model_name}_weight_distribution_{plot_type}.png')

