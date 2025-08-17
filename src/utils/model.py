import torch
import torch.nn as nn
from collections import OrderedDict


def count_parameters(module):
    """Count the number of parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def count_direct_parameters(module):
    """Count only the parameters directly belonging to this module, not its children."""
    direct_params = set(module.parameters(recurse=False))
    return sum(p.numel() for p in direct_params)


def print_model_hierarchy(model, indent=0, prefix="", show_shapes=False, max_depth=None):
    """
    Print a hierarchical structure of a PyTorch model with parameter counts.

    Args:
        model: PyTorch model
        indent: Current indentation level
        prefix: Prefix for the current line
        show_shapes: If True, also show parameter shapes
        max_depth: Maximum depth to traverse (None for unlimited)
    """
    if max_depth is not None and indent > max_depth:
        return

    # Count parameters for this module and its children
    total_params = count_parameters(model)
    direct_params = count_direct_parameters(model)

    # Format the parameter count
    param_str = f"{total_params:,}"
    if direct_params > 0 and direct_params != total_params:
        param_str = f"{total_params:,} (direct: {direct_params:,})"

    # Get module type name
    module_type = model.__class__.__name__

    # Print current module
    indent_str = "  " * indent
    print(f"{indent_str}{prefix}{module_type}: {param_str} params")

    # Optionally show parameter shapes
    if show_shapes and direct_params > 0:
        for name, param in model.named_parameters(recurse=False):
            print(f"{indent_str}    └─ {name}: {list(param.shape)} = {param.numel():,} params")

    # Recursively print children
    children = list(model.named_children())
    for i, (name, child) in enumerate(children):
        is_last = (i == len(children) - 1)
        child_prefix = f"{name} - "
        print_model_hierarchy(child, indent + 1, child_prefix, show_shapes, max_depth)


def analyze_model(model, show_shapes=False, max_depth=None):
    """
    Analyze and print a comprehensive summary of a PyTorch model.

    Args:
        model: PyTorch model to analyze
        show_shapes: If True, show parameter shapes
        max_depth: Maximum depth to traverse
    """
    print("=" * 80)
    print("MODEL ARCHITECTURE HIERARCHY")
    print("=" * 80)

    # Print total parameters
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print("-" * 80)
    print()

    # Print hierarchical structure
    print_model_hierarchy(model, show_shapes=show_shapes, max_depth=max_depth)
    print("=" * 80)


def get_model_summary_dict(model, prefix=""):
    """
    Get a dictionary representation of the model hierarchy with parameter counts.

    Returns:
        dict: Nested dictionary with module names and parameter counts
    """
    # Add current module info
    total_params = count_parameters(model)
    direct_params = count_direct_parameters(model)

    module_info = {
        "type": model.__class__.__name__,
        "total_params": total_params,
        "direct_params": direct_params,
        "children": OrderedDict()
    }

    # Add children recursively
    for name, child in model.named_children():
        module_info["children"][name] = get_model_summary_dict(child, f"{prefix}{name}.")

    return module_info
