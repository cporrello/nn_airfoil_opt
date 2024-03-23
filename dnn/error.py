import torch


# function to compute relative L2 error in percentage on an entire dataset
def rel_l2_error_on_set(pred, true):
    """A helper function to compute the relative L2 error in percentage

    Args:
        pred (torch.Tensor): Predicted values
        true (torch.Tensor): True values

    Returns:
        torch.Tensor: Relative L2 error in percentage
    """
    return (torch.norm(pred - true, dim=1) / torch.norm(true, dim=1))*100