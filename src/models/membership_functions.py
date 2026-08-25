import torch


def gaussian_membership(xx, center, std):
    return torch.exp(-0.5 * ((xx - center) / std) ** 2)


def triangular_membership(xx, lower, center, upper):
    return torch.clamp_min(torch.min((xx - lower) / (center - lower), (upper - xx) / (upper - center)), 0)


def trapezoidal_membership(xx, lower, center_left, center_right, upper):
    left = (xx - lower) / (center_left - lower)
    middle = torch.ones_like(xx)
    right = (upper - xx) / (upper - center_right)
    return torch.clamp_min(torch.min(torch.min(left, right), middle), 0)


def sigmoid_membership(xx, center, slope):
    return 1 / (1 + torch.exp(-slope * (xx - center)))


def bell_membership(xx, center, width, slope):
    return 1 / (1 + torch.abs((xx - center) / width) ** (2 * slope))


MEMBERSHIP_PARAM_COUNTS = {"gaussian": 2, "triangular": 3, "trapezoidal": 4, "sigmoid": 2, "bell": 3}
MEMBERSHIP_FUNCTIONS = {"gaussian": gaussian_membership, "triangular": triangular_membership,
                        "trapezoidal": trapezoidal_membership, "sigmoid": sigmoid_membership, "bell": bell_membership}
