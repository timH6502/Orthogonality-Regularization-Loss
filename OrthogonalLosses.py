import torch
from torch import nn


class SoftOrthogonalityLoss(nn.modules.loss._Loss):
    """
    Encurages orthogonality in weight matrices by penalizing the Frobenius norm of
    the difference between Gram matrix and identity matrix.
    """

    def __init__(self, exclude_head: bool = True, head_name: str = 'head', gamma: float = 1e-6) -> None:
        """
        Initialize soft orthogonality loss.

        Parameters
        ----------
        exclude_head : bool, default=True
            If true, the module with the name `head_name` will be ignored.
        head_name : str, default='head'
            Name of the head module to exclude.
        gamma : float, default=1e-6
            Scaling factor for the loss magnitude.
        """
        super().__init__()
        self.exclude_head = exclude_head
        self.head_name = head_name
        self.gamma = gamma

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute soft orthogonality regularization loss across all linear/conv layers.

        Parameters
        ----------
        model : nn.Module
            The module whose layers will be used to calculate the loss.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, module in model.named_modules():
            if not self.exclude_head or self.head_name not in name:
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    if isinstance(module, nn.Linear):
                        weight_matrix = module.weight
                    else:
                        weight_matrix = module.weight.view(
                            module.out_channels, -1)
                    if weight_matrix.shape[1] < weight_matrix.shape[0]:
                        gram_matrix = torch.matmul(
                            weight_matrix.t(), weight_matrix)
                    else:
                        gram_matrix = torch.matmul(
                            weight_matrix, weight_matrix.t())
                    identity = torch.eye(
                        gram_matrix.shape[0],
                        gram_matrix.shape[1],
                        device=gram_matrix.device,
                        dtype=gram_matrix.dtype)
                    diff = gram_matrix - identity
                    loss += torch.norm(diff, p='fro',
                                       dim=(-2, -1)) * self.gamma
        return loss


class SpectralRestrictedIsometryLoss(nn.modules.loss._Loss):
    """
    Encurages orthogonality py calculating the spectral restricted isometry property.
    More information can be found here: https://github.com/timH6502/Orthogonality-Regularization-Loss
    """

    def __init__(self,
                 exclude_head: bool = True,
                 head_name: str = 'head',
                 gamma: float = 1e-6,
                 n_iterations: int = 2,
                 eps: float = 1e-6) -> None:
        """
        Initialize spectral restricted isometry loss.

        Parameters
        ----------
        exclude_head : bool, default=True
            If true, the module with the name `head_name` will be ignored.
        head_name: str, default='head'
            Name of the head module to exclude.
        gamma : float, default=1e-6
            Scaling factor for the loss magnitude.
        n_iterations : int, default=2
            Number of power iterations for spectral norm estimation.
        eps : float, default=1e-6
            Epsilon for numerical stability.
        """
        super().__init__()
        self.exclude_head = exclude_head
        self.head_name = head_name
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.eps = eps

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute spectral restricted isometry loss across all linear/conv layers.

        Parameters
        ----------
        model : nn.Module
            The module whose layers will be used to calculate the loss.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, module in model.named_modules():
            if not self.exclude_head or self.head_name not in name:
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    if isinstance(module, nn.Linear):
                        weight_matrix = module.weight
                    else:
                        weight_matrix = module.weight.view(
                            module.out_channels, -1)
                    if weight_matrix.shape[1] < weight_matrix.shape[0]:
                        gram_matrix = torch.matmul(
                            weight_matrix.t(), weight_matrix)
                    else:
                        gram_matrix = torch.matmul(
                            weight_matrix, weight_matrix.t())
                    identity = torch.eye(
                        gram_matrix.shape[0],
                        gram_matrix.shape[1],
                        device=gram_matrix.device,
                        dtype=gram_matrix.dtype)
                    diff = gram_matrix - identity

                    v = torch.randn((diff.shape[0], 1), device=diff.device)
                    v = v / (torch.norm(v, p='fro', dim=(-2, -1)) + self.eps)
                    sigma = torch.tensor(1.0, device=diff.device)

                    for _ in range(self.n_iterations):
                        u = torch.matmul(diff, v)
                        sigma = torch.norm(u, p='fro', dim=(-2, -1))
                        u = u / (sigma + self.eps)
                        v = torch.matmul(diff, u)
                        v = v / (torch.norm(v, p='fro',
                                 dim=(-2, -1)) + self.eps)
                    loss += self.gamma * sigma
        return loss
