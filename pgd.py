import torch
import torch.nn as nn
import numpy as np

class PGD(nn.Module):
    def __init__(self, model, device, norm, eps, alpha, iters):
        super(PGD, self).__init__()
        assert(2 <= eps <= 10)
        assert(norm in [2, 'inf', np.inf])
        self.eps = eps / 255.0
        self.alpha = alpha
        self.norm = norm
        self.iterations = iters
        self.loss = nn.CrossEntropyLoss()
        self.model = model
        self.device = device

    def forward(self, images, labels):
        adv = images.clone().detach().requires_grad_(True).to(self.device)

        for i in range(self.iterations):
            _adv = adv.clone().detach().requires_grad_(True)
            outputs = self.model(_adv)

            self.model.zero_grad()
            cost = self.loss(outputs, labels)
            cost.backward()
            grad = _adv.grad

            if self.norm in ["inf", np.inf]:
                grad = grad.sign()

            elif self.norm == 2:
                ind = tuple(range(1, len(images.shape)))
                grad = grad / (torch.sqrt(torch.sum(grad * grad, dim=ind, keepdim=True)) + 10e-8)

            assert(images.shape == grad.shape)
            adv = adv + grad * self.alpha

            # project back onto Lp ball
            if self.norm in ["inf", np.inf]:
                adv = torch.max(torch.min(adv, images + self.eps), images - self.eps)

            elif self.norm == 2:
                delta = adv - images

                mask = delta.view(delta.shape[0], -1).norm(self.norm, dim=1) <= self.eps

                scaling_factor = delta.view(delta.shape[0], -1).norm(self.norm, dim=1)
                scaling_factor[mask] = self.eps

                delta *= self.eps / scaling_factor.view(-1, 1, 1, 1)

                adv = images + delta

            adv = adv.clamp(0.0, 1.0)

            return adv.detach()


