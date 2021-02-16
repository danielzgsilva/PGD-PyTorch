import torch
import torch.nn as nn
import numpy as np

class PGD(nn.Module):
    def __init__(self, model, device, norm, eps, alpha, iters):
        super(PGD, self).__init__()
        assert(2 <= eps <= 10)
        assert(norm in [2, 'inf', np.inf])
        # epsilon, magnitude of perturbation, make sure to normalize to 0-1 range
        self.eps = eps / 255.0
        # step size
        self.alpha = alpha
        # l2 or linf
        self.norm = norm
        #iterations
        self.iterations = iters
        self.loss = nn.CrossEntropyLoss()
        self.model = model
        self.device = device

    def forward(self, images, labels):
        # applies PGD to a batch of images

        adv = images.clone().detach().requires_grad_(True).to(self.device)

        # run for desired number of iterations
        for i in range(self.iterations):
            _adv = adv.clone().detach().requires_grad_(True)

            # predict on current perturbation + input
            outputs = self.model(_adv)

            # compute classification loss
            self.model.zero_grad()
            cost = self.loss(outputs, labels)

            # calculate gradient with respect to the input
            cost.backward()
            grad = _adv.grad

            # normalize gradient into lp ball
            if self.norm in ["inf", np.inf]:
                grad = grad.sign()
            elif self.norm == 2:
                grad = grad / (torch.sqrt(torch.sum(grad * grad, dim=(1,2,3), keepdim=True)) + 10e-8)

            assert(images.shape == grad.shape)

            # take step in direction of gradient and apply to current example
            adv = adv + grad * self.alpha

            # project current example back onto Lp ball
            if self.norm in ["inf", np.inf]:
                adv = torch.max(torch.min(adv, images + self.eps), images - self.eps)

            elif self.norm == 2:
                d = adv - images
                mask = self.eps >= d.view(d.shape[0], -1).norm(self.norm, dim=1)
                scale = d.view(d.shape[0], -1).norm(self.norm, dim=1)
                scale[mask] = self.eps
                d *= self.eps / scale.view(-1, 1, 1, 1)
                adv = images + d

            # clamp into 0-1 range
            adv = adv.clamp(0.0, 1.0)

            # return adversarial example
            return adv.detach()


