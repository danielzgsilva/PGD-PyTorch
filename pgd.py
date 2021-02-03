import torch
import torch.nn as nn

class PGD(nn.Module):
    def __init__(self, model, eps, alpha, p):
        super(PGD, self).__init__()
        assert(2 <= eps <= 10)
        self.eps = eps
        self.alpha = 1
        self.iterations = 2 * eps
        self.loss = nn.CrossEntropyLoss()
        self.model = model

    def forward(self, images, labels):
        ori_images = images.data

        for i in range(self.iterations):
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = self.loss(outputs, labels)
            cost.backward()

            adv_images = images + self.alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

        return images


