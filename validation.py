import torch
from tqdm import tqdm
from torchvision import datasets, transforms, models

from pgd import PGD


def main(experiment, apply_pgd=False):
    model = None
    if experiment == 'resnet':
        model = models.resnet34(pretrained=True)
    elif experiment == 'vgg':
        model = models.vgg16(pretrained=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    imagenet_data = datasets.ImageFolder('~/Imagenet/validation/',
                                      transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize]))

    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=24,
                                              shuffle=True)  # TODO num_workers for multi-GPU

    accuracy = validation(model, data_loader, apply_pgd, device)
    print('Accuracy: ', accuracy * 100)


def validation(model, data_loader, apply_pgd, device):
    if model is None:
        return

    model.to(device)
    model.eval()

    pgd_attack = PGD(model, device, 5, 1, 2)
    num_correct = 0
    total = 0

    batch = 0
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        if apply_pgd:
            images = pgd_attack(images, labels)

        with torch.no_grad():
            outputs = model(images)
            predictions = torch.argmax(outputs.data, 1)
            num_correct += (predictions == labels).sum().item()

        batch += 1
        total += images.size(0)

    return num_correct / total


if __name__ == '__main__':
    print('no pgd: ')
    main('resnet')

    print('pgd: ')
    main('resnet', True)
