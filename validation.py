import torch
from tqdm import tqdm
from torchvision import datasets, transforms, models

from pgd import PGD


def main(experiment, pgd_params=None):
    model = None
    if experiment == 'resnet':
        model = models.resnet34(pretrained=True)
    elif experiment == 'vgg':
        model = models.vgg16(pretrained=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Correct Metadata URL
    datasets.imagenet.ARCHIVE_DICT['devkit']['url'] = "https://github.com/goodclass/PythonAI/raw/master/imagenet/ILSVRC2012_devkit_t12.tar.gz"
    
    imagenet_data = datasets.ImageNet('~/Imagenet',
                                      transform=transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize]), split='val')


    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=24,
                                              shuffle=True)  # TODO num_workers for multi-GPU

    pgd_attack = None
    if pgd_params is not None:
        pgd_attack = PGD(model, device,
                         norm=pgd_params['norm'],
                         eps=pgd_params['eps'],
                         alpha=pgd_params['alpha'],
                         iters=pgd_params['iterations'])

    accuracy = validation(model, data_loader, device, pgd_attack)
    print('Accuracy: ', accuracy * 100)


def validation(model, data_loader, device, pgd_attack=None):
    if model is None:
        print('null model')
        return -1

    model.to(device)
    model.eval()

    num_correct = 0
    total = 0
    batch = 0

    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        if pgd_attack is not None:
            images = pgd_attack(images, labels)

        with torch.no_grad():
            outputs = model(images)
            predictions = torch.argmax(outputs.data, 1)
            num_correct += (predictions == labels).sum().item()

        batch += 1
        total += images.size(0)

    return num_correct / total


if __name__ == '__main__':
    model_archs = ['resnet', 'vgg']
    norms = [2, "inf"]
    epsilons = [i for i in range(2, 11)]

    # run all experiments
    for arch in model_archs:
        # first get baseline accuracy for each model
        print('<----------- Baseline for {} ----------->'.format(arch))
        #main(arch)

        # then evaluate with PGD attack while varying parameters
        # norm - 2 or inf
        # epsilon - 2 to 10
        # max iterations - 2 x epsilon
        # step size (alpha) always 1

        for norm in norms:
            for eps in epsilons:
                alpha = 1
                iterations = 2 * eps

                print('<----------- PGD attack on {} with norm={} epsilon={} step size={} iterations={} ----------->'.
                      format(arch, norm, eps, alpha, iterations))

                pgd_params = {'norm': norm, 'eps': eps, 'alpha': alpha, 'iterations': iterations}
                main(arch, pgd_params)
