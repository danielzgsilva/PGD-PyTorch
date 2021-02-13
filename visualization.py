import torch
from torchvision import datasets, transforms, models

from torchvision.utils import save_image

from pgd import PGD

if __name__ == '__main__':
    model = models.resnet34(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Correct Metadata URL
    datasets.imagenet.ARCHIVE_DICT['devkit'][
        'url'] = "https://github.com/goodclass/PythonAI/raw/master/imagenet/ILSVRC2012_devkit_t12.tar.gz"

    imagenet_data = datasets.ImageNet('~/Imagenet',
                                      transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()]), split='val')

    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=8,
                                              shuffle=True)  # TODO num_workers for multi-GPU

    pgd = PGD(model, device,
              norm='inf',
              eps=10,
              alpha=1,
              iters=20)

    model.to(device)
    model.eval()

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        adv_images = pgd(images, labels)

        with torch.no_grad():
            outputs = model(adv_images)
            predictions = torch.argmax(outputs.data, 1)

            adv_outputs = model(adv_images)
            adv_predictions = torch.argmax(adv_outputs.data, 1)

        bs = images.size(0)
        for i in range(bs):
            orig_image = images[i]
            adv_image = adv_images[i]

            gt = imagenet_data.classes[labels[i].item()]
            orig_pred = imagenet_data.classes[predictions[i].item()]
            adv_pred = imagenet_data.classes[adv_predictions[i].item()]

            orig_name = "orig_gt_{}_pred_{}.jpg".format(gt, orig_pred)
            adv_name = "adv_gt_{}_pred_{}.jpg".format(gt, adv_pred)

            save_image(orig_image, orig_name)
            save_image(adv_image, adv_name)

            break





