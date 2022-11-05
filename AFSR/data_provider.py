import numpy as np
from AFSR.data_list import ImageList, ImageList_idx, make_dataset
import torch.utils.data as util_data
from torchvision import transforms
from PIL import Image, ImageOps


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


def load_images(images_file_path, batch_size, prefix, resize_size=256, is_train=True,
                crop_size=224, is_cen=False, num_worker=4):
    if not is_train:  # evaluation
        start_center = (resize_size - crop_size - 1) / 2
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
        ])
        images = ImageList_idx(add_prefix_to_patch(open(images_file_path).readlines(), prefix), transform=transformer)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    else:  # training
        if is_cen:
            transformer = transforms.Compose([ResizeImage(resize_size),
                                              transforms.Scale(resize_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(crop_size),
                                              transforms.ToTensor(),
                                              ])
        else:
            transformer = transforms.Compose([ResizeImage(resize_size),
                                              transforms.RandomResizedCrop(crop_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              ])
        images = ImageList_idx(add_prefix_to_patch(open(images_file_path).readlines(), prefix), transform=transformer)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                             drop_last=True)

    return images_loader


def load_images_sample_list(config, images_file_path, batch_size, prefix, resize_size=256, is_train=True,
                            crop_size=224, is_cen=False, num_worker=4):
    pathes = add_prefix_to_patch(open(images_file_path).readlines(), prefix)
    patch_list_class = [[] for _ in range(config["class_num"])]
    for content in pathes:
        target = int(content.split()[1])
        patch_list_class[int(target)].append(content)

    num_samples_class = int((len(config["t_train_loader"]) + 1) * config["batch_size_train"] / config["class_num"])
    samples = []
    for category in range(config["class_num"]):
        path_category = np.array(patch_list_class[category])
        indexes = np.random.choice(len(path_category), num_samples_class, replace=True)
        sampled = path_category[indexes]
        samples.extend(sampled)
    samples = np.array(samples)
    np.random.shuffle(samples.flat)
    images_list = samples.tolist()

    if not is_train:  # evaluation
        start_center = (resize_size - crop_size - 1) / 2
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
        ])
        images = ImageList_idx(images_list, transform=transformer)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    else:  # training
        if is_cen:
            transformer = transforms.Compose([ResizeImage(resize_size),
                                              transforms.Scale(resize_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(crop_size),
                                              transforms.ToTensor(),
                                              ])
        else:
            transformer = transforms.Compose([ResizeImage(resize_size),
                                              transforms.RandomResizedCrop(crop_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              ])
        images = ImageList_idx(images_list, transform=transformer)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                             drop_last=True)

    return images_loader


def add_prefix_to_patch(image_list, prefix):
    for i in range(len(image_list)):
        line = image_list[i]
        line = line.strip('\n')
        # abs_path = os.path.join(prefix, line)
        abs_path = prefix + line
        image_list[i] = abs_path
    return image_list
