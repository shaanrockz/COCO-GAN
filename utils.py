
import numpy as np
from torchvision.utils import save_image
import torch

def save_manifold_images(images, size, image_path):
    images = (images+1) / 2
    manifold_image = np.squeeze(compose_manifold_images(images, size))
    return save_image(torch.tensor(manifold_image, dtype=torch.float64), image_path)


def compose_manifold_images(images, size):
    h, w = images.shape[2], images.shape[3]
    if (images.shape[1] in (3,4)):
        c = images.shape[1]
        img = np.zeros((c, h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[:,j * h:j * h + h, i * w:i * w + w] = image
        return img
    elif images.shape[1]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[0,:,:]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ' + 
            'must have dimensions: HxW or HxWx3 or HxWx4, got {}'.format(images.shape))
