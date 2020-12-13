from imports import *
## TODO: Completely untested, all transform parameters (angle, range, etc.
## need to be tuned). USES TORCHVISION

def generic_random_transform(p_transform, data_orig, labels_orig, transform_func):
    x_transformed = np.empty_like(data_orig[0])
    x_transformed = np.expand_dims(x_transformed, axis=0)
    y_transformed = np.empty_like(labels_orig[0])
    # x_orig = np.empty_like(data_orig[0])
    # x_orig = np.expand_dims(x_orig,axis=0)

    for (target, example, idx) in zip(labels_orig, data_orig, range(len(labels_orig))):
        example = torch.from_numpy(example)
        if (len(example.shape) < 3):
            example = torch.unsqueeze(example, 0)

        if (rn.uniform(0, 1) < p_transform):
            # x_orig = np.append(x_orig,example.numpy(),axis=0)
            temp = transform_func(example).numpy()
            x_transformed = np.append(x_transformed, temp, axis=0)
            y_transformed = np.append(y_transformed, target)

    return x_transformed[1:], y_transformed[1:]


def brightness(p_transform, data_orig, labels_orig):
    x_transformed = np.empty_like(data_orig[0])
    x_transformed = np.expand_dims(x_transformed, axis=0)
    y_transformed = np.empty_like(labels_orig[0])
    # x_orig = np.empty_like(data_orig[0])
    # x_orig = np.expand_dims(x_orig,axis=0)

    for (target, example, idx) in zip(labels_orig, data_orig, range(len(labels_orig))):
        example = torch.from_numpy(example)
        if (len(example.shape) < 3):
            example = torch.unsqueeze(example, 0)

        if (rn.uniform(0, 1) < p_transform):
            # x_orig = np.append(x_orig,example.numpy(),axis=0)
            # temp = transform_func(example).numpy()
            factor = rn.uniform(0.2, 2)
            temp = example.numpy() * factor
            temp = np.clip(temp, 0, 1)
            x_transformed = np.append(x_transformed, temp, axis=0)
            y_transformed = np.append(y_transformed, target)

    return x_transformed[1:], y_transformed[1:]


def random_rotate(p_transform, data_orig, labels_orig):
    rotate = transforms.RandomRotation((-180, 180))
    return generic_random_transform(p_transform, data_orig, labels_orig, rotate)


def random_shift(p_transform, data_orig, labels_orig):
    shift = transforms.RandomAffine(degrees=0, translate=(0.125, 0.125))
    return generic_random_transform(p_transform, data_orig, labels_orig, shift)


def random_scale(p_transform, data_orig, labels_orig):
    scale = transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))
    return generic_random_transform(p_transform, data_orig, labels_orig, scale)


# Doesn't seem to work
def random_crop(p_transform, data_orig, labels_orig):
    crop = transforms.RandomCrop(data_orig[0].shape, pad_if_needed=True)
    return generic_random_transform(p_transform, data_orig, labels_orig, crop)


def random_affine(p_transform, data_orig, labels_orig):
    affine = transforms.RandomAffine(degrees=40, translate=(0.125, 0.125), scale=(0.8, 1.2))
    return generic_random_transform(p_transform, data_orig, labels_orig, affine)


# Doesn't seem to work
def random_brightness(p_transform, data_orig, labels_orig):
    #  brightness = transforms.ColorJitter(brightness=(0.5,1.5))
    #  return generic_random_transform(p_transform, data_orig, labels_orig, brightness)
    return brightness(p_transform, data_orig, labels_orig)


# Doesn't seem to work
def random_color(p_transform, data_orig, labels_orig):
    def color(example):
        return rn.uniform(0.8, 1.2) * example

    return generic_random_transform(p_transform, data_orig, labels_orig, color)


def random_blur(p_transform, data_orig, labels_orig):
    def blur(example):
        k = rn.randint(data_orig[0].shape[1] / 8)
        if (k % 2 == 0):
            k = k + 1
        blur_temp = transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1))
        return blur_temp(example)

    return generic_random_transform(p_transform, data_orig, labels_orig, blur)


def random_composite(p_transform, data_orig, labels_orig):
    def composite(example):
        temp = rn.uniform(0.8, 1.2) * example
        affine = transforms.RandomAffine(degrees=40, translate=(0.125, 0.125), scale=(1, 1.5))
        return affine(temp)

    return generic_random_transform(p_transform, data_orig, labels_orig, composite)


def random_composite_blur(p_transform, data_orig, labels_orig):
    def composite_blur(example):
        # temp = rn.uniform(0.8,1.2) * example
        temp = example
        affine = transforms.RandomAffine(degrees=40, translate=(0.125, 0.125), scale=(1, 1.5))
        temp = affine(temp)

        k = rn.randint(data_orig[0].shape[1] / 8)
        if (k % 2 == 0):
            k = k + 1
        blur_temp = transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1))
        return blur_temp(temp)

    return generic_random_transform(p_transform, data_orig, labels_orig, composite_blur)