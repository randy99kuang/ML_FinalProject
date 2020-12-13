from imports import *
## Ready for Annie (NOT YET)
# grad_edsr_transform
# rand_edsr_transform
# blur_edsr_transform

# mnist_ff_transform  # Trained with transforms
# mnist_cnn_transform
# mnist_comp_transform

# grad_attack_transform_testloader
# rand_attack_transform_testloader
# blur_transform_testloader
# transform_attack_testloader

## Run with test-sets ([insert name here])
# mnist_test
## Annie
# denoiser_dict = {0: noattack, 1: grad_edsr_transform, 2: rand_edsr_transform, 3: blur_edsr_transform}
def denoiser_linker(input, images, encoding_method, denoiser_dict):
    # input = softmax prob from the adversarial detector
    # denoiser_dict = {0: noattack, 1: grad_edsr_transform, 2: rand_edsr_transform, 3: blur_edsr_transform}
    if encoding_method == 'one_hot':
        k = torch.argmax(input, 1)
        k = k.detach().cpu().numpy()
        # print(type(k))
        denoised_images = torch.empty((len(k), 1, 28, 28))
        # denoised_image = torch.empty()
        for i in range(len(k)):
            image = images[i, :, :, :].reshape((1, 1, 28, 28))
            denoiser_func = denoiser_dict[k[i]]
            denoised_image = denoiser_func(image.float())
            denoised_images[i, :, :, :] = denoised_image

    elif encoding_method == 'weighted':
        denoised_images = torch.empty((input.shape[0], 1, 28, 28))
        for im in range(input.shape[0]):
            image = images[im, :, :, :].reshape((1, 1, 28, 28))
            denoised_unweighted = {}
            image_sum = torch.zero((1, 1, 28, 28))
            for i in range(len(denoiser_dict.keys())):
                denoiser_func = denoiser_dict[i]
                denoised_unweighted[i] = denoiser_func(image)
                image_sum += denoised_unweighted[i] * input[im, i]
            image_sum = torchvision.transforms.Normalize(image_sum)
            denoised_images[i, :, :, :] = image_sum

    return denoised_images


def noattack(images):
    return images


def test_pipeline(denoiser_dict, detector, model, testloader, weight_scheme, trained=True):  # Use a loss fcn for dev_set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    pred_all = [];
    softmax_all = np.empty((0, 10), float)
    # if (trained):
    #     denoiser_dict = {0: noattack, 1: grad_edsr_transform, 2: rand_edsr_transform, 3: blur_edsr_transform}
    # else:
    #     denoiser_dict = {0: noattack, 1: grad_denoiser_conv, 2: rand_denoiser_conv, 3: blur_edsr_transform}
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            try:
                inputs, labels = inputs.to(device), labels.to(device)
            except:
                print("Found String in Data")
                continue
            if (len(labels.shape) == 3):
                labels = torch.unsqueeze(labels, 1)
            if (len(inputs.shape) == 3):
                inputs = torch.unsqueeze(inputs, 1)

            # Adversarial detector
            score = detector(inputs.float())
            score = torch.nn.functional.softmax(score, dim=1)
            if (weight_scheme == 'one_hot'):
                inputs = denoiser_linker(score, inputs, weight_scheme, denoiser_dict)
                inputs = inputs.to(device)

                output = model(inputs.float())
            else:
                output_ensemble = torch.zeros((inputs.shape[0], 10)).to(device)
                for idx in range(4):
                    denoiser_fcn = denoiser_dict[idx]
                    try:
                        denoiser_fcn = denoiser_fcn.to(device)
                    except:
                        denoiser_fcn = denoiser_fcn
                    output_ensemble += torch.unsqueeze(score[:, idx], -1) * model(denoiser_fcn(inputs.float()))
                output = output_ensemble

            if (len(output.shape) < 2):
                output = torch.unsqueeze(output, 0)
            _, pred = torch.max(output.data, 1)
            softmax_score = torch.nn.functional.softmax(output, dim=1)
            pred_all.append(pred.detach().cpu().numpy())

            softmax_all = np.append(softmax_all, softmax_score.cpu().detach().numpy(), axis=0)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return pred_all, softmax_all, (correct / total)

