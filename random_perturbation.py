from imports import *
from data_loading import *

def make_rand_attacks(name, x, y, model, method, step_size, chunk=50,
                      epsilon=0.1, n_steps=1):
    x_aug = []
    idx = 0

    while idx < len(x):
        if (idx + chunk < len(x)):
            x_aug.extend(gen_rand_attack(x[idx:idx + chunk],
                                         y[idx:idx + chunk],
                                         model, method, step_size,
                                         epsilon, n_steps))
        else:
            x_aug.extend(gen_rand_attack(x[idx:len(x)],
                                         y[idx:len(y)],
                                         model, method, step_size,
                                         epsilon, n_steps))
        idx += chunk
        print(idx)

    output = np.array(x_aug)
    PATH = "/content/drive/My Drive/ML Final Project Files/"  # Randy's Path
    with open(PATH + name + '.npy', 'wb') as f:
        np.save(f, output)
    return output


def gen_rp(x, epsilon):
    r = rn.uniform(-epsilon, epsilon, x.shape)
    return np.clip(x + r, 0, 1)  # Pixel range


####################################################################
# Guo, C., Gardner, J., You, Y., Wilson, A. and Weinberger, K., 2019.
# Simple Black-Box Adversarial Attacks
def get_probs(model, x, y):
    if (len(x.shape) < 4):
        x = torch.unsqueeze(x, 1)
    output = model(x.float())
    return torch.nn.functional.softmax(output, dim=1)


# (untargeted) SimBA for batch

def simba_single(model, x, y, num_iters=10000, step_size=0.2, epsilon=0.2):
    im = torch.clone(
        torch.from_numpy(x)).detach().requires_grad_(True).to(device)
    im_orig = torch.clone(im).detach().to("cpu")
    for i in range(num_iters):
        last_prob = get_probs(model, im, y)
        x_L = torch.clone(im).detach().to(device)
        x_R = torch.clone(im).detach().to(device)
        for j in range(im.shape[0]):
            dim1 = rn.randint(im[j].shape[0])
            dim2 = rn.randint(im[j].shape[1])
            x_L[j, dim1, dim2] = x_L[j, dim1, dim2] - step_size
            x_R[j, dim1, dim2] = x_R[j, dim1, dim2] + step_size
        x_L = torch.clone(x_L).detach().to("cpu")
        x_L = np.clip(x_L, im_orig - epsilon, im_orig + epsilon)
        x_L = torch.clone(x_L).detach().to(device)
        x_L = x_L.clamp(0, 1)
        #################################################
        x_R = torch.clone(x_R).detach().to("cpu")
        x_R = np.clip(x_R, im_orig - epsilon, im_orig + epsilon)
        x_R = torch.clone(x_R).detach().to(device)
        x_R = x_R.clamp(0, 1)
        left_prob = get_probs(model, x_L, y)
        right_prob = get_probs(model, x_R, y)
        for j in range(im.shape[0]):
            if left_prob[j, y[j]] < last_prob[j, y[j]]:
                im[j, :, :] = x_L[j, :, :]
            elif right_prob[j, y[j]] < last_prob[j, y[j]]:
                im[j, :, :] = x_R[j, :, :]
    ret = im.detach().to("cpu").numpy()
    return ret


####################################################################

# Method will either be "RP" or "SimBA"
def gen_rand_attack(x, y, model, method, step_size, epsilon, n_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (model):
        model.to(device)
        model.eval()

    if method == "RP":
        augment_x = gen_rp(x, epsilon)
    elif method == "SimBA":
        augment_x = simba_single(model=model, x=x, y=y, num_iters=n_steps,
                                 step_size=step_size, epsilon=epsilon)
    else:
        print("Unsupported TYPE (RP or SimBA)")
        return
    augment_x = np.clip(augment_x, x - epsilon, x + epsilon)
    return clean_images(augment_x)

