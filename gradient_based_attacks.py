from imports import *
from data_loading import *

## Spits it up into chuncks for efficient mini-batching
def make_grad_attacks(name, x, y, model, loss_fcn, method, step_size, chunk=50, epsilon=0.1,
                      rand=False, n_steps=1):
    x_aug = []
    idx = 0

    while idx < len(x):
        if (idx + chunk < len(x)):
            x_aug.extend(gen_grad_attack(x[idx:idx + chunk],
                                         y[idx:idx + chunk],
                                         model, loss_fcn, method, step_size,
                                         epsilon, rand=rand, n_steps=n_steps))
        else:
            x_aug.extend(gen_grad_attack(x[idx:len(x)],
                                         y[idx:len(y)],
                                         model, loss_fcn, method, step_size,
                                         epsilon, rand=rand, n_steps=n_steps))
        idx += chunk

    output = np.array(x_aug)
    PATH = "/content/drive/My Drive/ML Final Project Files/"  # Randy's Path
    with open(PATH + name + '.npy', 'wb') as f:
        np.save(f, output)
    return output


## Get gradient step on CPU as numpy array
def get_grad(x, y, model, loss_fcn, normalize=True):
    target = torch.clone(torch.from_numpy(y)).detach().to(device)
    example_tensor = torch.clone(
        torch.from_numpy(x)).detach().requires_grad_(True).to(device)

    if (len(example_tensor.shape) == 3):
        example_tensor = torch.unsqueeze(example_tensor, 1)

    if (len(target.shape) == 3):
        target = torch.unsqueeze(target, 1)

    loss = torch.zeros(1, requires_grad=True)
    output = torch.zeros(example_tensor.shape[0], 10, requires_grad=True)

    output = model(example_tensor.float())
    loss = loss_fcn(output, target.long())
    d_loss_dx = grad(outputs=loss, inputs=example_tensor)[0]
    d_loss_dx = d_loss_dx.cpu()

    if (len(d_loss_dx.shape) > 3):
        d_loss_dx = np.squeeze(d_loss_dx)

    norm = 1
    if (normalize):
        norm = torch.norm(d_loss_dx, dim=(1, 2)).numpy()
        norm[norm == 0] = 1
        return d_loss_dx.numpy() / norm[:, None, None]
    else:
        return d_loss_dx.numpy()


## Adapted from Madry et al, ICLR 2018
#######################################################################
# Type must be PGD, FGSM, BIM
def gen_grad_attack(x, y, model, loss_fcn, type, step_size, epsilon, rand, n_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    x_orig = x
    if rand:
        x = gen_rp(x, epsilon / 4)
    else:
        x = x

    d_loss_dx = get_grad(x, y, model, loss_fcn)
    if type == "PGD":
        augment_x = x + step_size * x * d_loss_dx
    elif type == "FGSM":
        augment_x = x + step_size * np.sign(d_loss_dx)
    elif type == "BIM":
        augment_x = x + step_size * x * d_loss_dx
        for i in range(n_steps - 1):
            d_loss_dx = get_grad(augment_x, y, model, loss_fcn)
            augment_x = augment_x + step_size * augment_x * d_loss_dx
    else:
        print("Unsupported TYPE (PGD, FGSM, or BIM)")
        return
    augment_x = np.clip(augment_x, x_orig - epsilon, x_orig + epsilon)
    return clean_images(augment_x)
#####################################################################