from imports import *

class FF(torch.nn.Module):
    def __init__(self, in_flat_dim, up1, out_dim, h1, h2):
        super(FF, self).__init__()
        self.flat_dim = up1 * up1 * in_flat_dim
        self.up0 = torch.nn.UpsamplingBilinear2d(scale_factor=up1)
        self.linear1 = torch.nn.Linear(up1 * up1 * in_flat_dim, h1)
        self.linear2 = torch.nn.Linear(h1, h2)
        self.linear3 = torch.nn.Linear(h2, out_dim)

    def forward(self, x):
        x = self.up0(x)
        x = x.reshape(x.shape[0], self.flat_dim)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)  # Need softmax outside (CE loss)


class CNN(torch.nn.Module):
    def __init__(self, up1, out_dim, chan1, chan2, chan3, k1, k2, k3, p12, color):
        super(CNN, self).__init__()
        self.up0 = torch.nn.UpsamplingBilinear2d(scale_factor=up1)
        self.conv1 = torch.nn.Conv2d(color, chan1, kernel_size=k1)
        self.conv2 = torch.nn.Conv2d(chan1, chan2, kernel_size=k2)
        self.conv3 = torch.nn.Conv2d(chan2, chan3, k3)
        self.linear4 = torch.nn.Linear(chan3, out_dim)
        self.pool12 = nn.MaxPool2d(p12, p12)

    def forward(self, x):
        x = self.up0(x)
        x = F.relu(self.conv1(x))
        x = self.pool12(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        pool_lin = nn.MaxPool2d(x.shape[2], x.shape[3])
        x = pool_lin(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.relu(self.linear4(x))  # Need softmax outside (CE loss)
        return x


class composite(torch.nn.Module):
    def __init__(self, up1, out_dim, chan1, chan2, chan3, k1, k2, k3, h4, h5, p12, p23, color):
        super(composite, self).__init__()
        self.up0 = torch.nn.UpsamplingBilinear2d(scale_factor=up1)
        self.conv1 = torch.nn.Conv2d(color, chan1, kernel_size=k1)
        self.conv2 = torch.nn.Conv2d(chan1, chan2, kernel_size=k2)
        self.conv3 = torch.nn.Conv2d(chan2, chan3, kernel_size=k3)
        self.linear4 = torch.nn.Linear(chan3, h4)
        self.linear5 = torch.nn.Linear(h4, h5)
        self.linear6 = torch.nn.Linear(h5, out_dim)

        self.pool12 = nn.MaxPool2d(p12, p12)
        self.pool23 = nn.MaxPool2d(p23, p23)

    def forward(self, x):
        x = self.up0(x)
        x = F.relu(self.conv1(x))
        x = self.pool12(x)
        x = F.relu(self.conv2(x))
        x = self.pool23(x)
        x = F.relu(self.conv3(x))
        pool_lin = nn.MaxPool2d(x.shape[2], x.shape[3])
        x = pool_lin(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        return self.linear6(x)  # Need softmax outside (CE loss)

def train(model, optimizer, trainloader, name, epoch=2,
          loss_fcn=torch.nn.CrossEntropyLoss(), clip=False,
          plot=False, devloader=None, denoiser=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    plot_batch = 50  # save plotting data every 50 mini-batches
    print_batch = 1  # print every 4*50 = 200 mini-batches

    iteration_track = []
    devloss_track = []
    devacc_track = []
    batch_count = 0

    running_loss = 0
    for ep in range(epoch):
        print_idx = 0
        for i_batch, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            inputs = inputs.float()

            if (len(labels.shape) == 3):
                labels = torch.unsqueeze(labels, 1)

            if (len(inputs.shape) == 3):
                inputs = torch.unsqueeze(inputs, 1)

            if (denoiser):
                inputs = denoiser(inputs)

            output = model(inputs)

            loss = loss_fcn(output, labels)
            loss.backward()
            if (clip == True):
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.00001)
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            if i_batch % plot_batch == plot_batch - 1:
                if (print_idx % print_batch == print_batch - 1):
                    print('[%d, %5d] loss: %.3f' %
                          (ep + 1, i_batch + 1, running_loss / plot_batch))
                print_idx += 1
                running_loss = 0.0
                if plot:
                    iteration_track = np.append(iteration_track, batch_count)
                    dev_acc, dev_loss = test(model, devloader, loss_fcn=loss_fcn)
                    devacc_track = np.append(devacc_track, dev_acc)
                    devloss_track = np.append(devloss_track, dev_loss.cpu())

                    model.train()
    if plot:
        fig, a = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        a[0].plot(iteration_track, devacc_track)
        a[0].set_title('Dev. Accuracy')
        a[0].set_xlabel("Total Iterations")
        # a[0].set_ylim([0, 1])
        a[1].plot(iteration_track, devloss_track)
        a[1].set_title('Dev. Loss')
        a[1].set_xlabel("Total Iterations")
        plt.show()

    PATH = "/content/drive/My Drive/ML Final Project Files/"
    SAVE_PATH = PATH + name + '.pth'
    torch.save(model.state_dict(), SAVE_PATH)


def test(model, testloader, loss_fcn=None, denoiser=None):  # Use a loss fcn for dev_set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.float()
            try:
                inputs, labels = inputs.to(device), labels.to(device)
            except:
                print("Found String in Data")
                continue
            if (len(labels.shape) == 3):
                labels = torch.unsqueeze(labels, 1)
            if (len(inputs.shape) == 3):
                inputs = torch.unsqueeze(inputs, 1)

            if (denoiser):
                inputs = denoiser(inputs)

            output = model(inputs.float())
            if (len(output.shape) < 2):
                output = torch.unsqueeze(output, 0)

            if (loss_fcn):
                total_loss += loss_fcn(output, labels)

            _, pred = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return (correct / total), (total_loss / total)
