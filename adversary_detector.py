from imports import *
# Copy, shuffle, and sample

def cosham(x,y,target,frac_size,dest_x,dest_y):
  size = frac_size * (x.shape)[0]
  size = int(np.floor(size))
  temp = np.copy(x)
  rn.shuffle(temp)
  dest_x = np.append(dest_x,temp[0:size],axis=0)
  temp_y = target * np.ones_like(y)
  dest_y = np.append(dest_y,temp_y[0:size],axis=0)
  return dest_x, dest_y


class detector(torch.nn.Module):
    def __init__(self, up1, out_dim, chan1, chan2, chan3, chan4, chan5, k1, k2, k3, k4, k5,
                 h6, h7, h8, p23, p34, color):
        super(detector, self).__init__()
        self.up0 = torch.nn.UpsamplingBilinear2d(scale_factor=up1)

        self.conv1 = torch.nn.Conv2d(color, chan1, kernel_size=k1)
        self.bn1 = torch.nn.BatchNorm2d(chan1)
        self.LR1 = torch.nn.LeakyReLU(0.1)

        self.conv2 = torch.nn.Conv2d(chan1, chan2, kernel_size=k2)
        self.bn2 = torch.nn.BatchNorm2d(chan2)
        self.LR2 = torch.nn.LeakyReLU(0.1)

        self.conv3 = torch.nn.Conv2d(chan2, chan3, kernel_size=k3)
        self.bn3 = torch.nn.BatchNorm2d(chan3)
        self.LR3 = torch.nn.LeakyReLU(0.1)

        self.conv4 = torch.nn.Conv2d(chan3, chan4, kernel_size=k4)
        self.bn4 = torch.nn.BatchNorm2d(chan4)
        self.LR4 = torch.nn.LeakyReLU(0.1)

        self.conv5 = torch.nn.Conv2d(chan4, chan5, kernel_size=k5)
        self.bn5 = torch.nn.BatchNorm2d(chan5)
        self.LR5 = torch.nn.LeakyReLU(0.1)

        self.linear6 = torch.nn.Linear(chan5, h6)
        self.linear7 = torch.nn.Linear(h6, h7)
        self.linear8 = torch.nn.Linear(h7, h8)
        self.linear9 = torch.nn.Linear(h8, out_dim)

        self.pool23 = nn.MaxPool2d(p23, p23)
        self.pool34 = nn.MaxPool2d(p34, p34)

    def forward(self, x):
        x = x.float()
        if (len(x.shape) < 4):
            x = torch.unsqueeze(x, 1)
        x = self.up0(x)
        # x = self.LR1(self.bn1(self.conv1(x)))
        x = self.LR1(self.conv1(x))
        # x = self.LR2(self.bn2(self.conv2(x)))
        x = self.LR2(self.conv2(x))
        x = self.pool23(x)
        # x = self.LR3(self.bn3(self.conv3(x)))
        x = self.LR3(self.conv3(x))
        x = self.pool34(x)
        # x = self.LR4(self.bn4(self.conv4(x)))
        x = self.LR4(self.conv4(x))
        # x = self.LR5(self.bn5(self.conv5(x)))
        x = self.LR5(self.conv5(x))
        pool_lin = nn.MaxPool2d(x.shape[2], x.shape[3])
        x = pool_lin(x)
        x = torch.squeeze(x)
        x = F.relu(self.linear6(x))
        x = F.relu(self.linear7(x))
        x = F.relu(self.linear8(x))
        return self.linear9(x)  # Need softmax outside (CE loss)


# This is Pytorch's example for reference. 57% on Cifar-10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_by_class(model, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    num_classes = 4
    predictions = [0] * num_classes
    total = [0] * num_classes
    recall = [0] * num_classes
    precision = [0] * num_classes
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

            output = model(inputs.float())
            if (len(output.shape) < 2):
                output = torch.unsqueeze(output, 0)
            _, pred = torch.max(output.data, 1)

            for i in range(labels.size(0)):
                predictions[pred[i]] = predictions[pred[i]] + 1
                total[labels[i]] = total[labels[i]] + 1
                if pred[i] == labels[i]:
                    recall[pred[i]] += 1

    print('Predicted labels:', predictions)
    print('Correct labels:', total)

    for i in range(num_classes):
        precision[i] = recall[i] / predictions[i]
    print('Precision by class:', precision)

    for i in range(num_classes):
        recall[i] = recall[i] / total[i]
    print('Recall by class:', recall)