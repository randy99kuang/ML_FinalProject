from imports import *
from baseline_classifiers import *

def train_adver(model, optimizer, trainloader, name, epoch=2,
                loss_fcn=torch.nn.CrossEntropyLoss(), clip=False,
                plot=False, devloader=None, denoiser=None, adversary=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    plot_batch = 50  # save plotting data every 50 mini-batches
    print_batch = 4  # print every 4*50 = 200 mini-batches

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

            if (adversary and batch_count % 7 == 6):
                inputs = make_grad_attacks('temp', x=inputs.cpu().numpy(), y=labels.cpu().numpy(),
                                           model=model, loss_fcn=loss_fcn, method="PGD",
                                           step_size=2.5, chunk=50, epsilon=0.3)
                inputs = torch.from_numpy(inputs).to(device)

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
                if (plot):
                    iteration_track = np.append(iteration_track, batch_count)
                    dev_acc, dev_loss = test(model, devloader, loss_fcn=loss_fcn)
                    devacc_track = np.append(devacc_track, dev_acc)
                    devloss_track = np.append(devloss_track, dev_loss.cpu())

                    model.train()
    if (plot):
        fig, a = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        a[0].plot(iteration_track, devacc_track)
        a[0].set_title('Dev. Accuracy')
        a[0].set_xlabel("Total Iterations")
        # a[0].set_ylim([0, 1])
        a[1].plot(iteration_track, devloss_track)
        a[1].set_title('Dev. Loss')
        a[1].set_xlabel("Total Iterations")
        plt.show()

    PATH = "/content/drive/My Drive/ML Final Project Files/"  # Randy's Path
    SAVE_PATH = PATH + name + '.pth'
    torch.save(model.state_dict(), SAVE_PATH)

