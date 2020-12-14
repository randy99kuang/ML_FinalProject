from imports import *

def print_info(labels, counts):
    total = np.sum(counts)
    for d in labels:
        print(f'Class: {d}, with {counts[d]} examples ({np.round(counts[d] / total, 3)})')


def clean_images(images_as_np):
    return np.nan_to_num(images_as_np, nan=0.5, posinf=1, neginf=0)


def load_data(X, Y, t_size=0.2, d_size=0.2, show=True):
    labels, counts = np.unique(Y, return_counts=True)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=t_size, random_state=rn.randint(101))
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                                                      test_size=d_size, random_state=rn.randint(42))
    x = [x_train, x_dev, x_test]
    y = [y_train, y_dev, y_test]

    for i in range(len(y)):
        y[i] = np.squeeze(y[i])

    if (show == True):
        print(f'=================== Data Summary =========================== \n'
              + f'Total: {Y.shape[0]} Examples')
        print_info(labels, counts)

        tr_labels, tr_counts = np.unique(y_train, return_counts=True)
        print(f'=================== Training Data =========================== \n'
              + f'Total: {y_train.shape[0]} Examples')
        print_info(tr_labels, tr_counts)

        d_labels, d_counts = np.unique(y_dev, return_counts=True)
        print(f'=================== Validation Data =========================== \n'
              + f'Total: {y_dev.shape[0]} Examples')
        print_info(d_labels, d_counts)

        t_labels, t_counts = np.unique(y_test, return_counts=True)
        print(f'=================== Test Data =========================== \n'
              + f'Total: {y_dev.shape[0]} Examples')
        print_info(t_labels, t_counts)
    # end if

    # label_dict = {"all":labels, "train": tr_labels, "dev": d_labels, "test": t_labels}
    count_dict = {"all": counts, "train": tr_counts, "dev": d_counts, "test": t_counts}
    data_dict = {"all": X, "train": x_train, "dev": x_dev, "test": x_test}
    target_dict = {"all": Y, "train": y_train, "dev": y_dev, "test": y_test}

    return count_dict, data_dict, target_dict


class ImageDataset(Dataset):
    def __init__(self, x_list, y_list):
        """
        Args: transform: to be applied on a sample automatically if desired
        """
        self.data = x_list
        self.label = y_list

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # sample = {'image': np.squeeze(self.data[idx]),
        #          'label': np.squeeze(self.label[idx])}
        sample = (np.squeeze(self.data[idx]), np.squeeze(self.label[idx]))
        return sample