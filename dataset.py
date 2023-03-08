from torch.utils.data import Dataset, DataLoader
import cv2
import os

class VD_dataset(Dataset):
    def __init__(self, rootdir):
        filenames = os.listdir(rootdir)
        self.rootdir = rootdir
        filenames_prefixs = []

        for filename in filenames:
            assert filename.split('_')[0] + '_target.png' in filenames
            assert filename.split('_')[0] + '_input.png' in filenames
            filenames_prefixs.append(filename.split('_')[0])

        self.filenames_prefixs = list(set(filenames_prefixs))

    def __len__(self):
        return len(self.filenames_prefixs)

    def __getitem__(self, idx):
        input = cv2.imread(os.path.join(self.rootdir, '{}_input.png'.format(self.filenames_prefixs[idx])))
        target = cv2.imread(os.path.join(self.rootdir, '{}_target.png'.format(self.filenames_prefixs[idx])))

        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        return input.transpose(2, 0, 1), target.transpose(2, 0, 1)


