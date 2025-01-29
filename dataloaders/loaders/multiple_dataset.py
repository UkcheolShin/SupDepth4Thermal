import random
from torch.utils.data import Dataset


class MultipleDataLoader(Dataset):
    """
    Dataset that handles multiple datasets
    """
    def __init__(self, dataset_list, opt, num_out_scales=1, shuffle=True):
        """
        Initialize
        :param subsets: params for sets
        :param frame_ids:
        :param augment:
        :param shuffle: if shuffle is true, then the return order of non-master subset items is not fixed.
        :param kwargs: other keyword parameters
        """
        # params
        self._shuffle = shuffle
        if opt.mode == 'parallel':
            self._mode_parallel = True
            self._getter = self.get_item_parallel
        else:
            self._mode_parallel = False
            self._getter = self.get_item_sequential

        self._subsets = dataset_list
        self._master = self._subsets[0]

        # compute data length
        self._data_len = {_set.name: len(_set) for _set in self._subsets}
        self._cum_len  = [0] + [len(_set) for _set in self._subsets]
        for i in range(1, len(self._cum_len)):
            self._cum_len[i] += self._cum_len[i-1]

        self._orders = {}
        self.make_orders()

    def make_orders(self):
        master_len = self._data_len[self._master.name]
        for name, length in self._data_len.items():
            if name != self._master:
                ids = list(range(length)) if length <= master_len else random.sample(range(length), master_len)
                random.shuffle(ids)
                self._orders[name] = ids

    def when_epoch_over(self):
        """
        Make sure this function be executed after every epoch, otherwise the pair combination would be fixed.
        :return:
        """
        if self._mode_parallel:
            self.make_orders()

    def get_item_sequential(self, idx):
        set_idx = 0
        for i in range(1, len(self._cum_len)):
            if idx < self._cum_len[i]:
                set_idx = i-1
                idx -= self._cum_len[i-1]
                break
        item = self._subsets[set_idx][idx]
        return item

    def get_item_parallel(self, idx):
        item = {}
        for subset in self._subsets:
            name = subset.name
            sub_idx = idx % self._data_len[name]
            item[name] = subset[self._orders[name][sub_idx]]
        return item

    def __getitem__(self, idx):
        return self._getter(idx)

    def __len__(self):
        if self._mode_parallel:
            return self._cum_len[1]
        else:
            return self._cum_len[-1]
