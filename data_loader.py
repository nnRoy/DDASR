
import torch
import torch.utils.data as data
import numpy as np
from helper import PAD_ID, SOS_ID, EOS_ID
from utils import read_corpus, load_dict

class APIdataset(data.Dataset):
    def __init__(self, desc_file, api_file, vocab_desc_file, vocab_api_file, max_desc_seq_len, max_api_seq_len, max_longtail_len, vocab_general_file, vocab_langtail_file):
        self.max_desc_seq_len = max_desc_seq_len
        self.max_api_seq_len = max_api_seq_len
        self.max_longtail_len = max_longtail_len


        vocab_desc = load_dict(vocab_desc_file)
        vocab_api = load_dict(vocab_api_file)
        vocab_general = load_dict(vocab_general_file)
        vocab_langtail = load_dict(vocab_langtail_file)

        i = 0
        desc_data = []
        desc_index = []
        for line in open(desc_file):
            sents = line.strip().split(' ')
            desc_index.append((len(sents), i))
            i = i + len(sents)
            for sent in sents:
                if sent in vocab_desc:
                    desc_data.append(vocab_desc.get(sent))
                else:
                    desc_data.append(3)

        array_desc_data = np.zeros(len(desc_data),dtype=np.long)

        for m in range(len(desc_data)):
            array_desc_data[m] = desc_data[m]
        array_desc_index = np.zeros(shape=(len(desc_index),2),dtype=np.int)
        for m in range(len(desc_index)):
            array_desc_index[m] = desc_index[m]

        self.desc_data = array_desc_data[:].astype(np.long)
        self.desc_index = array_desc_index

        i = 0
        api_data = []
        api_index = []
        api_longtail = []

        for line in open(api_file):
            sents = line.strip().split(' ')
            api_index_num = 0
            tail_list = []
            for sent in sents:
                if sent in vocab_general.keys():
                    api_index_num += 1
                    api_data.append(vocab_api.get(sent))
                    api_longtail.append(tail_list)
                    tail_list = []
                elif sent in vocab_langtail.keys():
                    if len(tail_list) < max_longtail_len:
                        tail_list.append(vocab_langtail.get(sent))
                else:
                    mm = 0
            api_index.append((api_index_num, i))
            i = i + api_index_num

        array_api_data = np.zeros(len(api_data), dtype=np.long)
        for m in range(len(api_data)):
            array_api_data[m] = api_data[m]
        array_api_index = np.zeros(shape=(len(api_index), 2), dtype=np.int)
        for m in range(len(api_index)):
            array_api_index[m] = api_index[m]
        array_api_longtail_data = np.zeros(shape=(len(api_longtail), max_longtail_len), dtype=np.long)
        for m in range(len(api_longtail)):
            for n in range(max_longtail_len):
                if n < len(api_longtail[m]):
                    array_api_longtail_data[m][n] = api_longtail[m][n]

        self.api_data = array_api_data[:].astype(np.long)
        self.api_index = array_api_index
        self.api_longtail = array_api_longtail_data[:].astype(np.long)

        assert self.desc_index.shape[0] == self.api_index.shape[0], "no equal entities"

        self.data_len = self.desc_index.shape[0]
        print("{} entries".format(self.data_len))

    def list2array(self, L, max_len, dtype=np.long, pad_idx=0):
        '''  convert a list to an array or matrix  '''
        arr = np.zeros(max_len, dtype=dtype)+pad_idx
        for i, v in enumerate(L): arr[i] = v
        return arr

    def richLongtailArray(self, api_longtail, max_api_len, max_api_longtail_len):
        arr = np.zeros(shape=(max_api_len,max_api_longtail_len), dtype= np.long)
        for i in range(len(api_longtail)):
            arr[i+1] = api_longtail[i]
        return arr
    def __getitem__(self, index):
        pos, desc_len = self.desc_index[index][1],self.desc_index[index][0]
        desc_len = min(int(desc_len), self.max_desc_seq_len-2)
        desc = [SOS_ID] + self.desc_data[pos:pos +desc_len].tolist()+ [EOS_ID]
        desc = self.list2array(desc,self.max_desc_seq_len,np.int,PAD_ID)

        pos, api_len = self.api_index[index][1], self.api_index[index][0]
        api_len = min(int(api_len), self.max_api_seq_len-2)
        api = [SOS_ID] + self.api_data[pos:pos + api_len].tolist() + [EOS_ID]
        api = self.list2array(api, self.max_api_seq_len, np.int, PAD_ID)

        api_longtail = self.api_longtail[pos:pos+api_len]
        api_longtail = self.richLongtailArray(api_longtail,self.max_api_seq_len, self.max_longtail_len)


        return desc, api, api_longtail

    def __len__(self):
        return self.data_len


if __name__ =='__main__':
    valid_set = APIdataset(config.test_desc_file, config.test_api_file, config.vocab_desc_file, config.vocab_api_file, config.max_desc_length,
                           config.max_api_length, config.max_longtail_length, config.vocab_general_file, config.vocab_longtail_file)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                    batch_size=config.valid_batch_size,
                                                    shuffle=False,
                                                    num_workers=1)
    for desc,api,longtail in valid_data_loader:
        print(longtail)