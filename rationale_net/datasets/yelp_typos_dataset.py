import torch
import torch.utils.data as data
import gzip
import rationale_net.utils.dataset as utils
import tqdm
import numpy as np
import pickle
import json
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset


SMALL_TRAIN_SIZE = 800

@RegisterDataset('yelp_typos')
class YelpTyposDataset(AbstractDataset):

    def __init__(self, args, word_to_indx, mode, max_length=250, stem='raw_data/yelp/yelp.disc'):
        self.args= args
        self.name = mode
        self.objective = args.objective
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.name_to_key = {'train':'train', 'dev':'heldout', 'test':'heldout'}
        self.class_balance = {}
        with gzip.open(stem+'.'+self.name_to_key[self.name]+'.txt.gz') as gfile:
            lines = gfile.readlines()
            lines = list(zip(range(len(lines)), lines))
            if args.debug_mode:
                lines = lines[:SMALL_TRAIN_SIZE]
            elif self.name == 'dev':
                lines = lines[:5000]
            elif self.name == 'test':
                lines = lines[5000:10000]
            elif self.name == 'train':
                lines = lines[0:20000]

            for indx, line in tqdm.tqdm(enumerate(lines)):
                uid, line_content = line
                sample = self.processLine(line_content, indx)

                if not sample['y'] in self.class_balance:
                    self.class_balance[ sample['y'] ] = 0
                self.class_balance[ sample['y'] ] += 1
                sample['uid'] = uid
                self.dataset.append(sample)
            gfile.close()
        print ("Class balance", self.class_balance)

        if args.class_balance:
            raise NotImplementedError("Yelp typo dataset doesn't support balanced sampling!")

    ## Convert one line from yelp dataset to {Text, Tensor, Labels}
    def processLine(self, line, i):
        if isinstance(line, bytes):
            line = line.decode()
        label = line.split('\t')[0]
        if self.objective == 'mse':
            label = float(label)
            self.args.num_class = 1
        else:
            label = int(label)
            self.args.num_class = 2
        token_seq = line.split('\t')[1].split()[:self.max_length]
        text = " ".join(token_seq)
        x =  utils.get_indices_tensor(token_seq, self.word_to_indx, self.max_length)
        sample = {'text':text,'x':x, 'y':label, 'i':i}
        return sample
