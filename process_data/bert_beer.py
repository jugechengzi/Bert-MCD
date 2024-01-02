import gzip
import json
import os
import random
import csv
import numpy as np
import torch

from torch.utils.data import Dataset

import transformers as ppb




class BeerData_bert(Dataset):
    def __init__(self, tokenizer,data_dir, aspect, mode, balance=False, max_length=256, neg_thres=0.4, pos_thres=0.6,
                 stem='reviews.aspect{}.{}.txt.gz'):
        super().__init__()
        self.mode_to_name = {'train': 'train', 'dev': 'heldout'}
        self.mode = mode
        self.neg_thres = neg_thres
        self.pos_thres = pos_thres
        self.input_file = os.path.join(data_dir, stem.format(str(aspect), self.mode_to_name[mode]))
        # self.inputs = []
        # self.masks = []
        self.labels = []

        # self._convert_examples_to_arrays(
        #     self._create_examples(aspect, balance), max_length, word2idx)
        examples_text=self._create_examples(aspect, balance)
        self.token=tokenizer(examples_text,padding="max_length", max_length=max_length,truncation=True)
        self.inputs = torch.tensor(self.token['input_ids'])
        self.masks=torch.tensor(self.token['attention_mask'])
        self.special_masks=self.get_special_mask(self.masks)        #将cls和sep设为1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs, masks, labels,special_masks = self.inputs[item], self.masks[item], self.labels[item],self.special_masks[item]
        return inputs, masks, labels,special_masks

    def get_special_mask(self,attention_mask):
        #将cls和sep的mask设为1
        sents_len=torch.sum(attention_mask,dim=1)
        special_mask=torch.zeros_like(attention_mask)
        for i in range(len(special_mask)):
            special_mask[i][0]=1
            special_mask[i][sents_len[i]-1]=1
        return special_mask



    def _create_examples(self, aspect, balance=False):
        examples = []
        texts=[]
        with gzip.open(self.input_file, "rt") as f:
            lines = f.readlines()
            for (i, line) in enumerate(lines):
                labels, text = line.split('\t')
                labels = [float(v) for v in labels.split()]
                if labels[aspect] <= self.neg_thres:
                    label = 0
                elif labels[aspect] >= self.pos_thres:
                    label = 1
                else:
                    continue
                examples.append({'text': text, "label": label})
        print('Dataset: Beer Review')
        print('{} samples has {}'.format(self.mode_to_name[self.mode], len(examples)))

        pos_examples = [example for example in examples if example['label'] == 1]
        neg_examples = [example for example in examples if example['label'] == 0]

        print('%s data: %d positive examples, %d negative examples.' %
              (self.mode_to_name[self.mode], len(pos_examples), len(neg_examples)))

        if balance:

            random.seed(20226666)

            print('Make the Training dataset class balanced.')

            min_examples = min(len(pos_examples), len(neg_examples))

            if len(pos_examples) > min_examples:
                pos_examples = random.sample(pos_examples, min_examples)

            if len(neg_examples) > min_examples:
                neg_examples = random.sample(neg_examples, min_examples)

            assert (len(pos_examples) == len(neg_examples))
            examples = pos_examples + neg_examples
            print(
                'After balance training data: %d positive examples, %d negative examples.'
                % (len(pos_examples), len(neg_examples)))
        for e in examples:
            self.labels.append(e['label'])
            texts.append(e['text'].strip())
        return texts




class BeerAnnotation_bert(Dataset):

    def __init__(self, tokenizer,annotation_path, aspect, max_length=256, neg_thres=0.4, pos_thres=0.6):
        super().__init__()
        self.inputs = []
        self.masks = []
        self.labels = []
        self.rationales = []
        self.tokenizer=tokenizer
        self._create_example(annotation_path, aspect, max_length, pos_thres, neg_thres)
        self.special_masks = self.get_special_mask(self.masks)  # 将cls和sep设为1
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs, masks, labels, rationales,special_masks = self.inputs[item], self.masks[item], self.labels[item], self.rationales[
            item], self.special_masks[item]
        return inputs, masks, labels, rationales,special_masks

    def get_special_mask(self,attention_mask):
        #将cls和sep的mask设为1
        sents_len=torch.sum(attention_mask,dim=1)
        special_mask=torch.zeros_like(attention_mask)
        for i in range(len(special_mask)):
            special_mask[i][0]=1
            special_mask[i][sents_len[i]-1]=1
        return special_mask

    def _create_example(self, annotation_path, aspect, max_length, pos_thres, neg_thres):
        data = []
        masks = []
        labels = []
        rationales = []

        print('Dataset: Beer Review')

        with open(annotation_path, "rt", encoding='utf-8') as fin:
            for counter, line in enumerate(fin):
                item = json.loads(line)

                # obtain the data
                text_ = item["x"]     #['word1','word2','word3']
                y = item["y"][aspect]
                rationale = item[str(aspect)]       #[[start_id,end_id]]



                # check if the rationale is all zero
                if len(rationale) == 0:
                    # no rationale for this aspect
                    continue

                # process the label
                if float(y) >= pos_thres:
                    y = 1
                elif float(y) <= neg_thres:
                    y = 0
                else:
                    continue

                # tokenize
                tokenized_words_onesent =self.tokenizer(text_,add_special_tokens=False)['input_ids']
                ids_onesent=[]
                for w in tokenized_words_onesent:
                    ids_onesent+=w


                # process the text
                # input_ids = []
                # if len(text_) > max_length:
                #     text_ = text_[0:max_length]
                #
                # for word in text_:
                #     word = word.strip()
                #     try:
                #         input_ids.append(word2idx[word])
                #     except:
                #         # word is not exist in word2idx, use <unknown> token
                #         input_ids.append(0)

                # process mask
                # The mask has 1 for real word and 0 for padding tokens.
                input_mask = [1] * len(ids_onesent)

                # zero-pad up to the max_seq_length.
                # while len(input_ids) < max_length:
                #     input_ids.append(0)
                #     input_mask.append(0)

                # assert (len(input_ids) == max_length)
                # assert (len(input_mask) == max_length)

                # construct rationale
                word_level_rationale=[0]*len(tokenized_words_onesent)
                token_level_rationale_onesent = []
                for zs in rationale:
                    start = zs[0]
                    end = zs[1]
                    # if start >= max_length:
                    #     continue
                    # if end >= max_length:
                    #     end = max_length
                    for idx in range(start, end):
                        word_level_rationale[idx] = 1

                for idx,word_mask in enumerate(word_level_rationale):
                    token_level_rationale_onesent+=[word_mask]*len(tokenized_words_onesent[idx])

                #pad
                if len(ids_onesent)<= (max_length-2):
                    pad_ids_onesent=[101]+ids_onesent+[102]+[0]*(max_length-2-len(ids_onesent))
                    pad_token_level_rationale_onesent=[0]+token_level_rationale_onesent+[0]+[0]*(max_length-2-len(token_level_rationale_onesent))
                    pad_input_mask=[1]+input_mask+[1]+[0]*(max_length-2-len(input_mask))
                elif len(ids_onesent)>(max_length-2):
                    pad_ids_onesent = [101] + ids_onesent[:max_length-2] + [102]
                    pad_token_level_rationale_onesent = [0] + token_level_rationale_onesent[:max_length-2] + [0]
                    pad_input_mask=[1]+input_mask[:max_length-2]+[1]
                else:
                    print('wrong')


                data.append(pad_ids_onesent)
                labels.append(y)
                masks.append(pad_input_mask)
                rationales.append(pad_token_level_rationale_onesent)

        self.inputs = torch.from_numpy(np.array(data))
        self.labels = torch.from_numpy(np.array(labels))
        self.masks = torch.from_numpy(np.array(masks))
        self.rationales = torch.from_numpy(np.array(rationales))
        tot = self.labels.shape[0]
        print('annotation samples has {}'.format(tot))
        pos = torch.sum(self.labels)
        neg = tot - pos
        print('annotation data: %d positive examples, %d negative examples.' %
              (pos, neg))


class BeerData_bert_correlated(Dataset):
    def __init__(self, tokenizer,data_dir, aspect, mode, balance=False, max_length=256, neg_thres=0.4, pos_thres=0.6,
                 stem='reviews.260k.{}.txt.gz'):
        super().__init__()
        self.mode_to_name = {'train': 'train', 'dev': 'heldout'}
        self.mode = mode
        self.neg_thres = neg_thres
        self.pos_thres = pos_thres
        self.input_file = os.path.join(data_dir, stem.format(self.mode_to_name[mode]))
        # self.inputs = []
        # self.masks = []
        self.labels = []

        # self._convert_examples_to_arrays(
        #     self._create_examples(aspect, balance), max_length, word2idx)
        examples_text=self._create_examples(aspect, balance)
        self.token=tokenizer(examples_text,padding="max_length", max_length=max_length,truncation=True)
        self.inputs = torch.tensor(self.token['input_ids'])
        self.masks=torch.tensor(self.token['attention_mask'])
        self.special_masks=self.get_special_mask(self.masks)        #将cls和sep设为1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs, masks, labels,special_masks = self.inputs[item], self.masks[item], self.labels[item],self.special_masks[item]
        return inputs, masks, labels,special_masks

    def get_special_mask(self,attention_mask):
        #将cls和sep的mask设为1
        sents_len=torch.sum(attention_mask,dim=1)
        special_mask=torch.zeros_like(attention_mask)
        for i in range(len(special_mask)):
            special_mask[i][0]=1
            special_mask[i][sents_len[i]-1]=1
        return special_mask



    def _create_examples(self, aspect, balance=False):
        examples = []
        texts=[]
        with gzip.open(self.input_file, "rt") as f:
            lines = f.readlines()
            for (i, line) in enumerate(lines):
                labels, text = line.split('\t')
                labels = [float(v) for v in labels.split()]
                if labels[aspect] <= self.neg_thres:
                    label = 0
                elif labels[aspect] >= self.pos_thres:
                    label = 1
                else:
                    continue
                examples.append({'text': text, "label": label})
        print('Dataset: Beer Review')
        print('{} samples has {}'.format(self.mode_to_name[self.mode], len(examples)))

        pos_examples = [example for example in examples if example['label'] == 1]
        neg_examples = [example for example in examples if example['label'] == 0]

        print('%s data: %d positive examples, %d negative examples.' %
              (self.mode_to_name[self.mode], len(pos_examples), len(neg_examples)))

        if balance:

            random.seed(20226666)

            print('Make the Training dataset class balanced.')

            min_examples = min(len(pos_examples), len(neg_examples))

            if len(pos_examples) > min_examples:
                pos_examples = random.sample(pos_examples, min_examples)

            if len(neg_examples) > min_examples:
                neg_examples = random.sample(neg_examples, min_examples)

            assert (len(pos_examples) == len(neg_examples))
            examples = pos_examples + neg_examples
            print(
                'After balance training data: %d positive examples, %d negative examples.'
                % (len(pos_examples), len(neg_examples)))
        for e in examples:
            self.labels.append(e['label'])
            texts.append(e['text'].strip())
        return texts



class Hotel_bert(Dataset):
    def __init__(self,tokenizer, data_dir, aspect, mode,max_length=256, balance=False):
        super(Hotel_bert, self).__init__()
        self.num_to_aspect = {0: 'Location', 1: 'Service', 2: 'Cleanliness'}
        self.inputs = []
        self.masks = []
        self.labels = []
        self.path = os.path.join(data_dir, 'hotel_{}.{}'.format(self.num_to_aspect[aspect], mode))
        examples = self._create_examples(self._read_csv(self.path), mode, balance=balance)
        # self._convert_examples_to_arrays(examples, max_length, word2idx)
        self.token = tokenizer(examples, padding="max_length", max_length=max_length, truncation=True)
        self.inputs = torch.tensor(self.token['input_ids'])
        self.masks = torch.tensor(self.token['attention_mask'])
        self.special_masks = self.get_special_mask(self.masks)  # 将cls和sep设为1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs, masks, labels, special_masks = self.inputs[item], self.masks[item], self.labels[item], \
                                               self.special_masks[item]
        return inputs, masks, labels, special_masks

    def _read_csv(self, file_path, quotechar=None):
        """Reads a tab separated value file."""
        with open(file_path, "rt") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, mode, balance=False):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = int(line[1])
            text = line[2]
            examples.append({'text': text, "label": label})

        print('Dataset: Hotel Review')
        print('{} samples has {}'.format(mode, len(examples)))

        pos_examples = [example for example in examples if example['label'] == 1]
        neg_examples = [example for example in examples if example['label'] == 0]

        print('%s data: %d positive examples, %d negative examples.' %
              (mode, len(pos_examples), len(neg_examples)))

        if balance:

            random.seed(20226666)

            print('Make the Training dataset class balanced.')

            min_examples = min(len(pos_examples), len(neg_examples))

            if len(pos_examples) > min_examples:
                pos_examples = random.sample(pos_examples, min_examples)

            if len(neg_examples) > min_examples:
                neg_examples = random.sample(neg_examples, min_examples)

            assert (len(pos_examples) == len(neg_examples))
            examples = pos_examples + neg_examples
            print(
                'After balance training data: %d positive examples, %d negative examples.'
                % (len(pos_examples), len(neg_examples)))
        return examples

    def _convert_single_text(self, text, max_length, word2idx):
        """
        Converts a single text into a list of ids with mask.
        """
        input_ids = []

        text_ = text.strip().split(" ")

        if len(text_) > max_length:
            text_ = text_[0:max_length]

        for word in text_:
            word = word.strip()
            try:
                input_ids.append(word2idx[word])
            except:
                # if the word is not exist in word2idx, use <unknown> token
                input_ids.append(0)

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)

        # zero-pad up to the max_seq_length.
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length

        return input_ids, input_mask

    def _convert_examples_to_arrays(self, examples, max_length, word2idx):
        """
        Convert a set of train/dev examples numpy arrays.
        Outputs:
            data -- (num_examples, max_seq_length).
            masks -- (num_examples, max_seq_length).
            labels -- (num_examples, num_classes) in a one-hot format.
        """

        data = []
        labels = []
        masks = []
        for example in examples:
            input_ids, input_mask = self._convert_single_text(example["text"],
                                                              max_length, word2idx)

            data.append(input_ids)
            masks.append(input_mask)
            labels.append(example["label"])

        self.inputs = torch.from_numpy(np.array(data))
        self.masks = torch.from_numpy(np.array(masks))
        self.labels = torch.from_numpy(np.array(labels))

    def get_special_mask(self,attention_mask):
        #将cls和sep的mask设为1
        sents_len=torch.sum(attention_mask,dim=1)
        special_mask=torch.zeros_like(attention_mask)
        for i in range(len(special_mask)):
            special_mask[i][0]=1
            special_mask[i][sents_len[i]-1]=1
        return special_mask

class HotelAnnotation_bert(Dataset):

    def __init__(self,tokenizer, data_dir, aspect, max_length=256):
        super(HotelAnnotation_bert, self).__init__()
        self.num_to_aspect = {0: 'Location', 1: 'Service', 2: 'Cleanliness'}
        self.input_ids = []
        self.masks = []
        self.labels = []
        self.rationales = []
        self.tokenizer = tokenizer
        self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'hotel_{}.train'.format(self.num_to_aspect[aspect]))),
            word2idx,
            max_length)

    def __getitem__(self, i):
        return self.input_ids[i], self.masks[i], self.labels[i], self.rationales[i]

    def __len__(self):
        return len(self.labels)

    def _read_tsv(self, annotation_path, quotechar=None):
        """Reads a tab separated value file."""
        with open(annotation_path, "rt") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, word2idx, max_length):
        data = []
        labels = []
        masks = []
        rationales = []

        print('Dataset: Hotel Review')

        for i, line in enumerate(lines):
            if i == 0:
                continue
            text_ = line[2].split(" ")
            label_ = int(line[1])
            rationale = [int(x) for x in line[3].split(" ")]
            # process the text
            input_ids = []
            if len(text_) > max_length:
                text_ = text_[0:max_length]

            # for word in text_:
            #     word = word.strip()
            #     try:
            #         input_ids.append(word2idx[word])
            #     except:
            #         # word is not exist in word2idx, use <unknown> token
            #         input_ids.append(0)
            # # process mask
            # # The mask has 1 for real word and 0 for padding tokens.
            # input_mask = [1] * len(input_ids)
            #
            # # zero-pad up to the max_seq_length.
            # while len(input_ids) < max_length:
            #     input_ids.append(0)
            #     input_mask.append(0)
            #
            # assert (len(input_ids) == max_length)
            # assert (len(input_mask) == max_length)
            #
            # # construct rationale
            # binary_rationale = [0] * len(input_ids)
            # for k in range(len(binary_rationale)):
            #     # print(k)
            #     if k < len(rationale):
            #         binary_rationale[k] = rationale[k]
            #
            # data.append(input_ids)
            # labels.append(label_)
            # masks.append(input_mask)
            # rationales.append(binary_rationale)

                # tokenize
                tokenized_words_onesent = self.tokenizer(text_, add_special_tokens=False)['input_ids']
                ids_onesent = []
                for w in tokenized_words_onesent:
                    ids_onesent += w

                # process the text
                # input_ids = []
                # if len(text_) > max_length:
                #     text_ = text_[0:max_length]
                #
                # for word in text_:
                #     word = word.strip()
                #     try:
                #         input_ids.append(word2idx[word])
                #     except:
                #         # word is not exist in word2idx, use <unknown> token
                #         input_ids.append(0)

                # process mask
                # The mask has 1 for real word and 0 for padding tokens.
                input_mask = [1] * len(ids_onesent)

                # zero-pad up to the max_seq_length.
                # while len(input_ids) < max_length:
                #     input_ids.append(0)
                #     input_mask.append(0)

                # assert (len(input_ids) == max_length)
                # assert (len(input_mask) == max_length)

                # construct rationale
                # word_level_rationale = [0] * len(tokenized_words_onesent)
                # token_level_rationale_onesent = []
                # for zs in rationale:
                #     start = zs[0]
                #     end = zs[1]
                #     for idx in range(start, end):
                #         word_level_rationale[idx] = 1

                word_level_rationale =rationale
                token_level_rationale_onesent = []
                for idx, word_mask in enumerate(word_level_rationale):
                    token_level_rationale_onesent += [word_mask] * len(tokenized_words_onesent[idx])

                # pad
                if len(ids_onesent) <= (max_length - 2):
                    pad_ids_onesent = [101] + ids_onesent + [102] + [0] * (max_length - 2 - len(ids_onesent))
                    pad_token_level_rationale_onesent = [0] + token_level_rationale_onesent + [0] + [0] * (
                                max_length - 2 - len(token_level_rationale_onesent))
                    pad_input_mask = [1] + input_mask + [1] + [0] * (max_length - 2 - len(input_mask))
                elif len(ids_onesent) > (max_length - 2):
                    pad_ids_onesent = [101] + ids_onesent[:max_length - 2] + [102]
                    pad_token_level_rationale_onesent = [0] + token_level_rationale_onesent[:max_length - 2] + [0]
                    pad_input_mask = [1] + input_mask[:max_length - 2] + [1]
                else:
                    print('wrong')

                data.append(pad_ids_onesent)
                labels.append(label_)
                masks.append(pad_input_mask)
                rationales.append(pad_token_level_rationale_onesent)

        self.input_ids = torch.from_numpy(np.array(data))
        self.masks = torch.from_numpy(np.array(masks))
        self.labels = torch.from_numpy(np.array(labels))
        self.rationales = torch.from_numpy(np.array(rationales))
        tot = self.labels.shape[0]
        print('annotation samples has {}'.format(tot))
        pos = torch.sum(self.labels)
        neg = tot - pos
        print('annotation data: %d positive examples, %d negative examples.' %
              (pos, neg))





# beer_data=BeerData_bert('../data/beer',0,'train')
# beer_ano=BeerAnnotation_bert("../data/beer/annotations.json",0)