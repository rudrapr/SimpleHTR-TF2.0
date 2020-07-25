from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from sample_preprocessor import pre_process


class Sample:
    "sample from the dataset"

    def __init__(self, gt_text, file_path):
        self.gtText = gt_text
        self.filePath = file_path


class Batch:
    "batch containing images and ground truth texts"

    def __init__(self, gt_texts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gt_texts


class DataLoader:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

    def __init__(self, file_path, batch_size, img_size, max_text_len):
        "loader for dataset at given location, preprocess images and text according to parameters"

        assert file_path[-1] == '/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.batch_size = batch_size
        self.imgSize = img_size
        self.samples = []

        f = open(file_path + 'words.txt')
        chars = set()
        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ')
            assert len(line_split) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            file_name_split = line_split[0].split('-')
            file_name = file_path + 'words/' + file_name_split[0] + '/' + file_name_split[0] + '-' + file_name_split[
                1] + '/' + line_split[0] + '.png'

            # GT text are columns starting at 9
            gt_text = self.truncate_label(' '.join(line_split[8:]), max_text_len)
            chars = chars.union(set(list(gt_text)))

            # check if image is not empty
            if not os.path.getsize(file_name):
                bad_samples.append(line_split[0] + '.png')
                continue

            # put sample into list
            self.samples.append(Sample(gt_text, file_name))

        # some images in the IAM dataset are known to be damaged, don't show warning for them
        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)

        # split into training and validation set: 95% - 5%
        split_idx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:split_idx]
        self.validationSamples = self.samples[split_idx:]

        # put words into lists
        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        # number of randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 25000  # --------------------------------------previously 25000

        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.charList = sorted(list(chars))

    def truncate_label(self, text, max_text_len):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def train_set(self):
        "switch to randomly chosen subset of training set"
        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

    def validation_set(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples

    def get_iterator_info(self):
        "current batch index and overall number of batches"
        return self.currIdx // self.batch_size + 1, len(self.samples) // self.batch_size

    def has_next(self):
        "iterator"
        return self.currIdx + self.batch_size <= len(self.samples)

    def get_next(self):
        "iterator"
        batch_range = range(self.currIdx, self.currIdx + self.batch_size)
        gt_texts = [self.samples[i].gtText for i in batch_range]
        imgs = [
            pre_process(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation)
            for i in batch_range]
        self.currIdx += self.batch_size
        return Batch(gt_texts, imgs)
