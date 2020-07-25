from pathlib import Path

import cv2
import editdistance
import tensorflow as tf

import sample_preprocessor
from model import MyModel


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../data/charList.txt'
    fnTrain = '../data/'
    fnInfer = '../data/test.png'
    fnWeight = '../data/weight'


def decoder_output_to_text(ctc_output, batch_size):
    char_list = open(FilePaths.fnCharList).read()
    encoded_label_strs = [[] for i in range(batch_size)]
    # ctc returns tuple, first element is SparseTensor
    decoded = ctc_output[0][0]

    # go over all indices and save mapping: batch -> values
    idxDict = {b: [] for b in range(batch_size)}
    for (idx, idx2d) in enumerate(decoded.indices):
        label = decoded.values[idx]
        batch_element = idx2d[0]  # index according to [b,t]
        encoded_label_strs[batch_element].append(label)

    # map labels to chars for all batch elements
    return [str().join([char_list[c] for c in labelStr]) for labelStr in encoded_label_strs]


def to_sparse(texts):
    "put ground truth texts into sparse tensor for ctc_loss"
    indices = []
    values = []
    shape = [len(texts), 0]  # last entry must be max(labelList[i])
    char_list = open(FilePaths.fnCharList).read()
    # go over all texts
    for (batchElement, text) in enumerate(texts):
        # convert to string of label (i.e. class-ids)
        label_str = [char_list.index(c) for c in text]
        # sparse tensor must have size of max. label-string
        if len(label_str) > shape[1]:
            shape[1] = len(label_str)
        # put each label into sparse tensor
        for (i, label) in enumerate(label_str):
            indices.append([batchElement, i])
            values.append(label)

    return indices, values, shape


class ModelHelper:
    def __init__(self):
        self.model = MyModel()

        # change the learning rate  rate to start training from scratch
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001)

    @tf.function
    def train_step(self, images, labels, seq_len):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = tf.math.reduce_mean(tf.nn.ctc_loss(labels=labels,
                                                      logits=predictions,
                                                      logit_length=seq_len,
                                                      label_length=None,
                                                      blank_index=-1))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def train(self, loader):
        # load previously trained weights
        self.load_weights(FilePaths.fnWeight)
        # define the number of epochs to run
        epochs = 1
        for epoch in range(epochs):
            loader.train_set()
            while loader.has_next():
                iter_info = loader.get_iterator_info()
                batch = loader.get_next()
                labels = to_sparse(batch.gtTexts)
                labels = tf.SparseTensor(labels[0], labels[1], labels[2])
                sequence_lengths = tf.cast(tf.fill([MyModel.batch_size], MyModel.max_text_len), dtype=tf.int32)
                loss = self.train_step(tf.cast(batch.imgs, dtype=tf.float32), tf.cast(labels, dtype=tf.int32),
                                       sequence_lengths)
                print('Epoch:', str(epoch + 1), 'Batch:', iter_info[0], '/', iter_info[1], 'Loss:', loss)
                # save weights after each epoch
                self.model.save_weights(FilePaths.fnWeight)
            self.validate(loader)

    def validate(self, loader):
        self.load_weights(FilePaths.fnWeight)
        # print('Validate NN')
        loader.validation_set()
        num_char_err = 0
        num_char_total = 0
        num_word_ok = 0
        num_word_total = 0
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            print('Batch:', iter_info[0], '/', iter_info[1])
            batch = loader.get_next()
            pred = self.model(batch.imgs)
            sequence_lengths = tf.cast(tf.fill([MyModel.batch_size], MyModel.max_text_len), dtype=tf.int32)
            pred = tf.nn.ctc_beam_search_decoder(pred, sequence_lengths, beam_width=50)
            recognized = decoder_output_to_text(pred, MyModel.batch_size)

            print('Ground truth -> Recognized')
            for i in range(len(recognized)):
                num_word_ok += 1 if batch.gtTexts[i] == recognized[i] else 0
                num_word_total += 1
                dist = editdistance.eval(recognized[i], batch.gtTexts[i])
                num_char_err += dist
                num_char_total += len(batch.gtTexts[i])
                print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                      '"' + recognized[i] + '"')

        # print validation result
        char_error_rate = num_char_err / num_char_total
        word_accuracy = num_word_ok / num_word_total
        print('Character error rate: %f%%. Word accuracy: %f%%.' % (char_error_rate * 100.0, word_accuracy * 100.0))
        return char_error_rate

    def infer(self):
        img = cv2.imread(FilePaths.fnInfer, cv2.IMREAD_GRAYSCALE)
        img = sample_preprocessor.pre_process(img, (MyModel.img_width, MyModel.img_height))
        self.load_weights(FilePaths.fnWeight)
        pred = self.model(tf.expand_dims(img, axis=0))
        sequence_lengths = tf.cast(tf.fill([1], MyModel.max_text_len), dtype=tf.int32)
        pred = tf.nn.ctc_beam_search_decoder(pred, sequence_lengths, beam_width=50)
        # tf.print(pred[0][0])
        pred = decoder_output_to_text(pred, 1)
        print(pred)

    def load_weights(self, file_path):
        path = Path(file_path)
        if path.is_file():
            self.model.load_weights(filepath=file_path)

        print('Note >  Previous Weights not available')
