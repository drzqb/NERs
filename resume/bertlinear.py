'''
    bert for resume ner with tf2.0
    bert通过transformers加载
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer, Dense
from transformers.optimization_tf import AdamWeightDecay
from transformers import TFBertModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import TruncatedNormal
from OtherUtils import load_vocab

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=20, help='Epochs during training')
parser.add_argument('--lr', type=float, default=1.0e-5, help='Initial learing rate')
parser.add_argument('--label_dim', type=int, default=17, help='number of ner labels')
parser.add_argument('--check', type=str, default='model/resume_bertlinear', help='The path where model shall be saved')
parser.add_argument('--mode', type=str, default='train0', help='The mode of train or predict as follows: '
                                                               'train0: begin to train or retrain'
                                                               'tran1:continue to train'
                                                               'predict: predict')
params = parser.parse_args()


def single_example_parser(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
        'lab': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    lab = sequence_parsed['lab']
    return {"sen": sen, "lab": lab}


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


class Mask(Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, sen, **kwargs):
        sequencemask = tf.cast(tf.greater(sen, 0), tf.int32)

        return tf.reduce_sum(sequencemask, axis=-1) - 2


class BERT(Layer):
    def __init__(self, **kwargs):
        super(BERT, self).__init__(**kwargs)

        self.bert = TFBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def call(self, inputs, **kwargs):
        return self.bert(input_ids=inputs,
                         token_type_ids=tf.zeros_like(inputs),
                         attention_mask=tf.cast(tf.greater(inputs, 0), tf.int32))[0]


class SplitSequence(Layer):
    def __init__(self, **kwargs):
        super(SplitSequence, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return x[:, 1:-1]


class Linear(Layer):
    def __init__(self, label_dim, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.label_dim = label_dim

        self.dense_ner = Dense(label_dim,
                               kernel_initializer=TruncatedNormal(stddev=0.02),
                               dtype=tf.float32,
                               name='ner')

    def get_config(self):
        config = super(Linear, self).get_config().copy()
        config.update({
            'label_dim': self.label_dim,
        })

        return config

    def call(self, inputs, **kwargs):
        x, label, seqlen = inputs

        sequence_mask = tf.sequence_mask(seqlen, tf.reduce_max(seqlen))

        seqlen_sum = tf.cast(tf.reduce_sum(seqlen), tf.float32)

        output = self.dense_ner(x)

        loss = tf.keras.losses.sparse_categorical_crossentropy(label, output, from_logits=True)
        lossf = tf.zeros_like(loss)
        loss = tf.where(sequence_mask, loss, lossf)

        self.add_loss(tf.reduce_sum(loss) / seqlen_sum)

        predict = tf.argmax(output, axis=-1, output_type=tf.int32)

        accuracy = tf.cast(tf.equal(predict, label), tf.float32)
        accuracyf = tf.zeros_like(accuracy)
        accuracy = tf.where(sequence_mask, accuracy, accuracyf)

        self.add_metric(tf.reduce_sum(accuracy) / seqlen_sum, name="acc")

        return predict


class CheckCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        predict = self.model.predict([sent, tf.ones_like(sent)[:, 1:-1]])
        querycheck(predict)


def querycheck(predict):
    sys.stdout.write('\n')
    sys.stdout.flush()
    for i, pre in enumerate(predict):
        for j in range(leng[i]):
            sys.stdout.write(sentences[i][j] + '\t' + ner_inverse_dict[pre[j]] + '\n')
        sys.stdout.write('\n\n')
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


class USER:
    def build_model(self):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)
        lab = Input(shape=[None], name='lab', dtype=tf.int32)

        seqlen = Mask(name="mask")(sen)

        sequence_output = BERT(name="bert")(sen)

        sequence_split = SplitSequence(name="splitsequence")(sequence_output)

        predict = Linear(label_dim=params.label_dim, name="linear")(inputs=(sequence_split, lab, seqlen))

        model = Model(inputs=[sen, lab], outputs=predict)

        model.summary()

        return model

    def train(self):
        model = self.build_model()

        if params.mode == 'train1':
            model.load_weights(params.check + '/resume.h5')

        optimizer = AdamWeightDecay(learning_rate=params.lr,
                                    weight_decay_rate=0.01,
                                    epsilon=1.0e-6,
                                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        model.compile(optimizer=optimizer)

        batch_data = batched_data(['data/TFRecordFiles/demo_train.tfrecord'],
                                  single_example_parser,
                                  params.batch_size,
                                  padded_shapes={"sen": [-1], "lab": [-1]},
                                  buffer_size=100 * params.batch_size)
        dev_data = batched_data(['data/TFRecordFiles/demo_dev.tfrecord'],
                                single_example_parser,
                                params.batch_size,
                                padded_shapes={"sen": [-1], "lab": [-1]},
                                buffer_size=100 * params.batch_size)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3),
            ModelCheckpoint(filepath=params.check + '/resume.h5',
                            monitor='val_loss',
                            save_best_only=True),
            # CheckCallback()
        ]

        history = model.fit(batch_data,
                            epochs=params.epochs,
                            validation_data=dev_data,
                            callbacks=callbacks
                            )

        with open(params.check + "/history.txt", "w", encoding="utf-8") as fw:
            fw.write(str(history.history))

    def predict(self):
        model = self.build_model()
        model.load_weights(params.check + '/resume.h5')

        predict = model.predict([sent, tf.ones_like(sent)[:, 1:-1]])

        querycheck(predict)

    def plot(self):
        with open("model/resume_bertlinear/history.txt", "r", encoding="utf-8") as fr:
            history_bertlinear = fr.read()
            history_bertlinear = eval(history_bertlinear)

        plt.subplot(221)
        plt.plot(history_bertlinear["loss"])
        plt.subplot(222)
        plt.plot(history_bertlinear["acc"])
        plt.subplot(223)
        plt.plot(history_bertlinear["val_loss"])
        plt.subplot(224)
        plt.plot(history_bertlinear["val_acc"])

        plt.show()


if __name__ == '__main__':
    ner_dict = {
        'O': 0,
        'B-GPE': 1,
        'M-GPE': 2,
        'E-GPE': 3,
        'B-LOC': 4,
        'M-LOC': 5,
        'E-LOC': 6,
        'B-ORG': 7,
        'M-ORG': 8,
        'E-ORG': 9,
        'B-PER': 10,
        'M-PER': 11,
        'E-PER': 12,
        'S-GPE': 13,
        'S-LOC': 14,
        'S-ORG': 15,
        'S-PER': 16
    }
    ner_inverse_dict = {v: k for k, v in ner_dict.items()}

    char_dict = load_vocab("data/OriginalFiles/vocab.txt")

    if not os.path.exists(params.check):
        os.makedirs(params.check)

    user = USER()

    sentences = [
        '不就因为景气看坏，开花店、专卖切花的老板王年丰近来唉声连连，平日常做影视与媒体业者生意的他，直言娱乐界的公关费少掉一半，他的花儿生意也黯淡许多。',
        '目前故宫收藏的近代书画包括有张大千、齐白石、于右任、台静农等人的作品。',
        '堪称文坛祭酒的隐地，本名柯青华（熟识的朋友称他为老K），浙江永嘉人，一九三七年生于上海，七岁时被父母送到江苏昆山一位顾姓人家寄养，在农村种稻插秧；',
        '十岁时由父亲自上海带来台湾，从基隆码头上岸，一路走、一路哭，哭着过仁爱路，哭着进宁波西街的家，直到遇见一只木头方盒子里，竟有人唱着京戏，哭声才歇在方盒子上，傻不愣登问道：「这盒里有人吗？」就在这一次童稚的惊讶中，正式开演了他在台湾的人生大戏。',
    ]

    m_samples = len(sentences)

    sent = []
    leng = []
    for sentence in sentences:
        sentence = sentence.lower()
        leng.append(len(sentence))

        sen2id = [char_dict['[CLS]']] + [
            char_dict[word] if word in char_dict.keys() else char_dict['[UNK]']
            for word in sentence.lower()] + [char_dict['[SEP]']]
        sent.append(sen2id)

    max_len = np.max(leng)
    for i in range(m_samples):
        if leng[i] < max_len:
            pad = [char_dict['[PAD]']] * (max_len - leng[i])
            sent[i] += pad

    sent = tf.constant(sent)

    if params.mode.startswith('train'):
        user.train()
    elif params.mode == "predict":
        user.predict()
    elif params.mode == "plot":
        user.plot()
