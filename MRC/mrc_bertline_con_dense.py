'''
    bert for ner with tf2.0
    bert通过transformers加载
    通过任务Dense来区分LOC、PER、ORG
    收敛较快
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras.optimizers import Adam
from transformers.optimization_tf import AdamWeightDecay
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import TruncatedNormal
from OtherUtils import load_vocab

import numpy as np
import sys, os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=10, help='Epochs during training')
parser.add_argument('--lr', type=float, default=1.0e-5, help='Initial learing rate')
parser.add_argument('--label_dim', type=int, default=7, help='number of ner labels')
parser.add_argument('--check', type=str, default='model/mrc_bertlinear_con_dense',
                    help='The path where model shall be saved')
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
                         attention_mask=tf.cast(tf.greater(inputs, 0), tf.int32)
                         )[0]


class SplitSequence(Layer):
    def __init__(self, **kwargs):
        super(SplitSequence, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return x[:, 1:-1]


class MRC(Layer):
    def __init__(self, label_dim, **kwargs):
        super(MRC, self).__init__(**kwargs)
        self.label_dim = label_dim

        self.dense_ner_loc = Dense(label_dim,
                                   kernel_initializer=TruncatedNormal(stddev=0.02),
                                   dtype=tf.float32,
                                   name='nerloc')
        self.dense_ner_per = Dense(label_dim,
                                   kernel_initializer=TruncatedNormal(stddev=0.02),
                                   dtype=tf.float32,
                                   name='nerper')
        self.dense_ner_org = Dense(label_dim,
                                   kernel_initializer=TruncatedNormal(stddev=0.02),
                                   dtype=tf.float32,
                                   name='nerorg')

    def get_config(self):
        config = super(MRC, self).get_config().copy()
        config.update({
            'label_dim': self.label_dim,
        })

        return config

    def call(self, inputs, **kwargs):
        x, label, seqlen = inputs

        output_loc = self.dense_ner_loc(x)
        output_per = self.dense_ner_per(x)
        output_org = self.dense_ner_org(x)

        output_r = tf.concat([output_loc, output_per, output_org], axis=0)

        label1 = tf.where(tf.logical_or(tf.equal(label, 1), tf.equal(label, 2)), label, tf.zeros_like(label))
        label2 = tf.where(tf.logical_or(tf.equal(label, 3), tf.equal(label, 4)), label, tf.zeros_like(label))
        label3 = tf.where(tf.logical_or(tf.equal(label, 5), tf.equal(label, 6)), label, tf.zeros_like(label))

        label_r = tf.concat([label1, label2, label3], axis=0)
        seqlen_r = tf.tile(seqlen, [3])
        sequence_mask_r = tf.sequence_mask(seqlen_r, tf.reduce_max(seqlen_r))
        seqlen_sum_r = tf.cast(tf.reduce_sum(seqlen_r), tf.float32)

        loss = tf.keras.losses.sparse_categorical_crossentropy(label_r, output_r, from_logits=True)
        lossf = tf.zeros_like(loss)
        loss = tf.where(sequence_mask_r, loss, lossf)

        self.add_loss(tf.reduce_sum(loss) / seqlen_sum_r)

        predict_r = tf.argmax(output_r, axis=-1, output_type=tf.int32)

        predict_r = tf.stack(tf.split(predict_r, 3), axis=2)

        predict_s = tf.reduce_sum(tf.cast(tf.greater(predict_r, 0), tf.int32), axis=-1)
        predict = tf.reduce_sum(predict_r, axis=-1)
        predict = tf.where(tf.greater(predict_s, 1), tf.zeros_like(predict), predict)

        sequence_mask = tf.cast(tf.sequence_mask(seqlen, tf.reduce_max(seqlen)), tf.float32)
        seqlen_sum = tf.cast(tf.reduce_sum(seqlen), tf.float32)

        accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, label), tf.float32) * sequence_mask) / seqlen_sum

        self.add_metric(accuracy, name="acc")

        tp = tf.reduce_sum(tf.where(tf.logical_and(tf.greater(label, 0), tf.equal(predict, label)),
                                    tf.ones_like(predict, tf.float32), tf.zeros_like(predict, tf.float32)))
        tn = tf.reduce_sum(tf.where(tf.logical_and(tf.greater(label, 0), tf.not_equal(predict, label)),
                                    tf.ones_like(predict, tf.float32), tf.zeros_like(predict, tf.float32)))
        fp = tf.reduce_sum(tf.where(tf.logical_and(tf.equal(label, 0), tf.greater(predict, 0)),
                                    tf.ones_like(predict, tf.float32),
                                    tf.zeros_like(predict, tf.float32)) * sequence_mask)
        return predict, tp, tn, fp


class CheckCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation):
        super(CheckCallback, self).__init__()

        self.validation_data = validation

    def on_epoch_end(self, epoch, logs=None):
        tp, tn, fp = 0.0, 0.0, 0.0

        for data in self.validation_data:
            _, tp_, tn_, fp_ = self.model.predict(data)

            tp += tp_
            tn += tn_
            fp += fp_

        precision = tp / (tp + tn)
        recall = tp / (tp + fp)
        f1 = 2.0 * precision * recall / (precision + recall)

        sys.stdout.write('\n - precision: %.4f - recall: %.4f - f1: %.4f' % (precision, recall, f1))
        sys.stdout.flush()

        predict, _, _, _ = self.model.predict([sent, tf.ones_like(sent)[:, 1:-1]])
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
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def build_model(self):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)
        lab = Input(shape=[None], name='lab', dtype=tf.int32)

        seqlen = Mask(name="mask")(sen)

        sequence_output = BERT(name="bert")(sen)

        sequence_split = SplitSequence(name="splitsequence")(sequence_output)

        predict = MRC(label_dim=params.label_dim, name="mrc")(inputs=(sequence_split, lab, seqlen))

        model = Model(inputs=[sen, lab], outputs=predict)

        model.summary()

        return model

    def train(self):
        model = self.build_model()

        if params.mode == 'train1':
            model.load_weights(params.check + '/mrc.h5')

        optimizer = AdamWeightDecay(learning_rate=params.lr,
                                    weight_decay_rate=0.01,
                                    epsilon=1.0e-6,
                                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        model.compile(optimizer=optimizer)

        batch_data = batched_data(['data/TFRecordFiles/mrc_train.tfrecord'],
                                  single_example_parser,
                                  params.batch_size,
                                  padded_shapes={"sen": [-1], "lab": [-1]},
                                  buffer_size=100 * params.batch_size)

        dev_data = batched_data(['data/TFRecordFiles/mrc_dev.tfrecord'],
                                single_example_parser,
                                params.batch_size,
                                padded_shapes={"sen": [-1], "lab": [-1]},
                                buffer_size=100 * params.batch_size)

        callbacks = [
            EarlyStopping(monitor='val_acc', patience=3),
            ModelCheckpoint(filepath=params.check + '/mrc.h5',
                            monitor='val_acc',
                            save_best_only=True),
            CheckCallback(dev_data)
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
        model.load_weights(params.check + '/mrc.h5')

        predict, _, _, _ = model.predict([sent, tf.ones_like(sent)[:, 1:-1]])

        querycheck(predict)


if __name__ == '__main__':
    ner_dict = {'O': 0,
                'B-LOC': 1, 'I-LOC': 2,
                'B-PER': 3, 'I-PER': 4,
                'B-ORG': 5, 'I-ORG': 6, }
    ner_inverse_dict = {v: k for k, v in ner_dict.items()}
    char_dict = load_vocab("data/OriginalFiles/vocab.txt")

    if not os.path.exists(params.check):
        os.makedirs(params.check)

    user = USER()
    sentences = [
        '国正学长的文章与诗词，早就读过一些，很是喜欢。',
        '阳关在敦煌西南相距七十公里处，当中的一座沙山，好似巨佛横卧。',
        '北京丰盛中学校长赵铮：我校是所普通完中，无择优生源，新生入校时约有四分之一属于差生。',
        '在南山村，我们见到了张兰凤和李伟夫妻二人。',
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
    else:
        user.predict()
