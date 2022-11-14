'''
    bert + bigru + crf for medical ner with tf2.0
    bert通过transformers加载
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer, Dense, Bidirectional, GRU
from transformers.optimization_tf import AdamWeightDecay
from transformers import TFBertModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow_addons as tfa
from OtherUtils import load_vocab

import numpy as np

import sys, os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=10, help='Epochs during training')
parser.add_argument('--lr', type=float, default=1.0e-5, help='Initial learing rate')
parser.add_argument('--label_dim', type=int, default=11, help='number of ner labels')
parser.add_argument('--check', type=str, default='model/medical_bertbigrucrf', help='The path where model shall be saved')
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


class CRF(Layer):
    def __init__(self, label_dim, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.label_dim = label_dim

        self.dense_ner = Dense(label_dim,
                               kernel_initializer=TruncatedNormal(stddev=0.02),
                               dtype=tf.float32,
                               name='ner')

        self.transitions = self.add_weight(name='transitions',
                                           shape=[label_dim, label_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)

    def get_config(self):
        config = super(CRF, self).get_config().copy()
        config.update({
            'label_dim': self.label_dim,
        })

        return config

    def call(self, inputs, **kwargs):
        x, label, seqlen = inputs

        output = self.dense_ner(x)

        log_likelihood, _ = tfa.text.crf_log_likelihood(output, label, seqlen, self.transitions)
        loss = tf.reduce_mean(-log_likelihood)

        self.add_loss(loss)

        viterbi_sequence, _ = tfa.text.crf_decode(output, self.transitions, seqlen)

        accuracy = tf.reduce_sum(tf.cast(tf.equal(viterbi_sequence, label), tf.float32) * tf.cast(
            tf.sequence_mask(seqlen, tf.reduce_max(seqlen)), tf.float32))
        accuracy = accuracy / tf.reduce_sum(tf.cast(seqlen, tf.float32))

        self.add_metric(accuracy, name="crf_acc")

        return viterbi_sequence


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

        gruoutput = Bidirectional(GRU(128, return_sequences=True))(sequence_split)

        predict = CRF(label_dim=params.label_dim, name="crf")(inputs=(gruoutput, lab, seqlen))

        model = Model(inputs=[sen, lab], outputs=predict)

        model.summary()

        return model

    def train(self):
        model = self.build_model()

        if params.mode == 'train1':
            model.load_weights(params.check + '/medical.h5')

        optimizer = AdamWeightDecay(learning_rate=params.lr,
                                    weight_decay_rate=0.01,
                                    epsilon=1.0e-6,
                                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        model.compile(optimizer=optimizer)

        batch_data = batched_data(['data/TFRecordFiles/train.tfrecord'],
                                  single_example_parser,
                                  params.batch_size,
                                  padded_shapes={"sen": [-1], "lab": [-1]},
                                  buffer_size=100 * params.batch_size)
        dev_data = batched_data(['data/TFRecordFiles/dev.tfrecord'],
                                single_example_parser,
                                params.batch_size,
                                padded_shapes={"sen": [-1], "lab": [-1]},
                                buffer_size=100 * params.batch_size)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3),
            ModelCheckpoint(filepath=params.check + '/medical.h5',
                            monitor='val_loss',
                            save_best_only=True),
            CheckCallback()
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
        model.load_weights(params.check + '/medical.h5')

        predict = model.predict([sent, tf.ones_like(sent)[:, 1:-1]])

        querycheck(predict)


if __name__ == '__main__':
    ner_dict = {
        'O': 0,
        'TREATMENT-I': 1,
        'TREATMENT-B': 2,
        'BODY-B': 3,
        'BODY-I': 4,
        'SIGNS-I': 5,
        'SIGNS-B': 6,
        'CHECK-B': 7,
        'CHECK-I': 8,
        'DISEASE-I': 9,
        'DISEASE-B': 10
    }
    ner_inverse_dict = {v: k for k, v in ner_dict.items()}

    char_dict = load_vocab("data/OriginalFiles/vocab.txt")

    if not os.path.exists(params.check):
        os.makedirs(params.check)

    user = USER()

    sentences = [
        '1.患者老年女性，88岁；2.既往体健，否认药物过敏史。3.患者缘于5小时前不慎摔伤，伤及右髋部。伤后患者自感伤处疼痛，呼我院120接来我院，查左髋部X光片示：左侧粗隆间骨折。',
        '患者精神状况好，无发热，诉右髋部疼痛，饮食差，二便正常，查体：神清，各项生命体征平稳，心肺腹查体未见异常。',
        '女性，88岁，农民，双滦区应营子村人，主因右髋部摔伤后疼痛肿胀，活动受限5小时于2016-10-29；11：12入院。',
        '入院后完善各项检查，给予右下肢持续皮牵引，应用健骨药物治疗，患者略发热，查血常规：白细胞数12.18*10^9/L，中性粒细胞百分比92.00%。',
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
