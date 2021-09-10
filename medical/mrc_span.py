'''
    bert for ner with tf2.0
    bert通过transformers加载
    通过任务Dense来区分 TREATMENT、BODY、SIGNS、CHECK、DISEASE
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras.optimizers import Adam
from transformers.optimization_tf import AdamWeightDecay
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import BinaryCrossentropy as bce
from OtherUtils import load_vocab

import numpy as np
import sys, os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=10, help='Epochs during training')
parser.add_argument('--lr', type=float, default=1.0e-5, help='Initial learing rate')
parser.add_argument('--eps', type=float, default=1.0e-6, help='epsilon')
parser.add_argument('--label_num', type=int, default=5, help='number of ner labels')
parser.add_argument('--check', type=str, default='model/mrc_span',
                    help='The path where model shall be saved')
parser.add_argument('--mode', type=str, default='train0', help='The mode of train or predict as follows: '
                                                               'train0: begin to train or retrain'
                                                               'tran1:continue to train'
                                                               'predict: predict')
params = parser.parse_args()


def single_example_parser(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
        'start': tf.io.FixedLenSequenceFeature([], tf.int64),
        'end': tf.io.FixedLenSequenceFeature([], tf.int64),
        'span': tf.io.FixedLenSequenceFeature([], tf.int64),
        'val': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    seqlen = tf.shape(sen)[0] - 2

    start = tf.reshape(sequence_parsed['start'], [params.label_num, seqlen])

    end = tf.reshape(sequence_parsed['end'], [params.label_num, seqlen])

    span = tf.reshape(sequence_parsed['span'], [params.label_num, seqlen, seqlen])

    val = tf.reshape(sequence_parsed['val'], [params.label_num, seqlen, seqlen])

    return {"sen": sen,
            "start": start,
            "end": end,
            "span": span,
            "val": val,
            }


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def focal_loss(y_true, y_pred, gamma=2.0):
    """
    Focal Loss 针对样本不均衡
    :param y_true: 样本标签
    :param y_pred: 预测值（sigmoid）
    :return: focal loss
    """

    alpha = 0.5
    loss = tf.where(tf.equal(y_true, 1),
                    -alpha * (1.0 - y_pred) ** gamma * tf.math.log(y_pred),
                    -(1.0 - alpha) * y_pred ** gamma * tf.math.log(1.0 - y_pred))

    return tf.squeeze(loss, axis=-1)


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
    def __init__(self, **kwargs):
        super(MRC, self).__init__(**kwargs)

        self.dense_startend = Dense(2 * params.label_num,
                                    kernel_initializer=TruncatedNormal(stddev=0.02),
                                    dtype=tf.float32,
                                    activation="sigmoid",
                                    name='startend')

        self.dense_span = Dense(params.label_num,
                                kernel_initializer=TruncatedNormal(stddev=0.02),
                                dtype=tf.float32,
                                activation="sigmoid",
                                name='span')

    def call(self, inputs, **kwargs):
        # x: B*N*768,start,end: B*5*N,span,val: B*5*N*N seqlen: B
        x, start, end, span, val, seqlen = inputs

        # B*N*10
        startend_output = self.dense_startend(x)

        # B*10*N*1
        startend_output = tf.expand_dims(tf.transpose(startend_output, [0, 2, 1]), axis=-1)

        # B*10*N
        startend = tf.concat([start, end], axis=1)

        # B*10*N*1
        startend = tf.expand_dims(startend, axis=-1)

        # B*10*N
        # startend_loss = bce(reduction=tf.keras.losses.Reduction.NONE)(startend, startend_output)
        startend_loss = focal_loss(startend, startend_output)

        # B*N
        sequencemask = tf.sequence_mask(seqlen, tf.reduce_max(seqlen))

        # B*10*N
        sequencemask = tf.cast(tf.tile(tf.expand_dims(sequencemask, axis=1), [1, 2 * params.label_num, 1]), tf.float32)

        # B*10*N
        startend_loss *= sequencemask

        startend_loss = tf.reduce_sum(startend_loss) / tf.reduce_sum(sequencemask)

        self.add_loss(startend_loss)

        N = tf.shape(x)[1]

        # B*N*N*768
        startx = tf.tile(tf.expand_dims(x, 2), [1, 1, N, 1])
        endx = tf.tile(tf.expand_dims(x, 1), [1, N, 1, 1])

        # B*N*N*(768*2)
        spanx = tf.concat([startx, endx], axis=-1)

        # B*N*N*5
        span_output = self.dense_span(spanx)

        # B*5*N*N
        span_output = tf.transpose(span_output, [0, 3, 1, 2])

        # B*5*N*N
        span_predict = tf.cast(tf.greater(span_output, 0.5), tf.int32)

        # B*5*N*N
        accuracy = tf.cast(tf.equal(span_predict, span), tf.float32)

        # B*5*N*N
        valf = tf.cast(val, tf.float32)

        accuracy *= valf

        valsum = tf.reduce_sum(valf) + params.eps

        accuracysum = tf.reduce_sum(accuracy)

        accuracy = accuracysum / valsum

        self.add_metric(accuracy, name="acc")

        # 是实体，预测是实体
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(span, 1), tf.equal(span_predict, 1)), tf.float32))

        # 是实体，预测不是实体
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(span, 1), tf.not_equal(span_predict, 1)), tf.float32))

        # 不是有效实体，预测是实体
        fp = tf.reduce_sum(tf.cast(
            tf.logical_and(tf.logical_and(tf.equal(span, 0), tf.equal(val, 1)), tf.equal(span_predict, 1)),
            tf.float32))

        # B*5*N*N*1
        span_outputlogits = tf.expand_dims(span_output, axis=-1)

        # B*5*N*N*1
        span = tf.expand_dims(span, axis=-1)

        # B*5*N*N
        span_loss = focal_loss(span, span_outputlogits)

        # B*5*N*N
        span_loss *= valf

        span_loss = tf.reduce_sum(span_loss) / valsum

        self.add_loss(span_loss)

        return span_predict, tp, tn, fp


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

        precision = tp / (tp + fp + params.eps)
        recall = tp / (tp + tn + params.eps)
        f1 = 2.0 * precision * recall / (precision + recall + params.eps)

        sys.stdout.write('\nprecision: %.4f recall: %.4f f1: %.4f\n\n' % (precision, recall, f1))
        sys.stdout.flush()

        # predict, _, _, _ = self.model.predict([sent, tf.ones_like(sent)[:, 1:-1]])
        # querycheck(predict)


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

        start = Input(shape=[params.label_num, None], name='start', dtype=tf.int32)

        end = Input(shape=[params.label_num, None], name='end', dtype=tf.int32)

        span = Input(shape=[params.label_num, None, None], name='span', dtype=tf.int32)

        val = Input(shape=[params.label_num, None, None], name='val', dtype=tf.int32)

        seqlen = Mask(name="mask")(sen)

        sequence_output = BERT(name="bert")(sen)

        sequence_split = SplitSequence(name="splitsequence")(sequence_output)

        predict = MRC(name="mrc")(inputs=(sequence_split, start, end, span, val, seqlen))

        model = Model(inputs=[sen, start, end, span, val], outputs=predict)

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

        batch_data = batched_data(['data/TFRecordFiles/train_span.tfrecord'],
                                  single_example_parser,
                                  params.batch_size,
                                  padded_shapes={"sen": [-1],
                                                 "start": [params.label_num, -1],
                                                 "end": [params.label_num, -1],
                                                 "span": [params.label_num, -1, -1],
                                                 "val": [params.label_num, -1, -1],
                                                 },
                                  buffer_size=100 * params.batch_size)

        dev_data = batched_data(['data/TFRecordFiles/dev_span.tfrecord'],
                                single_example_parser,
                                params.batch_size,
                                padded_shapes={"sen": [-1],
                                               "start": [params.label_num, -1],
                                               "end": [params.label_num, -1],
                                               "span": [params.label_num, -1, -1],
                                               "val": [params.label_num, -1, -1],
                                               },
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

    def test(self):
        model = self.build_model()
        model.load_weights(params.check + '/mrc.h5')

        dev_data = batched_data(['data/TFRecordFiles/dev_span.tfrecord'],
                                single_example_parser,
                                1,
                                padded_shapes={"sen": [-1],
                                               "start": [params.label_num, -1],
                                               "end": [params.label_num, -1],
                                               "span": [params.label_num, -1, -1],
                                               "val": [params.label_num, -1, -1],
                                               },
                                buffer_size=100 * params.batch_size)

        tp, tn, fp = 0.0, 0.0, 0.0

        fw = open(params.check + "/log.txt", "w", encoding="utf-8")

        for data in dev_data:
            sen = data["sen"][0][1:-1]
            start = data["start"][0]
            end = data["end"][0]
            span = data["span"][0]
            val = data["val"][0]

            fw.write("句子\n")
            fw.write("".join([char_inverse_dict[s] for s in sen.numpy()]) + "\n\n")

            fw.write("start\n")
            start = start.numpy()
            for i in range(len(start)):
                fw.write(str(i) + ": ")
                for j in range(len(start[i])):
                    if start[i, j] == 1:
                        fw.write("%d\t" % j)

                fw.write("\n")
            fw.write("\n")

            fw.write("end\n")
            end = end.numpy()
            for i in range(len(end)):
                fw.write(str(i) + ": ")
                for j in range(len(end[i])):
                    if end[i, j] == 1:
                        fw.write("%d\t" % j)

                fw.write("\n")
            fw.write("\n")

            fw.write("span\n")
            span = span.numpy()
            val = val.numpy()

            for i in range(len(span)):
                fw.write(str(i) + ": ")
                for j in range(len(span[i])):
                    for k in range(len(span[i, j])):
                        if val[i, j, k] == 1:
                            fw.write("%d;%d;%d\t" % (j, k, span[i, j, k]))
                fw.write("\n")
            fw.write("\n")

            fw.write("val\n")
            for i in range(len(val)):
                fw.write(str(i) + ": ")
                for j in range(len(val[i])):
                    for k in range(len(val[i, j])):
                        if val[i, j, k] == 1:
                            fw.write("%d;%d\t" % (j, k))
                fw.write("\n")
            fw.write("\n")

            span_predict_, tp_, tn_, fp_ = model.predict(data)

            fw.write("predict\n")
            span_predict_ = span_predict_[0]
            for i in range(len(span_predict_)):
                fw.write(str(i) + ": ")
                for j in range(len(span_predict_[i])):
                    for k in range(len(span_predict_[i, j])):
                        if val[i, j, k] == 1:
                            fw.write("%d;%d;%d\t" % (j, k, span_predict_[i, j, k]))
                fw.write("\n")
            fw.write("\n")

            fw.write("TP: %d TN: %d FP: %d\n\n" % (tp_, tn_, fp_))

            tp += tp_
            tn += tn_
            fp += fp_

        precision = tp / (tp + fp + params.eps)
        recall = tp / (tp + tn + params.eps)
        f1 = 2.0 * precision * recall / (precision + recall + params.eps)

        sys.stdout.write('\nprecision: %.4f recall: %.4f f1: %.4f\n\n' % (precision, recall, f1))
        sys.stdout.flush()


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
    char_inverse_dict = {v: k for k, v in char_dict.items()}

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
    elif params.mode == "test":
        user.test()
    else:
        user.predict()
