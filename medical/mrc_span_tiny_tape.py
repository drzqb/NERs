'''
    tinybert for ner with tf2.0
    tinybert通过transformers加载
    通过任务Dense来区分 TREATMENT、BODY、SIGNS、CHECK、DISEASE
    自定义训练过程
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel, BertTokenizer, BertConfig
from transformers.optimization_tf import WarmUp, AdamWeightDecay
from OtherUtils import load_vocab

import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=100, help='Epochs during training')
parser.add_argument('--lr', type=float, default=5.0e-4, help='Initial learing rate')
parser.add_argument('--eps', type=float, default=1.0e-6, help='epsilon')
parser.add_argument('--label_num', type=int, default=5, help='number of ner labels')
parser.add_argument('--per_save', type=int, default=440, help='save model per num')
parser.add_argument('--check', type=str, default='model/mrc_span_tiny_tape', help='The path where model saved')
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
                    -alpha * (1.0 - y_pred) ** gamma * tf.math.log(y_pred + params.eps),
                    -(1.0 - alpha) * y_pred ** gamma * tf.math.log(1.0 - y_pred + params.eps))

    return tf.squeeze(loss, axis=-1)


class Mask(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, sen, **kwargs):
        sequencemask = tf.cast(tf.greater(sen, 0), tf.int32)

        return tf.reduce_sum(sequencemask, axis=-1) - 2


class BERT(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BERT, self).__init__(**kwargs)

        config = BertConfig.from_pretrained("uer/chinese_roberta_L-2_H-128", output_hidden_states=True)
        self.bert = TFBertModel.from_pretrained("uer/chinese_roberta_L-2_H-128", config=config)

    def call(self, inputs, **kwargs):
        return self.bert(input_ids=inputs,
                         token_type_ids=tf.zeros_like(inputs),
                         attention_mask=tf.cast(tf.greater(inputs, 0), tf.int32)
                         )[0]


class SplitSequence(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SplitSequence, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return x[:, 1:-1]


class MRC(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MRC, self).__init__(**kwargs)

        self.dense_startend = keras.layers.Dense(2 * params.label_num,
                                                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                                 dtype=tf.float32,
                                                 activation="sigmoid",
                                                 name='startend')

        self.dense_span = keras.layers.Dense(params.label_num,
                                             kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                             dtype=tf.float32,
                                             activation="sigmoid",
                                             name='span')

    def call(self, inputs, **kwargs):
        # x: B*N*768,start,end: B*5*N,span,val: B*5*N*N seqlen: B
        x, start, end, span, val, seqlen = inputs

        ##################################### start 和 end 的 logits  ###################################################

        # B*N*10
        startend_output = self.dense_startend(x)

        # B*10*N
        startend_output = tf.transpose(startend_output, [0, 2, 1])

        ##################################### start 和 end 预测  ########################################################

        # B*10*N
        startend_predict = tf.cast(tf.greater(startend_output, 0.5), tf.int32)

        # B*N
        sequencemask = tf.sequence_mask(seqlen, tf.reduce_max(seqlen))

        # B*10*N
        sequencemask = tf.cast(tf.tile(tf.expand_dims(sequencemask, axis=1), [1, 2 * params.label_num, 1]), tf.int32)

        # B*10*N
        startend_predict *= sequencemask

        # B*5*N,B*5*N
        start_predict, end_predict = tf.split(startend_predict, 2, axis=1)

        #################################### start 和 end 损失  #########################################################

        # B*10*N*1
        startend_output = tf.expand_dims(startend_output, axis=-1)

        # B*10*N
        startend = tf.concat([start, end], axis=1)

        # B*10*N*1
        startend = tf.expand_dims(startend, axis=-1)

        # B*10*N
        # startend_loss = bce(reduction=tf.keras.losses.Reduction.NONE)(startend, startend_output)
        startend_loss = focal_loss(startend, startend_output)

        # B*10*N
        sequencemask = tf.cast(sequencemask, tf.float32)

        # B*10*N
        startend_loss *= sequencemask

        startend_loss = tf.reduce_sum(startend_loss) / tf.reduce_sum(sequencemask)

        ############################################ span 的 logits  ###################################################

        B = tf.shape(x)[0]
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

        ############################################ span 预测  #########################################################

        # N*N 右上三角
        rightuptria = tf.transpose(tf.sequence_mask(tf.range(1, N + 1), N, tf.int32))
        rightuptria = tf.tile(tf.expand_dims(rightuptria, axis=0), [params.label_num, 1, 1])
        rightuptria = tf.tile(tf.expand_dims(rightuptria, axis=0), [B, 1, 1, 1])

        # B*5*N*N
        span_predict = tf.cast(tf.greater(span_output, 0.5), tf.int32)

        # B*5*N --> B*5*N*1
        start_predict_expand = tf.cast(tf.expand_dims(start_predict, axis=-1), tf.int32)

        # B*5*N*N
        start_predict_expand = tf.tile(start_predict_expand, [1, 1, 1, N])

        # B*5*N --> B*5*1*N
        end_predict_expand = tf.cast(tf.expand_dims(end_predict, axis=2), tf.int32)

        # B*5*N*N
        end_predict_expand = tf.tile(end_predict_expand, [1, 1, N, 1])

        # B*5*N*N
        span_predict = span_predict * start_predict_expand * end_predict_expand * rightuptria

        # B*5*N*N
        accuracy = tf.cast(tf.equal(span_predict, span), tf.float32)

        # B*5*N*N
        valf = tf.cast(val, tf.float32)

        accuracy *= valf

        valsum = tf.reduce_sum(valf)

        accuracysum = tf.reduce_sum(accuracy)

        ########################################### span 损失  ##########################################################

        # B*5*N*N*1
        span_outputlogits = tf.expand_dims(span_output, axis=-1)

        # B*5*N*N*1
        spanexpand = tf.expand_dims(span, axis=-1)

        # B*5*N*N
        span_loss = focal_loss(spanexpand, span_outputlogits)

        # B*5*N*N
        span_loss *= valf

        span_loss = tf.reduce_sum(span_loss) / (valsum + params.eps)

        loss = 0.15 * startend_loss + 0.7 * span_loss

        ###########################################  TP TN FP  #########################################################

        # 是实体，预测是实体
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(span, 1), tf.equal(span_predict, 1)), tf.float32))

        # 是实体，预测不是实体
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(span, 1), tf.not_equal(span_predict, 1)), tf.float32))

        # 不是有效实体，预测是实体
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(span, 0), tf.equal(span_predict, 1)), tf.float32))

        return span_predict, start_predict, end_predict, tp, tn, fp, loss, accuracysum, valsum


def querycheck(predict, start_predict, end_predict):
    sys.stdout.write('\n')
    sys.stdout.flush()
    labels = ["TREATMENT", "BODY", "SIGNS", "CHECK", "DISEASE"]

    for i, pre in enumerate(predict):
        sys.stdout.write(sentences[i] + '\n')

        for j in range(len(predict[i])):
            sys.stdout.write(labels[j] + ': ')

            for k in range(leng[i]):
                for l in range(leng[i]):
                    if l >= k and predict[i, j, k, l] == 1 and start_predict[i, j, k] == 1 and end_predict[
                        i, j, l] == 1:
                        sys.stdout.write("%d;%d\t" % (k, l))

            sys.stdout.write('\n')
        sys.stdout.write('\n\n')

    sys.stdout.flush()


@tf.function(experimental_relax_shapes=True)
def train_step(data, model, optimizer):
    with tf.GradientTape() as tape:
        _, _, _, tp, tn, fp, loss, accuracysum, valsum = model(data, training=True)

    trainable_variables = model.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return tp, tn, fp, loss, accuracysum, valsum


@tf.function(experimental_relax_shapes=True)
def dev_step(data, model):
    _, _, _, tp, tn, fp, loss, accuracysum, valsum = model(data, training=False)

    return tp, tn, fp, loss, accuracysum, valsum


class USER:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-2_H-128")

    def build_model(self):
        sen = keras.layers.Input(shape=[None], name='sen', dtype=tf.int32)

        start = keras.layers.Input(shape=[params.label_num, None], name='start', dtype=tf.int32)

        end = keras.layers.Input(shape=[params.label_num, None], name='end', dtype=tf.int32)

        span = keras.layers.Input(shape=[params.label_num, None, None], name='span', dtype=tf.int32)

        val = keras.layers.Input(shape=[params.label_num, None, None], name='val', dtype=tf.int32)

        seqlen = Mask(name="mask")(sen)

        sequence_output = BERT(name="bert")(sen)

        sequence_split = SplitSequence(name="splitsequence")(sequence_output)

        predict = MRC(name="mrc")(inputs=(sequence_split, start, end, span, val, seqlen))

        model = keras.Model(inputs=[sen, start, end, span, val], outputs=predict)

        model.summary()

        return model

    def train(self):
        history = {
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "F1": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_F1": []
        }

        model = self.build_model()

        if params.mode == 'train1':
            model.load_weights(params.check + '/mrc.h5')

        decay_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=params.lr,
                                                                    decay_steps=params.epochs * params.per_save,
                                                                    end_learning_rate=0.0,
                                                                    power=1.0,
                                                                    cycle=False)

        warmup_schedule = WarmUp(initial_learning_rate=params.lr,
                                 decay_schedule_fn=decay_schedule,
                                 warmup_steps=2 * params.per_save,
                                 )

        optimizer = AdamWeightDecay(learning_rate=warmup_schedule,
                                    weight_decay_rate=0.01,
                                    epsilon=1.0e-6,
                                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        train_data = batched_data(['data/TFRecordFiles/train_span.tfrecord'],
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
        F1_max = 0.0

        for epoch in range(params.epochs):
            tp = []
            tn = []
            fp = []

            loss = []
            acc = []
            val = []

            for batch, data in enumerate(train_data):
                tp_, tn_, fp_, loss_, accuracysum_, valsum_ = train_step(data, model, optimizer)

                tp.append(tp_)
                tn.append(tn_)
                fp.append(fp_)
                loss.append(loss_)
                acc.append(accuracysum_)
                val.append(valsum_)

                loss_av = np.mean(loss)
                acc_av = np.sum(acc) / (np.sum(val) + params.eps)

                tpsum = np.sum(tp)
                tnsum = np.sum(tn)
                fpsum = np.sum(fp)
                precision = tpsum / (tpsum + fpsum + params.eps)
                recall = tpsum / (tpsum + tnsum + params.eps)
                F1 = 2.0 * precision * recall / (precision + recall + params.eps)

                completeratio = batch / params.per_save
                total_len = 20
                rationum = int(completeratio * total_len)
                if rationum < total_len:
                    ratiogui = "=" * rationum + ">" + "." * (total_len - 1 - rationum)
                else:
                    ratiogui = "=" * total_len

                # 每一步都是显示经过累积后的各项指标，而不是每个batch的各项的指标
                print(
                    '\rEpoch %d/%d %d/%d [%s] -loss: %.6f -acc:%6.1f -precision:%6.1f -recall:%6.1f -F1:%6.1f' % (
                        epoch + 1, params.epochs, batch + 1, params.per_save,
                        ratiogui,
                        loss_av, 100.0 * acc_av,
                        100.0 * precision, 100.0 * recall, 100.0 * F1,
                    ), end=""
                )

            history["loss"].append(loss_av)
            history["acc"].append(acc_av)
            history["precision"].append(precision)
            history["recall"].append(recall)
            history["F1"].append(F1)

            loss_val, acc_val, precision_val, recall_val, F1_val = self.dev_train(dev_data, model)

            print(" -val_loss: %.6f -val_acc:%6.1f -val_precision:%6.1f -val_recall:%6.1f -val_F1:%6.1f\n" % (
                loss_val, 100.0 * acc_val,
                100.0 * precision_val, 100.0 * recall_val, 100.0 * F1_val))

            history["val_loss"].append(loss_val)
            history["val_acc"].append(acc_val)
            history["val_precision"].append(precision_val)
            history["val_recall"].append(recall_val)
            history["val_F1"].append(F1_val)

            if F1_val > F1_max:
                model.save_weights(params.check + '/mrc.h5')
                F1_max = F1_val

        with open(params.check + "/history.txt", "w", encoding="utf-8") as fw:
            fw.write(str(history))

    def dev_train(self, dev_data, model):
        tp = []
        tn = []
        fp = []

        loss = []
        acc = []
        val = []

        for batch, data in enumerate(dev_data):
            tp_, tn_, fp_, loss_, accuracysum_, valsum_ = dev_step(data, model)

            tp.append(tp_)
            tn.append(tn_)
            fp.append(fp_)
            loss.append(loss_)
            acc.append(accuracysum_)
            val.append(valsum_)

        loss_av = np.mean(loss)
        acc_av = np.sum(acc) / (np.sum(val) + params.eps)
        tp_sum = np.sum(tp)
        tn_sum = np.sum(tn)
        fp_sum = np.sum(fp)

        precision = tp_sum / (tp_sum + fp_sum + params.eps)
        recall = tp_sum / (tp_sum + tn_sum + params.eps)
        F1 = 2.0 * precision * recall / (precision + recall + params.eps)

        return loss_av, acc_av, precision, recall, F1

    def predict(self):
        model = self.build_model()
        model.load_weights(params.check + '/mrc.h5')

        BN = tf.shape(sent)
        B, N = BN[0], BN[1] - 2

        predict, start_predict, end_predict, _, _, _ = model.predict([sent,
                                                                      tf.ones([B, params.label_num, N], tf.int32),
                                                                      tf.ones([B, params.label_num, N], tf.int32),
                                                                      tf.ones([B, params.label_num, N, N], tf.int32),
                                                                      tf.ones([B, params.label_num, N, N], tf.int32)])

        querycheck(predict, start_predict, end_predict)

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

            span_predict_, _, _, tp_, tn_, fp_ = model.predict(data)

            fw.write("span_real\n")
            for i in range(len(span)):
                fw.write(str(i) + ": ")
                for j in range(len(span[i])):
                    for k in range(len(span[i, j])):
                        if val[i, j, k] == 1 and span[i, j, k] == 1:
                            fw.write("%d;%d\t" % (j, k))
                fw.write("\n")
            fw.write("\n")

            fw.write("predict\n")
            span_predict_ = span_predict_[0]
            for i in range(len(span_predict_)):
                fw.write(str(i) + ": ")
                for j in range(len(span_predict_[i])):
                    for k in range(len(span_predict_[i, j])):
                        if span_predict_[i, j, k] == 1:
                            fw.write("%d;%d\t" % (j, k))
                fw.write("\n")
            fw.write("\n")

            fw.write("TP: %d TN: %d FP: %d\n\n" % (tp_, tn_, fp_))

            tp += tp_
            tn += tn_
            fp += fp_

        precision = tp / (tp + fp + params.eps)
        recall = tp / (tp + tn + params.eps)
        F1 = 2.0 * precision * recall / (precision + recall + params.eps)

        sys.stdout.write('\nprecision: %.4f recall: %.4f F1: %.4f\n\n' % (precision, recall, F1))
        sys.stdout.flush()

    def plothistory(self):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        with open(params.check + "/history.txt", "r", encoding="utf-8") as fr:
            history = fr.read()
            history = eval(history)

        gs = gridspec.GridSpec(2, 6)
        plt.subplot(gs[0, 1:3])
        plt.plot(history["loss"])
        plt.plot(history["val_loss"])
        plt.grid()
        plt.title('loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='best', prop={'size': 4})

        plt.subplot(gs[0, 3:5])
        plt.plot(history["acc"])
        plt.plot(history["val_acc"])
        plt.grid()
        plt.title('acc')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='best', prop={'size': 4})

        plt.subplot(gs[1, :2])
        plt.plot(history["precision"])
        plt.plot(history["val_precision"])
        plt.grid()
        plt.title('precision')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='best', prop={'size': 4})

        plt.subplot(gs[1, 2:4])
        plt.plot(history["recall"])
        plt.plot(history["val_recall"])
        plt.grid()
        plt.title('recall')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='best', prop={'size': 4})

        plt.subplot(gs[1, 4:])
        plt.plot(history["F1"])
        plt.plot(history["val_F1"])
        plt.grid()
        plt.title('F1')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='best', prop={'size': 4})

        plt.suptitle("Model Metrics")

        plt.tight_layout()
        plt.savefig("mrc_span_tiny_tape_PRF.jpg", dpi=500, bbox_inches="tight")


if __name__ == '__main__':
    if not os.path.exists(params.check):
        os.makedirs(params.check)

    user = USER()

    if params.mode == "plot":
        user.plothistory()
    else:
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

        sentences = [
            '1.患者老年女性，88岁；2.既往体健，否认药物过敏史。3.患者缘于5小时前不慎摔伤，伤及右髋部。伤后患者自感伤处疼痛，呼我院120接来我院，查左髋部X光片示：左侧粗隆间骨折。',
            '患者精神状况好，无发热，诉右髋部疼痛，饮食差，二便正常，查体：神清，各项生命体征平稳，心肺腹查体未见异常。',
            '女性，88岁，农民，双滦区应营子村人，主因右髋部摔伤后疼痛肿胀，活动受限5小时于2016-10-29；11：12入院。',
            '入院后完善各项检查，给予右下肢持续皮牵引，应用健骨药物治疗，患者略发热，查血常规：白细胞数12.18*10^9/L，中性粒细胞百分比92.00%。',
            '1患者老年男性，既往有高血压病史5年，血压最高达180/100mmHg，长期服用降压药物治疗，血压控制欠佳。'
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

            user.plothistory()
        elif params.mode == "test":
            user.test()
        elif params.mode == "predict":
            user.predict()
