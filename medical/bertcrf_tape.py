'''
    bert + crf for medical ner with tf2.0
    bert通过transformers加载
    tf.GradientTape
    CRF层1000倍BERT层学习率
'''

import tensorflow as tf
from tensorflow import keras

from transformers.optimization_tf import AdamWeightDecay
from transformers import TFBertModel, BertConfig
import tensorflow_addons as tfa
from OtherUtils import load_vocab

import numpy as np

import sys, os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=20, help='Epochs during training')
parser.add_argument('--lr', type=float, default=1.0e-5, help='Initial learing rate')
parser.add_argument('--eps', type=float, default=1.0e-6, help='epsilon')
parser.add_argument('--label_dim', type=int, default=11, help='number of ner labels')
parser.add_argument('--check', type=str, default='model/medical_bertcrf_tape',
                    help='The path where model shall be saved')
parser.add_argument('--hfmodel', type=str, default='e:/tools/chinese-roberta-wwm-ext',
                    help='The path where hugging face model saved')
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


class Mask(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, sen, **kwargs):
        mask = tf.greater(sen, 0)
        sequencemask = tf.cast(mask, tf.int32)

        return tf.reduce_sum(sequencemask, axis=-1) - 2, mask[:, 1:-1], mask


class BERT(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BERT, self).__init__(**kwargs)

        config = BertConfig.from_pretrained(params.hfmodel, output_hidden_states=True)
        self.bert = TFBertModel.from_pretrained(params.hfmodel, config=config)

    def call(self, inputs, **kwargs):
        sen, mask = inputs
        return self.bert(input_ids=sen,
                         token_type_ids=tf.zeros_like(sen),
                         attention_mask=tf.cast(mask, tf.int32))[0]


class SplitSequence(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SplitSequence, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return x[:, 1:-1]


class CRF(keras.layers.Layer):
    def __init__(self, label_dim, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.label_dim = label_dim

    def build(self, input_shape):
        self.dense_ner = keras.layers.Dense(self.label_dim,
                                            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                            dtype=tf.float32,
                                            name='ner')

        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.label_dim, self.label_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        super(CRF, self).build(input_shape)

    def get_config(self):
        config = super(CRF, self).get_config().copy()
        config.update({
            'label_dim': self.label_dim,
        })

        return config

    def call(self, inputs, **kwargs):
        x, label, seqlen, mask = inputs

        output = self.dense_ner(x)

        log_likelihood, _ = tfa.text.crf_log_likelihood(output, label, seqlen, self.transitions)
        loss = tf.reduce_mean(-log_likelihood)

        viterbi_sequence, _ = tfa.text.crf_decode(output, self.transitions, seqlen)

        accuracy = tf.reduce_sum(tf.cast(tf.equal(viterbi_sequence, label), tf.float32) * tf.cast(
            tf.sequence_mask(seqlen, tf.reduce_max(seqlen)), tf.float32))
        accuracy = accuracy / tf.reduce_sum(tf.cast(seqlen, tf.float32))

        tp, tn, fp = PRF(viterbi_sequence, label, mask)

        return viterbi_sequence, tp, tn, fp, loss, accuracy


def PRF(y_pred, y_label, mask):
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_label, 0), tf.equal(y_label, y_pred)), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_label, 0), tf.not_equal(y_label, y_pred)), tf.float32))
    fp = tf.reduce_sum(
        tf.cast(tf.logical_and(mask, tf.logical_and(tf.equal(y_label, 0), tf.greater(y_pred, 0))), tf.float32))

    return tp, tn, fp


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


@tf.function(experimental_relax_shapes=True)
def train_step(data, model, optimizercrf, optimizerother):
    with tf.GradientTape(persistent=True) as tape:
        _, tp, tn, fp, loss, accuracy = model(data, training=True)

    crf_trainable_variables = [v for v in model.trainable_variables if v.name.startswith("crf")]
    other_trainable_variables = [v for v in model.trainable_variables if not v.name.startswith("crf")]

    gradientscrf = tape.gradient(loss, crf_trainable_variables)
    optimizercrf.apply_gradients(zip(gradientscrf, crf_trainable_variables))

    gradientsother = tape.gradient(loss, other_trainable_variables)
    optimizerother.apply_gradients(zip(gradientsother, other_trainable_variables))

    return tp, tn, fp, loss, accuracy


@tf.function(experimental_relax_shapes=True)
def dev_step(data, model):
    _, tp, tn, fp, loss, accuracy = model(data, training=False)

    return tp, tn, fp, loss, accuracy


class USER:
    def build_model(self):
        sen = keras.layers.Input(shape=[None], name='sen', dtype=tf.int32)
        lab = keras.layers.Input(shape=[None], name='lab', dtype=tf.int32)

        seqlen, mask, attn_mask = Mask(name="mask")(sen)

        sequence_output = BERT(name="bert")(inputs=(sen, attn_mask))

        sequence_split = SplitSequence(name="splitsequence")(sequence_output)

        predict = CRF(label_dim=params.label_dim, name="crf")(inputs=(sequence_split, lab, seqlen, mask))

        model = keras.Model(inputs=[sen, lab], outputs=predict)

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
            model.load_weights(params.check + '/medical.h5')

        optimizerother = AdamWeightDecay(learning_rate=params.lr,
                                         weight_decay_rate=0.01,
                                         epsilon=1.0e-6,
                                         exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        optimizercrf = AdamWeightDecay(learning_rate=1000.0 * params.lr,
                                       weight_decay_rate=0.01,
                                       epsilon=1.0e-6,
                                       exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        train_data = batched_data(['data/TFRecordFiles/train.tfrecord'],
                                  single_example_parser,
                                  params.batch_size,
                                  padded_shapes={"sen": [-1], "lab": [-1]},
                                  buffer_size=100 * params.batch_size)
        dev_data = batched_data(['data/TFRecordFiles/dev.tfrecord'],
                                single_example_parser,
                                params.batch_size,
                                padded_shapes={"sen": [-1], "lab": [-1]},
                                buffer_size=100 * params.batch_size)

        F1_max = 0.0
        per_save = 0

        for epoch in range(params.epochs):
            tp = []
            tn = []
            fp = []

            loss = []
            acc = []

            loss_av = 0.0
            acc_av = 0.0

            precision = 0.0
            recall = 0.0
            F1 = 0.0

            for batch, data in enumerate(train_data):
                tp_, tn_, fp_, loss_, accuracy_ = train_step(data, model, optimizercrf, optimizerother)

                tp.append(tp_)
                tn.append(tn_)
                fp.append(fp_)
                loss.append(loss_)
                acc.append(accuracy_)

                loss_av = np.mean(loss)
                acc_av = np.mean(acc)

                tpsum = np.sum(tp)
                tnsum = np.sum(tn)
                fpsum = np.sum(fp)
                precision = tpsum / (tpsum + fpsum + params.eps)
                recall = tpsum / (tpsum + tnsum + params.eps)
                F1 = 2.0 * precision * recall / (precision + recall + params.eps)

                if epoch == 0:
                    per_save += 1

                    # 每一步都是显示经过累积后的各项指标，而不是每个batch的各项的指标
                    print(
                        '\rEpoch %d/%d %d/? -loss:%.5f -acc:%5.1f -precision:%5.1f -recall:%5.1f -F1:%5.1f' % (
                            epoch + 1, params.epochs, batch + 1,
                            loss_av, 100.0 * acc_av,
                            100.0 * precision, 100.0 * recall, 100.0 * F1,
                        ), end=""
                    )

                else:
                    completeratio = batch / per_save
                    total_len = 20
                    rationum = int(completeratio * total_len)
                    if rationum < total_len:
                        ratiogui = "=" * rationum + ">" + "." * (total_len - 1 - rationum)
                    else:
                        ratiogui = "=" * total_len

                    # 每一步都是显示经过累积后的各项指标，而不是每个batch的各项的指标
                    print(
                        '\rEpoch %d/%d %d/%d [%s] -loss:%.5f -acc:%5.1f -precision:%5.1f -recall:%5.1f -F1:%5.1f' % (
                            epoch + 1, params.epochs, batch + 1, per_save,
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

            if epoch == 0:
                total_len = 20
                ratiogui = "=" * total_len
                print(
                    '\rEpoch %d/%d %d/%d [%s] -loss:%.5f -acc:%5.1f -precision:%5.1f -recall:%5.1f -F1:%5.1f' % (
                        epoch + 1, params.epochs, per_save, per_save,
                        ratiogui,
                        loss_av, 100.0 * acc_av,
                        100.0 * precision, 100.0 * recall, 100.0 * F1,
                    ), end=""
                )

            print(" -val_loss:%.5f -val_acc:%5.1f -val_precision:%5.1f -val_recall:%5.1f -val_F1:%5.1f\n" % (
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

        for batch, data in enumerate(dev_data):
            tp_, tn_, fp_, loss_, accuracy_ = dev_step(data, model)

            tp.append(tp_)
            tn.append(tn_)
            fp.append(fp_)
            loss.append(loss_)
            acc.append(accuracy_)

        loss_av = np.mean(loss)
        acc_av = np.mean(acc)
        tp_sum = np.sum(tp)
        tn_sum = np.sum(tn)
        fp_sum = np.sum(fp)

        precision = tp_sum / (tp_sum + fp_sum + params.eps)
        recall = tp_sum / (tp_sum + tn_sum + params.eps)
        F1 = 2.0 * precision * recall / (precision + recall + params.eps)

        return loss_av, acc_av, precision, recall, F1

    def predict(self):
        model = self.build_model()
        model.load_weights(params.check + '/medical.h5')

        predict, _, _, _, _, _, _ = model.predict([sent, tf.ones_like(sent)[:, 1:-1]])

        querycheck(predict)

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
        plt.savefig("bertcrf_tape_PRF.jpg", dpi=500, bbox_inches="tight")


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

            user.plothistory()
        else:
            user.predict()
