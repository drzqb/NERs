'''
    robert for medical ner with tf2.0
'''

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Embedding, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from official.nlp.optimization import AdamWeightDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import TruncatedNormal
from OtherUtils import load_vocab

import numpy as np

import os, sys
import argparse

from FuncUtils import gelu, softmax, create_initializer, checkpoint_loader

PARAMS_bert_path = "pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12"

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=10, help='Epochs during training')
parser.add_argument('--patience', type=int, default=3, help='epochs of early stop tolerance')
parser.add_argument('--lr', type=float, default=1.0e-5, help='Initial learing rate')
parser.add_argument('--drop_rate', type=float, default=0.1, help='drop rate')
parser.add_argument('--label_dim', type=int, default=11, help='number of ner labels')
parser.add_argument('--check', type=str, default='model/medical_robertlinear', help='The path where model shall be saved')
parser.add_argument('--mode', type=str, default='train0', help='The mode of train or predict as follows: '
                                                               'train0: begin to train or retrain'
                                                               'tran1:continue to train'
                                                               'predict: predict')
params = parser.parse_args()

PARAMS_maxword = 512
PARAMS_vocab_size = 21128
PARAMS_type_vocab_size = 2

PARAMS_head = 12
PARAMS_hidden_size = 768
PARAMS_intermediate_size = 4 * 768

PARAMS_train_file = [
    'data/TFRecordFiles/train.tfrecord',
]

PARAMS_dev_file = [
    'data/TFRecordFiles/dev.tfrecord',
]


def load_model_weights_from_checkpoint_bert(model, checkpoint_file):
    """Load trained official modelfiles from checkpoint.

    :param model: Built keras modelfiles.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    """
    loader = checkpoint_loader(checkpoint_file)

    weights = [
        loader('bert/embeddings/position_embeddings'),
        loader('bert/embeddings/word_embeddings'),
        loader('bert/embeddings/token_type_embeddings'),
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ]
    model.get_layer('embeddings').set_weights(weights)

    weights_a = []
    weights_f = []
    for i in range(12):
        pre = 'bert/encoder/layer_' + str(i) + '/'
        weights_a.extend([
            loader(pre + 'attention/self/query/kernel'),
            loader(pre + 'attention/self/query/bias'),
            loader(pre + 'attention/self/key/kernel'),
            loader(pre + 'attention/self/key/bias'),
            loader(pre + 'attention/self/value/kernel'),
            loader(pre + 'attention/self/value/bias'),
            loader(pre + 'attention/output/dense/kernel'),
            loader(pre + 'attention/output/dense/bias'),
            loader(pre + 'attention/output/LayerNorm/gamma'),
            loader(pre + 'attention/output/LayerNorm/beta')])

        weights_f.extend([
            loader(pre + 'intermediate/dense/kernel'),
            loader(pre + 'intermediate/dense/bias'),
            loader(pre + 'output/dense/kernel'),
            loader(pre + 'output/dense/bias'),
            loader(pre + 'output/LayerNorm/gamma'),
            loader(pre + 'output/LayerNorm/beta')])

    weights = weights_a + weights_f
    model.get_layer('encoder').set_weights(weights)


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

    def call(self, senwrong, **kwargs):
        sequencemask = tf.greater(senwrong, 0)
        seq_length = tf.shape(senwrong)[1]
        mask = tf.tile(tf.expand_dims(sequencemask, axis=1), [PARAMS_head, seq_length, 1])
        sum_ls = tf.reduce_sum(tf.cast(sequencemask, tf.int32), axis=-1) - 2

        return mask, seq_length, sum_ls


class Embeddings(Layer):
    def __init__(self, **kwargs):
        super(Embeddings, self).__init__(**kwargs)

        self.word_embeddings = Embedding(PARAMS_vocab_size,
                                         PARAMS_hidden_size,
                                         embeddings_initializer=create_initializer(),
                                         dtype=tf.float32,
                                         name="word_embeddings")

        self.token_embeddings = Embedding(PARAMS_type_vocab_size,
                                          PARAMS_hidden_size,
                                          embeddings_initializer=create_initializer(),
                                          dtype=tf.float32,
                                          name='token_type_embeddings')

        self.position_embeddings = self.add_weight(name='position_embeddings',
                                                   shape=[PARAMS_maxword, PARAMS_hidden_size],
                                                   dtype=tf.float32,
                                                   initializer=create_initializer())

        self.layernormanddrop = LayerNormalizeAndDrop(name="layernormanddrop")

    def call(self, inputs, **kwargs):
        sen, seqlen = inputs
        sen_embed = self.word_embeddings(sen)

        token_embed = self.token_embeddings(tf.zeros_like(sen, dtype=tf.int32))
        pos_embed = self.position_embeddings[:seqlen]

        all_embed = sen_embed + token_embed + pos_embed

        return self.layernormanddrop(all_embed)


class LayerNormalizeAndDrop(Layer):
    def __init__(self, **kwargs):
        super(LayerNormalizeAndDrop, self).__init__(**kwargs)

        self.layernorm = LayerNormalization(name="layernorm")
        self.dropout = Dropout(params.drop_rate)

    def call(self, inputs, **kwargs):
        return self.dropout(self.layernorm(inputs))


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.dense_q = Dense(PARAMS_hidden_size,
                             name='query',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_k = Dense(PARAMS_hidden_size,
                             name='key',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_v = Dense(PARAMS_hidden_size,
                             name='value',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_o = Dense(PARAMS_hidden_size,
                             name='output',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dropoutsoft = Dropout(params.drop_rate)
        self.dropoutres = Dropout(params.drop_rate)
        self.layernorm = LayerNormalization(name='layernormattn')

    def call(self, inputs, **kwargs):
        x, mask = inputs
        q = tf.concat(tf.split(self.dense_q(x), PARAMS_head, axis=-1), axis=0)
        k = tf.concat(tf.split(self.dense_k(x), PARAMS_head, axis=-1), axis=0)
        v = tf.concat(tf.split(self.dense_v(x), PARAMS_head, axis=-1), axis=0)
        qk = tf.matmul(q, tf.transpose(k, [0, 2, 1])) / tf.sqrt(PARAMS_hidden_size / PARAMS_head)
        attention_output = self.dense_o(tf.concat(
            tf.split(tf.matmul(self.dropoutsoft(softmax(qk, mask)), v), PARAMS_head, axis=0),
            axis=-1))

        return self.layernorm(x + self.dropoutres(attention_output))


class FeedFord(Layer):
    def __init__(self, **kwargs):
        super(FeedFord, self).__init__(**kwargs)

        self.dense_ffgelu = Dense(PARAMS_intermediate_size,
                                  kernel_initializer=create_initializer(),
                                  dtype=tf.float32,
                                  name='intermediate',
                                  activation=gelu)
        self.dense_ff = Dense(PARAMS_hidden_size,
                              kernel_initializer=create_initializer(),
                              dtype=tf.float32,
                              name='output')
        self.dropoutres = Dropout(params.drop_rate)
        self.layernorm = LayerNormalization(name='layernormffd')

    def call(self, inputs, **kwargs):
        return self.layernorm(inputs + self.dropoutres(self.dense_ff(self.dense_ffgelu(inputs))))


class Encoder(Layer):
    def __init__(self, layers, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.layers = layers

        self.attention = [Attention(name="attnlayer_%d" % k) for k in range(self.layers)]
        self.ffd = [FeedFord(name="ffdlayer_%d" % k) for k in range(self.layers)]

    def get_config(self):
        config = {"layers": self.layers}
        base_config = super(Encoder, self).get_config()
        return dict(base_config, **config)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        for k in range(self.layers):
            x = self.ffd[k](self.attention[k](inputs=(x, mask)))

        return x


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


class USER():
    def build_model(self):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)
        lab = Input(shape=[None], name='lab', dtype=tf.int32)

        mask, seqlen, sumls = Mask(name="mask")(sen)

        now = Embeddings(name="embeddings")(inputs=(sen, seqlen))

        now = Encoder(layers=12, name="encoder")(inputs=(now, mask))

        sequence_split = SplitSequence(name="splitsequence")(now)

        predict = Linear(label_dim=params.label_dim, name="linear")(inputs=(sequence_split, lab, sumls))

        model = Model(inputs=[sen, lab], outputs=predict)

        model.summary()

        return model

    def train(self):
        model = self.build_model()

        if params.mode == 'train0':
            load_model_weights_from_checkpoint_bert(model, PARAMS_bert_path + "/bert_model.ckpt")
        if params.mode == 'train1':
            model.load_weights(params.check + '/medical.h5')

        optimizer = AdamWeightDecay(learning_rate=params.lr,
                                    weight_decay_rate=0.01,
                                    epsilon=1.0e-6,
                                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        model.compile(optimizer=optimizer)

        batch_data = batched_data([PARAMS_train_file],
                                  single_example_parser,
                                  params.batch_size,
                                  padded_shapes={"sen": [-1], "lab": [-1]},
                                  buffer_size=100 * params.batch_size)
        dev_data = batched_data([PARAMS_dev_file],
                                single_example_parser,
                                params.batch_size,
                                padded_shapes={"sen": [-1], "lab": [-1]},
                                buffer_size=100 * params.batch_size)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=params.patience),
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


if __name__ == "__main__":
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
