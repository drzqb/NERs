import tensorflow as tf
from OtherUtils import load_vocab
from tqdm import tqdm
import numpy as np


def medical(filepath, train_tfrecordfilepath, dev_tfrecordfilepath):
    char_dict = load_vocab("data/OriginalFiles/vocab.txt")

    writer_train = tf.io.TFRecordWriter(train_tfrecordfilepath)
    writer_dev = tf.io.TFRecordWriter(dev_tfrecordfilepath)

    label_dict = {
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

    m_samples_train = 0
    m_samples_dev = 0

    with open(filepath, "r", encoding="utf-8") as fr:
        sent2id = [101]
        label2id = []

        for line in tqdm(fr):
            line = line.rstrip().split("\t")

            char = line[0]
            label = line[1]

            sent2id.append(char_dict[char] if char in char_dict.keys() else char_dict["[UNK]"])

            label2id.append(label_dict[label])

            if char == "。":
                sent2id.append(102)

                assert len(sent2id) == len(label2id) + 2

                sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in sent2id]

                lab_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[lab_])) for lab_ in label2id]

                seq_example = tf.train.SequenceExample(
                    feature_lists=tf.train.FeatureLists(feature_list={
                        'sen': tf.train.FeatureList(feature=sen_feature),
                        'lab': tf.train.FeatureList(feature=lab_feature)
                    })
                )

                serialized = seq_example.SerializeToString()

                if np.random.random() > 0.1:
                    writer_train.write(serialized)
                    m_samples_train += 1
                else:
                    writer_dev.write(serialized)
                    m_samples_dev += 1

                print("训练集样本数：", m_samples_train)
                print("验证集样本数：", m_samples_dev)

                sent2id = [101]
                label2id = []

    writer_train.close()
    writer_dev.close()

    print("训练集样本数：", m_samples_train)
    print("验证集样本数：", m_samples_dev)


if __name__ == "__main__":
    medical("data/OriginalFiles/train.txt",
            "data/TFRecordFiles/train.tfrecord",  # 7036
            "data/TFRecordFiles/dev.tfrecord",  # 781
            )
