import tensorflow as tf
from OtherUtils import load_vocab
from tqdm import tqdm


def resume(filepath, tfrecordfilepath):
    char_dict = load_vocab("data/OriginalFiles/vocab.txt")

    writer = tf.io.TFRecordWriter(tfrecordfilepath)

    label_dict = {
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

    m_samples = 0

    with open(filepath, "r", encoding="utf-8") as fr:
        sent2id = [101]
        label2id = []

        for line in tqdm(fr):
            if line.strip() == "":
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

                writer.write(serialized)
                m_samples += 1

                print("样本数：", m_samples)

                sent2id = [101]
                label2id = []

            else:
                line = line.rstrip().split(" ")

                char = line[0]
                label = line[1]

                sent2id.append(char_dict[char] if char in char_dict.keys() else char_dict["[UNK]"])

                label2id.append(label_dict[label])

    writer.close()

    print("样本数：", m_samples)


if __name__ == "__main__":
    # resume("data/OriginalFiles/demo.train.char",
    #        "data/TFRecordFiles/demo_train.tfrecord",  # 1148
    #        )
    # resume("data/OriginalFiles/demo.dev.char",
    #        "data/TFRecordFiles/demo_dev.tfrecord",  # 113
    #        )
    resume("data/OriginalFiles/demo.test.char",
           "data/TFRecordFiles/demo_test.tfrecord",  # 316
           )
