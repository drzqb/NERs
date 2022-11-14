import tensorflow as tf
from OtherUtils import load_vocab
from tqdm import tqdm


def mrc(filepath, tfrecordfilepath):
    char_dict = load_vocab("data/OriginalFiles/vocab.txt")

    writer = tf.io.TFRecordWriter(tfrecordfilepath)

    label_dict = {
        'O': 0,
        'B-LOC': 1, 'I-LOC': 2,
        'B-PER': 3, 'I-PER': 4,
        'B-ORG': 5, 'I-ORG': 6,
    }

    m_samples = 0

    with open(filepath, "r", encoding="utf-8") as fr:
        for line in tqdm(fr):
            sent2id = [101]
            line = line.rstrip().split("\t")

            char = line[0].split("\x02")
            label = line[1].split("\x02")

            if len(char) > 200:
                char = char[:200]
                label = label[:200]

            sent2id.extend([char_dict.get(c, char_dict["[UNK]"]) for c in char])

            label2id = [label_dict[l] for l in label]

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

        writer.close()

        print("样本数：", m_samples)


def rewritehorizontal(sourcefilepath, destfilepath):
    fw = open(destfilepath, "w", encoding="utf-8")

    with open(sourcefilepath, "r", encoding="utf-8") as fr:
        for line in tqdm(fr):
            line = line.rstrip().split("\t")

            char = line[0].split("\x02")
            label = line[1].split("\x02")

            fw.write("".join(char) + "\t" + " ".join(label) + "\n")

    fw.close()


def rewritevertical(sourcefilepath, destfilepath):
    fw = open(destfilepath, "w", encoding="utf-8")

    with open(sourcefilepath, "r", encoding="utf-8") as fr:
        for line in tqdm(fr):
            line = line.rstrip().split("\t")

            char = line[0].split("\x02")
            label = line[1].split("\x02")

            for k in range(len(char)):
                fw.write(char[k] + "\t" + label[k] + "\n")

            fw.write("\n")

    fw.close()


if __name__ == "__main__":
    # mrc("data/OriginalFiles/msra_ner/train/part.0",
    #     "data/TFRecordFiles/mrc_train.tfrecord",  # 20864
    #     )
    # mrc("data/OriginalFiles/msra_ner/dev/part.0",
    #     "data/TFRecordFiles/mrc_dev.tfrecord",  # 2318
    #     )
    # mrc("data/OriginalFiles/msra_ner/test/part.0",
    #     "data/TFRecordFiles/mrc_test.tfrecord",  # 4636
    #     )

    rewritehorizontal("data/OriginalFiles/msra_ner/train/part.0",
            "data/OriginalFiles/msra_ner/train/parth.txt", )
    rewritehorizontal("data/OriginalFiles/msra_ner/dev/part.0",
            "data/OriginalFiles/msra_ner/dev/parth.txt", )
    rewritehorizontal("data/OriginalFiles/msra_ner/test/part.0",
            "data/OriginalFiles/msra_ner/test/parth.txt", )

    # rewritevertical("data/OriginalFiles/msra_ner/train/part.0",
    #                 "data/OriginalFiles/msra_ner/train/partv.txt", )
    # rewritevertical("data/OriginalFiles/msra_ner/dev/part.0",
    #                 "data/OriginalFiles/msra_ner/dev/partv.txt", )
    # rewritevertical("data/OriginalFiles/msra_ner/test/part.0",
    #                 "data/OriginalFiles/msra_ner/test/partv.txt", )
