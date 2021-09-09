import tensorflow as tf
from OtherUtils import load_vocab
from tqdm import tqdm
import numpy as np
import json


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


def bi2bieso():
    """
    BI标注转BIESO标注
    :return:
    """
    fw = open("data/OriginalFiles/train_bieso.txt", "w", encoding="utf-8")

    lastchar = "-2"
    lastlabel = "-2"

    with open("data/OriginalFiles/train.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            linenew = line.rstrip()

            if linenew == "":
                fw.write(lastchar + "\t" + lastlabel + "\n")
                lastchar = "-1"
                lastlabel = "-1"

            else:
                linenew = linenew.split("\t")
                char = linenew[0]
                label = linenew[1]
                if label != "O":
                    if lastlabel == "-2":
                        pass
                    elif lastlabel == "-1":
                        fw.write("\n")
                    elif lastlabel == "O":
                        fw.write(lastchar + "\t" + lastlabel + "\n")
                    else:
                        label_prefix, label_suffix = label.split("-")
                        lastlabel_prefix, lastlabel_suffix = lastlabel.split("-")
                        if lastlabel_suffix == "B":
                            if label_suffix == "B":
                                fw.write(lastchar + "\t" + lastlabel_prefix + "-S" + "\n")
                            else:
                                fw.write(lastchar + "\t" + lastlabel + "\n")
                        else:
                            if label_suffix == "B":
                                fw.write(lastchar + "\t" + lastlabel_prefix + "-E" + "\n")
                            else:
                                fw.write(lastchar + "\t" + lastlabel + "\n")
                else:
                    if lastlabel == "-2":
                        pass
                    elif lastlabel == "-1":
                        fw.write("\n")
                    elif lastlabel == "O":
                        fw.write(lastchar + "\t" + lastlabel + "\n")
                    else:
                        lastlabel_prefix, lastlabel_suffix = lastlabel.split("-")
                        if lastlabel_suffix == "B":
                            fw.write(lastchar + "\t" + lastlabel_prefix + "-S" + "\n")
                        else:
                            fw.write(lastchar + "\t" + lastlabel_prefix + "-E" + "\n")

                lastchar = char
                lastlabel = label

    fw.close()


def bieso2span():
    all_datas = []

    k = 0
    startid = 0
    context = ""
    span_posLabel = {}
    data = {}

    with open("data/OriginalFiles/train_bieso.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.rstrip()

            if line == "":
                data["context"] = context
                data["span_posLabel"] = span_posLabel
                all_datas.append(data)

                context = ""
                span_posLabel = {}
                data = {}
                k = 0
            else:
                char, label = line.split("\t")

                if label == "O":
                    pass
                else:
                    label_prefix, label_suffix = label.split("-")

                    if label_suffix == "B":
                        startid = k

                    elif label_suffix == "E":
                        endid = k
                        span_posLabel[str(startid) + ";" + str(endid)] = label_prefix

                    elif label_suffix == "S":
                        span_posLabel[str(k) + ";" + str(k)] = label_prefix

                context += char

                k += 1

    with open("data/OriginalFiles/train_span.txt", "w", encoding="utf-8") as f:
        json.dump(all_datas, f, ensure_ascii=False, indent=2)


def medicalspan(filepath, train_tfrecordfilepath, dev_tfrecordfilepath):
    char_dict = load_vocab("data/OriginalFiles/vocab.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        all_datas = json.load(f)

    writer_train = tf.io.TFRecordWriter(train_tfrecordfilepath)
    writer_dev = tf.io.TFRecordWriter(dev_tfrecordfilepath)

    m_samples_train = 0
    m_samples_dev = 0

    for data in tqdm(all_datas):
        text = data["context"]
        seqlen = len(text)

        sent2id = [101]
        sent2id += [char_dict.get(char, char_dict["[UNK]"]) for char in text]
        sent2id += [102]

        spanMatrix = {
            "TREATMENT": np.zeros([seqlen, seqlen], np.int),
            "BODY": np.zeros([seqlen, seqlen], np.int),
            "SIGNS": np.zeros([seqlen, seqlen], np.int),
            "CHECK": np.zeros([seqlen, seqlen], np.int),
            "DISEASE": np.zeros([seqlen, seqlen], np.int),
        }

        for se, label in data["span_posLabel"].items():
            startid, endid = se.split(";")
            spanMatrix[label][int(startid), int(endid)] = 1

        sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in sent2id]
        TREATMENT_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                             spanMatrix["TREATMENT"].flatten()]
        BODY_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                        spanMatrix["BODY"].flatten()]
        SIGNS_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                         spanMatrix["SIGNS"].flatten()]
        CHECK_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                         spanMatrix["CHECK"].flatten()]
        DISEASE_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in
                           spanMatrix["DISEASE"].flatten()]

        seq_example = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(feature_list={
                'sen': tf.train.FeatureList(feature=sen_feature),
                'treatment': tf.train.FeatureList(feature=TREATMENT_feature),
                'body': tf.train.FeatureList(feature=BODY_feature),
                'signs': tf.train.FeatureList(feature=SIGNS_feature),
                'check': tf.train.FeatureList(feature=CHECK_feature),
                'disease': tf.train.FeatureList(feature=DISEASE_feature),
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


if __name__ == "__main__":
    # medical("data/OriginalFiles/train.txt",
    #         "data/TFRecordFiles/train.tfrecord",  # 7036
    #         "data/TFRecordFiles/dev.tfrecord",  # 781
    #         )

    # bi2bieso()

    # bieso2span()
    medicalspan("data/OriginalFiles/train_span.txt",
                "data/TFRecordFiles/train_span.tfrecord",  # 7036
                "data/TFRecordFiles/dev_span.tfrecord",  # 781
                )
