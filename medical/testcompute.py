import tensorflow as tf


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

    start = tf.reshape(sequence_parsed['start'], [5, seqlen])

    end = tf.reshape(sequence_parsed['end'], [5, seqlen])

    span = tf.reshape(sequence_parsed['span'], [5, seqlen, seqlen])

    val = tf.reshape(sequence_parsed['val'], [5, seqlen, seqlen])

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


train_data = batched_data(['data/TFRecordFiles/dev_span.tfrecord'],
                          single_example_parser,
                          8,
                          padded_shapes={"sen": [-1],
                                         "start": [5, -1],
                                         "end": [5, -1],
                                         "span": [5, -1, -1],
                                         "val": [5, -1, -1],
                                         },
                          buffer_size=100 * 8,
                          shuffle=False)

for batch, data in enumerate(train_data):
    sen, _, _, _, _ = data
    print(batch + 1, " : ", tf.shape(data['sen']))
