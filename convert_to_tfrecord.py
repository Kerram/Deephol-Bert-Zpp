# Code used to convert tsv files to tf_record files used by run_deephol.py fine tuning.
# Code is a modified part of older run_deephol.py, which is and was a mix of
# bert's classifier.py and deephol's architectures.py.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import tensorflow as tf
import sys

import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path",
    None,
    "Path to input file. It should be in tsv format.",
)

flags.DEFINE_string(
    "vocab_file", None, "The vocabulary file that the BERT model was trained on."
)

flags.DEFINE_string(
    "output_path",
    None,
    "Path where we will save tf_record file.",
)

flags.DEFINE_string(
    "set_type",
    None,
    "Flag specifying whether we are to convert train, valid or test set.",
)

flags.DEFINE_integer(
    "max_seq_length",
    512,
    "The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval."
                                           "Important, because we need to pad with fake samples to match it.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, goal, thm, tac_id=None, is_negative=None):
        """Constructs an InputExample.

    Args:
      guid: Unique id for the example.
      goal: The untokenized goal string
      thm:  The untokenized theorem string.
      tac_id: id of tactic for the goal
      is_negative: indicates whether the theorem matches the goal
    """
        self.guid = guid
        self.goal = goal
        self.thm = thm
        self.tac_id = tac_id
        self.is_negative = is_negative


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
     See run_classifier.py for details.
  """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        goal_input_ids,
        thm_input_ids,
        tac_id,
        is_negative,
        goal_str,
        thm_str,
        goal_input_mask,
        thm_input_mask,
        is_real_example=True,
    ):

        self.goal_input_ids = goal_input_ids
        self.goal_input_mask = goal_input_mask
        self.thm_input_ids = thm_input_ids
        self.thm_input_mask = thm_input_mask
        self.tac_id = tac_id
        self.is_negative = is_negative
        self.is_real_example = is_real_example
        self.goal_str = goal_str
        self.thm_str = thm_str


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_tac_labels(self):
        """Gets the list tac_ids"""
        raise NotImplementedError()

    def get_is_negative_labels(self):
        """Gets the list of is_negative labels"""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class DeepholProcessor(DataProcessor):
    """Processor for Deephol dataset"""

    def get_examples(self, data_path, set_type):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path), set_type
        )

    def get_tac_labels(self):
        return [str(i) for i in range(41)]

    def get_is_negative_labels(self):
        return ["False", "True"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            """ skip header """
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                #  The values really don't matter, because we are using test set only as a hack to export a model.
                goal = tokenization.convert_to_unicode(line[0])
                thm = tokenization.convert_to_unicode(line[1])
                is_negative = "True"
                tac_id = "0"
            else:
                goal = tokenization.convert_to_unicode(line[0])
                thm = tokenization.convert_to_unicode(line[1])
                is_negative = tokenization.convert_to_unicode(line[2])
                tac_id = tokenization.convert_to_unicode(line[3])
            examples.append(
                InputExample(
                    guid=guid,
                    goal=goal,
                    thm=thm,
                    tac_id=tac_id,
                    is_negative=is_negative,
                )
            )
        return examples


def convert_single_example(
    ex_index, example, tac_label_list, is_negative_label_list, max_seq_length, tokenizer
):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            goal_input_ids=[0] * max_seq_length,
            thm_input_ids=[0] * max_seq_length,
            tac_id=0,
            is_negative=True,
            is_real_example=False,
            goal_str="",
            thm_str="",
            goal_input_mask=[0] * max_seq_length,
            thm_input_mask=[0] * max_seq_length,
        )

    tac_label_map = {}
    for (i, label) in enumerate(tac_label_list):
        tac_label_map[label] = i

    is_negative_label_map = {}
    for (i, label) in enumerate(is_negative_label_list):
        is_negative_label_map[label] = i

    goal_input_ids, goal_input_mask = tokenizer.tokenize(example.goal, max_seq_length)
    thm_input_ids, thm_input_mask = tokenizer.tokenize(example.thm, max_seq_length)

    assert len(goal_input_ids) == max_seq_length
    assert len(thm_input_ids) == max_seq_length
    assert len(goal_input_mask) == max_seq_length
    assert len(thm_input_mask) == max_seq_length

    tac_id = tac_label_map[example.tac_id]
    is_negative = is_negative_label_map[example.is_negative]

    if ex_index < 1:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info(
            "goal_input_ids: %s" % " ".join([str(x) for x in goal_input_ids])
        )
        tf.logging.info("thm_input_ids: %s" % " ".join([str(x) for x in thm_input_ids]))
        tf.logging.info("tac_id (example): %s" % (example.tac_id))
        tf.logging.info("goal_input_mask: %s" % " ".join([str(x) for x in goal_input_mask]))
        tf.logging.info("thm_input_mask: %s" % " ".join([str(x) for x in thm_input_mask]))
        tf.logging.info("tac_id: %d" % (tac_id))
        tf.logging.info("is_negative: %d" % (is_negative))
        tf.logging.info("is_negative (example): %s" % (example.is_negative))

    feature = InputFeatures(
        goal_input_ids=goal_input_ids,
        thm_input_ids=thm_input_ids,
        tac_id=tac_id,
        is_negative=is_negative,
        is_real_example=True,
        goal_str=example.goal,
        thm_str=example.thm,
        goal_input_mask=goal_input_mask,
        thm_input_mask=thm_input_mask,
    )

    return feature


def file_based_convert_examples_to_features(
    examples, tac_label_list, is_negative_list, max_seq_length, tokenizer, output_file
):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(
            ex_index,
            example,
            tac_label_list,
            is_negative_list,
            max_seq_length,
            tokenizer,
        )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["goal_input_ids"] = create_int_feature(feature.goal_input_ids)
        features["thm_input_ids"] = create_int_feature(feature.thm_input_ids)
        features["tac_ids"] = create_int_feature([feature.tac_id])
        features["is_negative"] = create_int_feature([feature.is_negative])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        features["goal_input_mask"] = create_int_feature(feature.goal_input_mask)
        features["thm_input_mask"] = create_int_feature(feature.thm_input_mask)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def test_set_convert_examples_to_features(
    examples, tac_label_list, is_negative_list, max_seq_length, tokenizer, output_file
):
    """Convert a set of `InputExample`s to a TFRecord file.
    In case of a test set we add goal and theorem strings as additional fetures."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(
            ex_index,
            example,
            tac_label_list,
            is_negative_list,
            max_seq_length,
            tokenizer,
        )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["goal_input_ids"] = create_int_feature(feature.goal_input_ids)
        features["thm_input_ids"] = create_int_feature(feature.thm_input_ids)
        features["tac_ids"] = create_int_feature([feature.tac_id])
        features["is_negative"] = create_int_feature([feature.is_negative])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        features["goal_str"] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[bytes(feature.goal_str, encoding="utf-8")]))
        features['thm_str'] = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[bytes(feature.thm_str, encoding="utf-8")]))
        features["goal_input_mask"] = create_int_feature(feature.goal_input_mask)
        features["thm_input_mask"] = create_int_feature(feature.thm_input_mask)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    csv.field_size_limit(sys.maxsize)
    tf.logging.set_verbosity(tf.logging.INFO)

    processor = DeepholProcessor()
    tokenizer = tokenization.WordSplitterTokenizer(vocab=FLAGS.vocab_file)

    examples = processor.get_examples(FLAGS.data_path, FLAGS.set_type)

    tac_labels = processor.get_tac_labels()
    is_negative_labels = processor.get_is_negative_labels()

    if FLAGS.set_type == 'eval':
        # TPU requires a fixed batch size for all batches, therefore the number
        # of examples must be a multiple of the batch size, or else examples
        # will get dropped. So we pad with fake examples which are ignored
        # later on. These do NOT count towards the metric (all tf.metrics
        # support a per-instance weight, and these get a weight of 0.0).
        while len(examples) % FLAGS.eval_batch_size != 0:
            examples.append(PaddingInputExample())

    if FLAGS.set_type == 'test':
        test_set_convert_examples_to_features(
            examples,
            tac_labels,
            is_negative_labels,
            FLAGS.max_seq_length,
            tokenizer,
            FLAGS.output_path,
        )
    else:
        file_based_convert_examples_to_features(
            examples,
            tac_labels,
            is_negative_labels,
            FLAGS.max_seq_length,
            tokenizer,
            FLAGS.output_path,
        )


if __name__ == "__main__":
    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("output_path")
    flags.mark_flag_as_required("set_type")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
