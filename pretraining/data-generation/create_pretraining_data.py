# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../../bert")

from tensorflow.keras.utils import Progbar
from tree_parser import is_parsable, split_into_subtrees

import collections
import random
import json
import os
import tokenization
import modeling
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("configuration_dir", "../../configuration/",
                    "Path to the configuration directory.")

GSG_PROB = 0

class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(self.tokens))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        self.masked_lm_labels))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instance, writer, tokenizer, max_seq_length,
                                    max_predictions_per_seq, show_example=False):
  
    """Create TF example files from `TrainingInstance`s."""
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writer.write(tf_example.SerializeToString())

    if show_example:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(instance.tokens))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, writers, tokenizer, max_seq_length,
                              dupe_factor, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  vocab_words = list(tokenizer.vocab.keys())

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  
  for input_file in input_files:
    lines = 0
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        if not reader.readline():
          break
        lines += 1

    bar = Progbar(lines)

    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
    
        if tokens:
          all_documents[-1].append(tokens)
        bar.add(1)

  # Remove empty documents
  
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  instances = []

  bar2 = Progbar(dupe_factor)
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      create_instances_from_document(
              all_documents, document_index, writers, tokenizer, max_seq_length,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
    bar2.add(1)

def create_instances_from_document(
    all_documents, document_index, writers, tokenizer, max_seq_length,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]
  rng.shuffle(document)

  # Account for [CLS] and [SEP]
  max_num_tokens = max_seq_length - 2

  writer_index = 0
  instances = 0

  for sentence in document:
    if (not is_parsable(sentence)):
      print('not parsable: ', sentence)
      continue
    subtrees = split_into_subtrees(sentence, max_num_tokens)

    for subtree in subtrees:
      subtree = list(filter(lambda a: a != '(' and a != ')', subtree))
       
      tokens = ["[CLS]"] + subtree + ["[SEP]"]
      assert len(tokens) <= max_seq_length

      segment_ids = [0] * len(tokens)

      (tokens, masked_lm_positions, 
        masked_lm_labels) = create_masked_lm_predictions(
               tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
          
      instance = TrainingInstance(
        tokens=tokens,
        segment_ids=segment_ids,
        is_random_next=False,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)
      
      show_example = instances < 50
      
      write_instance_to_example_files(
        instance, writers[writer_index], tokenizer, max_seq_length, max_predictions_per_seq, show_example)

      writer_index = (writer_index + 1) % len(writers)
      instances += 1
  
  tf.logging.info("Wrote %d instances.\n", instances) 

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""
  
  print(GSG_PROB)
  gsg_prob = GSG_PROB
  masked_lms = []
  masked_subs = []
  output_tokens = list(tokens)
  gsg_to_cover = round(len(tokens) - 2) * gsg_prob

  min_subtree_size = int(gsg_to_cover * 1 / 5)
  max_subtree_size = int(gsg_to_cover)

  subs = get_small_subtrees(tokens[1:-1], min_subtree_size, max_subtree_size)
  rng.shuffle(subs)

  cover_len = 0
  for subtree in subs:
    if (cover_len + len(subtree) > int(gsg_to_cover)):
      break

    cover_len += len(subtree)
    masked_subs.append(subtree)

  for subtree in masked_subs:
    beg = find_subtree(tokens, subtree)

    for pos in range(len(subtree)):
      idx = beg + pos

      # We don't want to cover any token twice
      if (output_tokens[idx] != "[MASK2]"):
        output_tokens[idx] = "[MASK2]"
        masked_lms.append(MaskedLmInstance(index=idx, label=tokens[idx]))


  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]" or output_tokens[i] == "[MASK2]":
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)
  gsg_masked = len(masked_lms)

  num_to_predict = min(max_predictions_per_seq - gsg_masked,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict + gsg_masked:
      break

    convered_indexes.add(index)
    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token
    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  
  assert len(masked_lms) <= num_to_predict + gsg_masked
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)

def add_conf_dir_to_paths(configuration, config_dir):
    configuration["general"]["vocab_file"] = os.path.join(
        config_dir, configuration["general"]["vocab_file"]
    )
    configuration["general"]["bert_config_file"] = os.path.join(
        config_dir, configuration["general"]["bert_config_file"]
    )
    configuration["pretraining"]["data-generation"]["input_data_file"] = os.path.join(
        config_dir, configuration["pretraining"]["data-generation"]["input_data_file"]
    )

def main(_):
    with open(os.path.join(FLAGS.configuration_dir, "config.json")) as f:
        configuration = json.load(f)
        add_conf_dir_to_paths(configuration, FLAGS.configuration_dir)

    vocab_file = configuration["general"]["vocab_file"]
    input_data_file = configuration["pretraining"]["data-generation"]["input_data_file"]
    num_shards = configuration["pretraining"]["data-generation"]["num_shards"]
    random_seed = configuration["pretraining"]["data-generation"]["random_seed"]
    max_seq_length = configuration["general"]["max_seq_length"]
    dupe_factor = configuration["pretraining"]["data-generation"]["dupe_factor"]
    max_predictions_per_seq = configuration["pretraining"]["data-generation"][
        "max_predictions_per_seq"
    ]
    masked_lm_prob = configuration["pretraining"]["data-generation"]["masked_lm_prob"]
    output_data_dir = configuration["pretraining"]["data-generation"]["output_data_dir"]
    GSG_PROB = configuration["pretraining"]["data-generation"]["gsg_prob"]

    bert_config_file = configuration["general"]["bert_config_file"]
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.SpaceTokenizer(vocab=vocab_file, vocab_size=bert_config.vocab_size)

    input_files = [input_data_file]
    output_files = [os.path.join(output_data_dir, "train_{0:04d}.tfrecord".format(i)) for i in range(num_shards)]

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)
    
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)
    
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    rng = random.Random(random_seed)
    create_training_instances(
        input_files,
        writers,
        tokenizer,
        max_seq_length,
        dupe_factor,
        masked_lm_prob,
        max_predictions_per_seq,
        rng
    )
    
    for writer in writers:
        writer.close()

if __name__ == "__main__":
    flags.mark_flag_as_required("configuration_dir")
    tf.app.run()
