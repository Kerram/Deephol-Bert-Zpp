# Modified code from bert repository.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six
import tensorflow as tf


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def get_token_index(vocab_file, target_token): 
  target_index = -1
  with tf.gfile.GFile(vocab_file, "r") as reader:
    i = 0
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      if token.strip() == target_token:
        target_index = i
      i += 1     

  assert target_index != -1
  return target_index
 

def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""

  unk_index = get_token_index(vocab_file, '[UNK]')
  vocab = collections.defaultdict(lambda: unk_index)
  index = 0

  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()

      if token not in vocab:
        vocab[token] = index
        index += 1
      else:
        print("ALARM!!!!")  # Legacy

  return vocab


class WordSplitterTokenizer(object):
  def __init__(self, vocab):
    self.vocab = load_vocab(vocab)

  def tokenize(self, text, max_seq_length):
    """Tokenizes string according to lookup table."""

    text = ' '.join(['[CLS]', text.strip()])
    # Remove parentheses.
    text = text.replace('(', ' ').replace(')', ' ')
    words = text.split()

    # Truncate long terms.
    words = words[:(max_seq_length - 1)]
    words.append('[SEP]')

    ids = list(map(lambda word: self.vocab[word], words))
    mask = [1] * len(ids)

    while len(ids) < max_seq_length:
        ids.append(0)
        mask.append(0)

    return ids, mask


class TensorWordSplitter(object):
  """Extract terms/thms and tokenize based on vocab.
  Code mostly from deephol train -- extractor.py.

  Attributes:
    vocab_table: Lookup table for goal vocab embeddings.
  """

  def __init__(self, vocab_file):
    # Create vocab lookup tables from existing vocab id lists.

    self.unk_index = get_token_index(vocab_file, '[UNK]')
    self.sep_index = get_token_index(vocab_file, '[SEP]')
    # Asserts that [CLS] exists
    get_token_index(vocab_file, '[CLS]') 

    with tf.variable_scope('extractor'):
      self.vocab_table = self._vocab_table_from_file(vocab_file)

  def _vocab_table_from_file(self, filename):
    with tf.gfile.Open(filename, 'r') as f:
      keys = [s.strip() for s in f.readlines()]
      values = tf.range(len(keys), dtype=tf.int64)
      init = tf.contrib.lookup.KeyValueTensorInitializer(keys, values)
      return tf.contrib.lookup.HashTable(init, self.unk_index)

  def tokenize(self, tm, max_seq_length):
    """Tokenizes tensor string according to lookup table."""
    tm = tf.strings.join(['[CLS] ', tf.strings.strip(tm)])
    tf.logging.info("  name = %s, shape = %s" % ("tm", tm.shape))
    # Remove parentheses - they can be recovered for S-expressions.
    tm = tf.strings.regex_replace(tm, r'\(', ' ')
    tm = tf.strings.regex_replace(tm, r'\)', ' ')
    words = tf.strings.split(tm)

    # Truncate long terms.
    tf.logging.info("  name = %s, shape = %s" % ("words", words.shape))
    words = tf.sparse.slice(words, [0, 0],
                            [tf.shape(words)[0], max_seq_length])

    word_values = words.values
    id_values = tf.to_int32(self.vocab_table.lookup(word_values))
    tf.logging.info("  name = %s, shape = %s" % ("id_values", id_values.shape))
    ids = tf.SparseTensor(words.indices, id_values, words.dense_shape)
    ids = tf.sparse_tensor_to_dense(ids)

    def add_sep(tensor):
      original_shape = tf.shape(tensor)

      tensor = tf.slice(tensor, [0], [original_shape[0] - 1])
      mask = tf.not_equal(tensor, tf.zeros(tf.shape(tensor), dtype=tf.int32))
      tensor = tf.boolean_mask(tensor, mask)

      tensor = tf.concat([tensor, [self.sep_index]], axis=0)
      tensor = tf.pad(tensor, [[0, original_shape[0] - tf.shape(tensor)[0]]])

      return tensor

    ids = tf.map_fn(add_sep, ids, dtype=tf.int32)
    tf.logging.info("  name = %s, shape = %s" % ("ids", ids.shape))

    return ids
