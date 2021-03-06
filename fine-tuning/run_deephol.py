# Mix of bert's classifier.py and deephol's architectures.py
# Code used to run deephol training on tpu outside of deephol infrastructure

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import tensorflow as tf
import sys
import json

sys.path.append("../bert")
sys.path.append("../../bert")

import modeling
import tokenization
import optimization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("configuration_dir", None, "Path to the configuration directory.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_export", False, "Whether to export the model.")

flags.DEFINE_bool("use_tpu", True, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool(
    "infinite_eval",
    False,
    "If infinite we will run eval on currently newest checkpoint."
    "If finite than we will run evaluation only once.",
)


def get_tac_labels():
    return [str(i) for i in range(41)]


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "goal_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "thm_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "tac_ids": tf.FixedLenFeature([], tf.int64),
        "is_negative": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        "goal_input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "thm_input_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=400000)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        )

        return d

    return input_fn


def build_predict_fake_input_fn(input_file, seq_length, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator.
    This function differs from the one used for train and valid sets, because
    we add goal and thm strings as additional features."""

    name_to_features = {
        "goal_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "thm_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "tac_ids": tf.FixedLenFeature([], tf.int64),
        "is_negative": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        "goal_str": tf.FixedLenFeature((), tf.string, default_value=""),
        "thm_str": tf.FixedLenFeature((), tf.string, default_value=""),
        "goal_input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "thm_input_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        d = tf.data.TFRecordDataset(input_file)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        )

        return d

    return input_fn


def bert_encoding(
    is_training,
    bert_config,
    input_ids,
    input_mask,
    segment_ids,
    use_one_hot_embeddings,
):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )

    output = model.get_pooled_output()

    return output


def goal_encoding(
    is_training, bert_config, input_ids, input_mask, use_one_hot_embeddings,
):
    segment_ids = tf.to_int32(tf.zeros(tf.shape(input_ids)))

    with tf.variable_scope("goal", reuse=False):
        # output shape: [batch_size, hidden_size]
        goal_net = bert_encoding(
            is_training,
            bert_config,
            input_ids,
            input_mask,
            segment_ids,
            use_one_hot_embeddings,
        )

    tf.add_to_collection("goal_net", goal_net)

    return goal_net


def thm_encoding(
    is_training, bert_config, input_ids, input_mask, use_one_hot_embeddings,
):
    segment_ids = tf.to_int32(tf.zeros(tf.shape(input_ids)))

    with tf.variable_scope("thm", reuse=False):
        # output shape: [batch_size, hidden_size]
        thm_net = bert_encoding(
            is_training,
            bert_config,
            input_ids,
            input_mask,
            segment_ids,
            use_one_hot_embeddings,
        )

    tf.add_to_collection("thm_net", thm_net)

    return thm_net


def tactic_classifier(
    goal_net, is_training, tac_ids, num_tac_labels, is_real_example, hidden_size
):
    tf.add_to_collection("tactic_net", goal_net)

    # Adding 3 dense layers with dropout like in deephol
    # with tf.variable_scope("loss"):
    if is_training:
        goal_net = tf.nn.dropout(goal_net, rate=(1 - 0.7))
    goal_net = tf.layers.dense(
        goal_net, hidden_size, activation=tf.nn.relu, name="tac_dense1"
    )

    if is_training:
        goal_net = tf.nn.dropout(goal_net, rate=(1 - 0.7))
    goal_net = tf.layers.dense(
        goal_net, hidden_size, activation=tf.nn.relu, name="tac_dense2"
    )

    if is_training:
        goal_net = tf.nn.dropout(goal_net, rate=(1 - 0.7))
    tac_logits = tf.layers.dense(
        goal_net, num_tac_labels, activation=None, name="tac_dense3"
    )

    tf.add_to_collection("tactic_logits", tac_logits)

    # Compute the log loss for the tactic logits.
    log_prob_tactic = tf.losses.sparse_softmax_cross_entropy(
        logits=tac_logits, labels=tac_ids, weights=is_real_example
    )

    tf.losses.add_loss(log_prob_tactic)

    tac_probabilities = tf.nn.softmax(tac_logits, axis=-1)

    return log_prob_tactic, tac_logits, tac_probabilities


def pairwise_scorer(net, is_negative_labels, is_real_example):
    # [batch_size, 1]
    par_logits = tf.layers.dense(net, 1, activation=None)

    tf.add_to_collection("pairwise_score", par_logits)

    # scalar
    ce_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.expand_dims(
            (1 - tf.to_float(is_negative_labels)) * is_real_example, 1
        ),
        logits=par_logits,
        reduction=tf.losses.Reduction.SUM,
    )

    tf.losses.add_loss(ce_loss * 0.5)

    return ce_loss, par_logits


def create_model(
    bert_config,
    goal_input_ids,
    goal_input_mask,
    thm_input_ids,
    thm_input_mask,
    tac_ids,
    num_tac_labels,
    is_negative_labels,
    is_training,
    is_real_example,
    use_one_hot_embeddings,
    hidden_size,
):
    with tf.variable_scope("encoder"):
        with tf.variable_scope("dilated_cnn_pairwise_encoder"):
            goal_net = goal_encoding(
                is_training,
                bert_config,
                goal_input_ids,
                goal_input_mask,
                use_one_hot_embeddings,
            )
            thm_net = thm_encoding(
                is_training,
                bert_config,
                thm_input_ids,
                thm_input_mask,
                use_one_hot_embeddings,
            )

            # Concatenate theorem encoding, goal encoding, their dot product.
            # This attention-style concatenation performed well in language models.
            # Output shape: [batch_size, 3 * hidden_size]
            net = tf.concat([goal_net, thm_net, tf.multiply(goal_net, thm_net)], -1)

            if is_training:
                net = tf.nn.dropout(net, rate=(1 - 0.7))
            net = tf.layers.dense(net, hidden_size, activation=tf.nn.relu)
            if is_training:
                net = tf.nn.dropout(net, rate=(1 - 0.7))
            net = tf.layers.dense(net, hidden_size, activation=tf.nn.relu)
            if is_training:
                net = tf.nn.dropout(net, rate=(1 - 0.7))
            net = tf.layers.dense(net, hidden_size, activation=tf.nn.relu)
            tf.add_to_collection("thm_goal_fc", net)

    with tf.variable_scope("classifier"):
        (tac_loss, tac_logits, tac_probabilities,) = tactic_classifier(
            goal_net, is_training, tac_ids, num_tac_labels, is_real_example, hidden_size
        )

    with tf.variable_scope("pairwise_scorer"):
        (par_loss, par_logits) = pairwise_scorer(
            net, is_negative_labels, is_real_example
        )

    total_loss = (0.5 * par_loss) + tac_loss

    return (
        total_loss,
        tac_logits,
        tac_probabilities,
        par_loss,
        par_logits,
        tac_loss,
    )


def _pad_up_to(value, size, axis, name=None):
    """Pad a tensor with zeros on the right along axis to a least the given size.

  Args:
    value: Tensor to pad.
    size: Minimum size along axis.
    axis: A nonnegative integer.
    name: Optional name for this operation.

  Returns:
    Padded value.
  """
    with tf.name_scope(name, "pad_up_to") as name:
        value = tf.convert_to_tensor(value, name="value")
        axis = tf.convert_to_tensor(axis, name="axis")
        need = tf.nn.relu(size - tf.shape(value)[axis])
        ids = tf.stack([tf.stack([axis, 1])])
        paddings = tf.sparse_to_dense(ids, tf.stack([tf.rank(value), 2]), need)
        padded = tf.pad(value, paddings, name=name)
        # Fix shape inference
        axis = tf.contrib.util.constant_value(axis)
        shape = value.get_shape()
        if axis is not None and shape.ndims is not None:
            shape = shape.as_list()
            shape[axis] = None
            padded.set_shape(shape)
        return padded


def update_features_using_deephol(features, max_seq_length, vocab_file):
    tensor_tokenizer = tokenization.TensorWordSplitter(vocab_file=vocab_file)

    with tf.variable_scope("extractor"):
        tf.logging.info("********** Tokenization of goal in Holist ********")
        tf.add_to_collection("goal_string", features["goal_str"])

        goal = tensor_tokenizer.tokenize(features["goal_str"], max_seq_length)

        goal_input_mask = tf.ones(tf.shape(goal))
        goal = _pad_up_to(goal, max_seq_length, 1)
        goal_input_mask = _pad_up_to(goal_input_mask, max_seq_length, 1)

        features["goal_input_ids"] = goal
        features["goal_input_mask"] = goal_input_mask

        tf.logging.info("********** Tokenization of theorem in Holist ********")
        tf.add_to_collection("thm_string", features["thm_str"])

        thm = tensor_tokenizer.tokenize(features["thm_str"], max_seq_length)

        thm_input_mask = tf.ones(tf.shape(thm))
        thm = _pad_up_to(thm, max_seq_length, 1)
        thm_input_mask = _pad_up_to(thm_input_mask, max_seq_length, 1)

        features["thm_input_ids"] = thm
        features["thm_input_mask"] = thm_input_mask


# Kuba's hack
class Remover:
    prefixes = []
    keys_list = []

    def __init__(self, sl):
        self.prefixes = sl
        self.keys_list = []

    def get(self, key):
        pref = ""
        for p in self.prefixes:
            if key.startswith(p):
                pref = p
        assert pref
        self.keys_list.append(key)
        return key[len(pref) :]

    def keys(self):
        return self.keys_list


def model_fn_builder(
    bert_config,
    num_tac_labels,
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
    use_one_hot_embeddings,
    max_seq_length,
    do_export,
    vocab_file,
):
    def model_fn(features, labels, mode, params):

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        if do_export:  # HOList has different expectation towards input features.
            update_features_using_deephol(features, max_seq_length, vocab_file)

        tf.logging.info("*** Features after deephol ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        goal_input_ids = features["goal_input_ids"]
        thm_input_ids = features["thm_input_ids"]
        goal_input_mask = features["goal_input_mask"]
        thm_input_mask = features["thm_input_mask"]
        tac_ids = features["tac_ids"]
        is_negative = features["is_negative"]

        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(tac_ids), dtype=tf.float32)

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        (
            total_loss,
            tac_logits,
            tac_probabilities,
            par_loss,
            par_logits,
            tac_loss,
        ) = create_model(
            bert_config=bert_config,
            goal_input_ids=goal_input_ids,
            goal_input_mask=goal_input_mask,
            thm_input_ids=thm_input_ids,
            thm_input_mask=thm_input_mask,
            tac_ids=tac_ids,
            num_tac_labels=num_tac_labels,
            is_real_example=is_real_example,
            is_training=is_training,
            is_negative_labels=is_negative,
            use_one_hot_embeddings=use_one_hot_embeddings,
            hidden_size=bert_config.hidden_size,
        )

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        scaffold_fn = None

        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            if use_tpu:

                def tpu_scaffold():
                    tf.train.warm_start(
                        init_checkpoint,
                        "encoder/dilated_cnn_pairwise_encoder/thm/bert*",
                        None,
                        Remover(["encoder/dilated_cnn_pairwise_encoder/thm/"]),
                    )

                    tf.train.warm_start(
                        init_checkpoint,
                        "encoder/dilated_cnn_pairwise_encoder/goal/bert*",
                        None,
                        Remover(["encoder/dilated_cnn_pairwise_encoder/goal/"]),
                    )

                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold

            elif do_export:
                tf.train.warm_start(init_checkpoint, "encoder/*")

                tf.train.warm_start(init_checkpoint, "classifier/*")

                tf.train.warm_start(init_checkpoint, "pairwise_scorer/*")

            else:
                tf.train.warm_start(
                    init_checkpoint,
                    "encoder/dilated_cnn_pairwise_encoder/thm/bert*",
                    None,
                    Remover(["encoder/dilated_cnn_pairwise_encoder/thm/"]),
                )

                tf.train.warm_start(
                    init_checkpoint,
                    "encoder/dilated_cnn_pairwise_encoder/goal/bert*",
                    None,
                    Remover(["encoder/dilated_cnn_pairwise_encoder/goal/"]),
                )

        # These logs could not contain all variables!
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info(
                "  name = %s, shape = %s%s", var.name, var.shape, init_string
            )

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = tf.losses.get_total_loss()

            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu
            )

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn,
            )

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(
                tac_logits,
                par_logits,
                tac_ids,
                is_negative,
                is_real_example,
                tac_loss,
                par_loss,
            ):
                tac_predictions = tf.argmax(tac_logits, axis=-1, output_type=tf.int32)

                REWRITE_ID = 5.0
                chosen_tac = tf.metrics.mean(tac_predictions)
                rewrite_acc = tf.metrics.accuracy(
                    labels=tac_predictions,
                    predictions=tf.ones(tf.shape(tac_predictions)) * REWRITE_ID,
                )

                is_negative = tf.to_float(is_negative)

                # Tactic accuracy
                tac_accuracy = tf.metrics.accuracy(
                    labels=tac_ids,
                    predictions=tac_predictions,
                    weights=is_real_example * (1 - is_negative),
                )

                # Top 5 tactics accuracy
                topk_preds = tf.to_float(tf.nn.in_top_k(tac_logits, tac_ids, 5))
                tac_topk_accuracy = tf.metrics.mean(
                    values=topk_preds, weights=is_real_example * (1 - is_negative)
                )

                par_pred = tf.sigmoid(par_logits)
                pos_guess = tf.to_float(tf.greater(par_pred, 0.5))
                neg_guess = tf.to_float(tf.less(par_pred, 0.5))

                pos_acc = tf.metrics.mean(
                    values=pos_guess, weights=is_real_example * (1 - is_negative)
                )
                neg_acc = tf.metrics.mean(
                    values=neg_guess, weights=is_real_example * is_negative
                )

                tot_loss = (0.5 * par_loss) + tac_loss
                tot_loss = tf.metrics.mean(tot_loss)

                tac_loss = tf.metrics.mean(tac_loss)
                par_loss = tf.metrics.mean(par_loss)

                res = {
                    "pos_acc": pos_acc,
                    "neg_acc": neg_acc,
                    "tac_accuracy": tac_accuracy,
                    "tac_topk_accuracy": tac_topk_accuracy,
                    "tac_loss": tac_loss,
                    "par_loss": par_loss,
                    "total_loss": tot_loss,
                    "chosen_tactic (mean)": chosen_tac,
                    "rewrite accuracy": rewrite_acc,
                }

                return res

            eval_metrics = (
                metric_fn,
                [
                    tac_logits,
                    par_logits,
                    tac_ids,
                    is_negative,
                    is_real_example,
                    tf.ones(params["batch_size"]) * tac_loss,
                    tf.ones(params["batch_size"]) * par_loss,
                ],
            )

            loss = tf.losses.get_total_loss()
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=loss, eval_metrics=eval_metrics,
            )

        else:
            preds = {
                "tac_probabilities": tac_probabilities,
                "is_negative_prob": tf.sigmoid(par_logits),
            }

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=preds,)

        return output_spec

    return model_fn


def count_records(file_path):
    ans = 0
    for _ in tf.python_io.tf_record_iterator(file_path):
        ans += 1
    return ans


def add_conf_dir_to_paths(configuration, config_dir):
    configuration["general"]["vocab_file"] = os.path.join(
        config_dir, configuration["general"]["vocab_file"]
    )
    configuration["general"]["bert_config_file"] = os.path.join(
        config_dir, configuration["general"]["bert_config_file"]
    )


def main(_):
    csv.field_size_limit(sys.maxsize)
    tf.logging.set_verbosity(tf.logging.INFO)

    with open(os.path.join(FLAGS.configuration_dir, "config.json")) as f:
        configuration = json.load(f)
        add_conf_dir_to_paths(configuration, FLAGS.configuration_dir)

    vocab_file = configuration["general"]["vocab_file"]
    bert_config_file = configuration["general"]["bert_config_file"]
    max_seq_length = configuration["general"]["max_seq_length"]
    gcp_project = configuration["general"]["gcp_project"]
    num_tpu_cores = configuration["general"]["num_tpu_cores"]
    tpu_zone = configuration["general"]["tpu_zone"]

    train_batch_size = configuration["fine-tuning"]["train_batch_size"]
    eval_batch_size = configuration["fine-tuning"]["eval_batch_size"]
    warmup_proportion = configuration["fine-tuning"]["warmup_proportion"]
    num_train_epochs = configuration["fine-tuning"]["num_train_epochs"]
    learning_rate = configuration["fine-tuning"]["learning_rate"]
    save_checkpoint_steps = configuration["fine-tuning"]["save_checkpoint_steps"]
    iterations_per_loop = configuration["fine-tuning"]["iterations_per_loop"]

    input_data_dir = configuration["fine-tuning"]["data-generation"]["output_data_dir"]

    if FLAGS.do_export:
        init_checkpoint = tf.train.latest_checkpoint(
            configuration["fine-tuning"]["model_dir"]
        )
        model_dir = configuration["fine-tuning"]["export_dir"]
    else:
        init_checkpoint = tf.train.latest_checkpoint(
            configuration["pretraining"]["model_dir"]
        )
        model_dir = configuration["fine-tuning"]["model_dir"]

    train_file = os.path.join(input_data_dir, "train.tf_record")
    test_file = os.path.join(input_data_dir, "test.tf_record")

    if FLAGS.infinite_eval:
        tpu_name = configuration["fine-tuning"]["debug_tpu_name"]
        eval_file = os.path.join(input_data_dir, "valid_mini.tf_record")
    else:
        tpu_name = configuration["fine-tuning"]["tune_tpu_name"]
        eval_file = os.path.join(input_data_dir, "valid.tf_record")

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_export:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_export` must be True."
        )

    if (FLAGS.do_train or FLAGS.do_eval) and FLAGS.do_export:
        raise ValueError(
            "It is not a good idea to export and train/eval, because it will work on CPU."
        )

    if FLAGS.do_export and FLAGS.use_tpu:
        raise ValueError("You cannot export model using TPU.")

    if not FLAGS.do_eval and FLAGS.infinite_eval:
        raise ValueError(
            "You cannot do eval infinitely if you are not doing it at all. Set `do_eval` to True."
        )

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d"
            % (max_seq_length, bert_config.max_position_embeddings)
        )

    tf.gfile.MakeDirs(model_dir)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name, zone=tpu_zone, project=gcp_project
        )

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=model_dir,
        save_checkpoints_steps=save_checkpoint_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host,
        ),
    )

    tac_labels = get_tac_labels()

    train_examples_count = 0
    num_train_steps = None
    num_warmup_steps = None

    tf.logging.info("Preparation completed!")

    if FLAGS.do_train:
        train_examples_count = count_records(train_file)
        num_train_steps = int(
            train_examples_count / train_batch_size * num_train_epochs
        )
        num_warmup_steps = int(num_train_steps * warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_tac_labels=len(tac_labels),
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        max_seq_length=max_seq_length,
        do_export=FLAGS.do_export,
        vocab_file=vocab_file,
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=8,  # Does not matter.
    )

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", train_examples_count)
        tf.logging.info("  Batch size = %d", train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_seq_length,
            is_training=True,
            drop_remainder=True,
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples_count = count_records(eval_file)

        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. Fake examples should already be in the file.
            if eval_examples_count % eval_batch_size != 0:
                raise ValueError(
                    "Samples from eval file not padded to eval batch size!"
                )

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert eval_examples_count % eval_batch_size == 0
            eval_steps = int(eval_examples_count // eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
        )

        def run_eval():
            tf.logging.info("***** Running evaluation *****")
            tf.logging.info(
                "  Num examples = %d (with assumed padding (if running on TPU))",
                eval_examples_count,
            )
            tf.logging.info("  Batch size = %d", eval_batch_size)

            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

            output_eval_file = os.path.join(model_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        if FLAGS.infinite_eval:
            while True:
                run_eval()
        else:
            run_eval()

    if FLAGS.do_export:
        feature_spec = {
            "goal_input_ids": tf.placeholder(
                dtype=tf.int32, shape=[None, max_seq_length]
            ),
            "thm_input_ids": tf.placeholder(
                dtype=tf.int32, shape=[None, max_seq_length]
            ),
            "tac_ids": tf.placeholder(dtype=tf.int32, shape=[None]),
            "is_negative": tf.placeholder(dtype=tf.int32, shape=[None]),
            "is_real_example": tf.placeholder(dtype=tf.int32, shape=[None]),
            "goal_str": tf.placeholder(dtype=tf.string, shape=[None]),
            "thm_str": tf.placeholder(dtype=tf.string, shape=[None]),
            "thm_input_mask": tf.placeholder(
                dtype=tf.int32, shape=[None, max_seq_length]
            ),
            "goal_input_mask": tf.placeholder(
                dtype=tf.int32, shape=[None, max_seq_length]
            ),
        }
        label_spec = {}
        build_input = tf.contrib.estimator.build_raw_supervised_input_receiver_fn
        input_receiver_fn = build_input(feature_spec, label_spec)

        predict_input_fn = build_predict_fake_input_fn(
            input_file=test_file, seq_length=max_seq_length, drop_remainder=False,
        )

        # Export final model
        tf.logging.info("*****  Starting to export model.   *****")
        save_hook = tf.train.CheckpointSaverHook(model_dir, save_secs=1)
        result = estimator.predict(input_fn=predict_input_fn, hooks=[save_hook])
        # As result is a generator we want to force it to calculate first value (and save checkpoint needed for export).
        for _ in result:
            break

        estimator._export_to_tpu = False
        estimator.export_savedmodel(
            export_dir_base=os.path.join(
                model_dir, "export/best_exporter"
            ),  # It is not best exporter,
            serving_input_receiver_fn=input_receiver_fn,
        )  # but HOList expects it to be.


if __name__ == "__main__":
    flags.mark_flag_as_required("configuration_dir")
    tf.app.run()
