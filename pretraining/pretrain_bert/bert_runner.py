import os
import sys
import json
import nltk
import random
import logging
import tensorflow as tf
import sentencepiece as spm
from glob import glob
from tensorflow.keras.utils import Progbar

sys.path.append(".")
sys.path.append("..")
sys.path.append("bert")

from bert_config import BertConfig
from parser import setup_parser
from dict_helper import AttrDict
from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder


class BertRunner:
    def __init__(self, config):
        self.config = config
        self.bert_base_config = {
          "attention_probs_dropout_prob": 0.1,
          "directionality": "bidi",
          "hidden_act": "gelu",
          "hidden_dropout_prob": 0.1,
          "hidden_size": 768,
          "initializer_range": 0.02,
          "intermediate_size": 3072,
          "max_position_embeddings": 512,
          "num_attention_heads": 12,
          "num_hidden_layers": 12,
          "pooler_fc_size": 768,
          "pooler_num_attention_heads": 12,
          "pooler_num_fc_layers": 3,
          "pooler_size_per_head": 128,
          "pooler_type": "first_token_transform",
          "type_vocab_size": 2,
          "vocab_size": self.config.VOC_SIZE
        }

    def setup_logger(self):
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s :  %(message)s')
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        log.handlers = [sh]
        return log

    def setup_vocab(self):
        bert_vocab = []
        with open(self.config.vocab_thms_file_path, 'r') as f:
            for line in f:
                bert_vocab.append(line.rstrip('\n'))

        ctrl_symbols = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "(", ")"]
        bert_vocab = ctrl_symbols + bert_vocab
        bert_vocab += ["[UNUSED_{}]".format(i) for i in range(self.config.VOC_SIZE - len(bert_vocab))]

        with open(self.config.VOC_FNAME, "w") as fo:
            for token in bert_vocab:
                fo.write(token + "\n")

        tf.gfile.MkDir(self.config.MODEL_DIR)

        with open("{}/{}".format(self.config.MODEL_DIR, self.config.bert_config_file), "w") as fo:
            json.dump(self.bert_base_config, fo, indent=2)

        with open("{}/{}".format(self.config.MODEL_DIR, self.config.VOC_FNAME), "w") as fo:
            for token in bert_vocab:
                fo.write(token + "\n")

    def setup_run(self, use_checkpoint, log):
        if use_checkpoint:
            INIT_CHECKPOINT = tf.train.latest_checkpoint(self.config.BERT_GCS_DIR)
        else:
            INIT_CHECKPOINT = None

        bert_config = modeling.BertConfig.from_json_file(self.config.CONFIG_FILE)
        input_files = tf.gfile.Glob(os.path.join(self.config.DATA_GCS_DIR, '*tfrecord'))

        log.info("Using checkpoint: {}".format(INIT_CHECKPOINT))
        log.info("Using {} data shards".format(len(input_files)))

        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=INIT_CHECKPOINT,
            learning_rate=self.config.LEARNING_RATE,
            num_train_steps=self.config.TRAIN_STEPS,
            num_warmup_steps=10,
            use_tpu=self.config.USE_TPU,
            use_one_hot_embeddings=True)

        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            self.config.TPU_NAME,
            zone=self.config.TPU_ZONE,
            project=self.config.PROJECT)

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=self.config.BERT_GCS_DIR,
            save_checkpoints_steps=self.config.SAVE_CHECKPOINTS_STEPS,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.config.SAVE_CHECKPOINTS_STEPS,
                num_shards=self.config.NUM_TPU_CORES,
                per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self.config.USE_TPU,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self.config.TRAIN_BATCH_SIZE,
            eval_batch_size=self.config.EVAL_BATCH_SIZE)

        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            max_predictions_per_seq=self.config.MAX_PREDICTIONS,
            is_training=True)

        estimator.train(input_fn=train_input_fn, max_steps=self.config.TRAIN_STEPS)

    def run(self, use_checkpoint=True):
        log = self.setup_logger()
        self.setup_vocab()
        self.setup_run(use_checkpoint, log)


args = setup_parser().parse_args()
if args.config_dump is not None:
    f = open(args.config_dump, "r")
    args = AttrDict(json.loads(f.read()))
bert_config = BertConfig(
        bert_folder=args.bert_folder,
        voc_size=args.voc_size,
        vocab_thms_file_path=args.vocab_thms_ls,
        vocab_filename=args.vocab_filename,
        max_seq_length=args.max_seq_length,
        masked_lm_prob=args.masked_lm_prob,
        max_predictions=args.max_predictions,
        pretraining_dir=args.pretraining_dir,
        bucket_name=args.bucket_name,
        model_dir=args.model_dir,
        gcp_model_dir=args.gcp_model_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        train_steps=args.train_steps,
        save_checkpoints_steps=args.checkpoints_steps,
        num_tpu_cores=args.tpu_cores,
        bert_config_file=args.bert_config_filename,
        use_tpu=True,
        tpu_name=args.tpu_name,
        tpu_zone=args.zone,
        project=args.project_name)
runner = BertRunner(bert_config)
runner.run()
