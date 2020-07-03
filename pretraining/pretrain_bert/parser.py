import argparse


def setup_parser():
    parser_tmp = argparse.ArgumentParser()
    parser_tmp.add_argument("--session-name", default="11", help="Name of a tmux console")
    parser_tmp.add_argument("--executable", default="pretrain_bert/bert_runner.py", help="Name of executable "
                                                                                               "bert pretraining")
    parser_tmp.add_argument("--zone", default="us-central1-f", help="tpu zone")
    parser_tmp.add_argument("--tpu-name", default="bert", help="tpu name to run")
    parser_tmp.add_argument("--project-name", default="zpp-mim-1920", help="gcp project name/id in which you run tpu")
    parser_tmp.add_argument("--bert-folder", default="bert", help="a folder name to which bert was cloned")
    parser_tmp.add_argument("--voc-size", default=2000, help="vocabulary size for a tokenizer")
    parser_tmp.add_argument("--vocab-thms-ls", default="vocab_thms_ls.txt", help="vocab thms ls file path")
    parser_tmp.add_argument("--vocab-filename", default="vocab.txt", help="vocab file name")
    parser_tmp.add_argument("--bert-config-filename", default="bert_config.json", help="config for bert given by a file name")
    parser_tmp.add_argument("--checkpoints-steps", default=2500)
    parser_tmp.add_argument("--train-steps", default=500000)
    parser_tmp.add_argument("--tpu-cores", default=8)
    parser_tmp.add_argument("--eval-batch-size", default=64)
    parser_tmp.add_argument("--train-batch-size", default=128)
    parser_tmp.add_argument("--config-dump", default="config_dump.json", help="dump of a config")
    parser_tmp.add_argument("--max-predictions", default=80)
    parser_tmp.add_argument("--max-seq-length", default=512)
    parser_tmp.add_argument("--masked-lm-prob", default=0.8)
    parser_tmp.add_argument("--bucket-name", default="zpp-bucket-1920", help="a name of a bucket to get data from/to")
    parser_tmp.add_argument("--model-dir", default="bert_model")
    parser_tmp.add_argument("--gcp-model-dir", default="bert-bucket-golkarolka/bert_model11")
    parser_tmp.add_argument("--pretraining-dir", default="bert-bucket-golkarolka/pretrain_data11")
    return parser_tmp
