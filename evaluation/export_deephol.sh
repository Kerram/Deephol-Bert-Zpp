sudo python3 run_deephol.py --bert_config_file=bert/model/bert_config.json --vocab_file=bert/model/vocab.txt \
--do_train=False --do_eval=False --do_export=True --use_tpu=False \
\
--init_checkpoint=gs://zpp-bucket-1920/tpu-fine-tune/models/beta_bert/model.ckpt-903935 \
--output_dir=gs://zpp-bucket-1920/tpu-fine-tune/exported/beta_bert \
--test_file=gs://zpp-bucket-1920/tpu-fine-tune/data/preprocessed/test_with_mask.tf_record
