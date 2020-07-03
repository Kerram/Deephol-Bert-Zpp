sudo python3 run_deephol.py --data_dir=bert/data \
--vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/tpu-fine-tune/models/beta_bert \
--num_train_epochs=3.0 --bert_config_file=bert/model/bert_config.json \
--init_checkpoint=gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model11/model.ckpt-500000 \
--max_seq_length=512 --do_train=True --do_eval=True --do_predict=False --do_export=False --use_tpu=True --tpu_name=bert10 \
--gcp_project=zpp-mim-1920 --tpu_zone=us-central1-f --num_tpu_cores 8 --save_checkpoints_steps 10000 \
--train_file=gs://zpp-bucket-1920/tpu-fine-tune/data/preprocessed/train_with_mask.tf_record \
--eval_file=gs://zpp-bucket-1920/tpu-fine-tune/data/preprocessed/valid_with_mask.tf_record