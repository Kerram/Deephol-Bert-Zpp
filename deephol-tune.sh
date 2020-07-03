sudo python3 run_deephol.py --vocab_file=bert/model/vocab.txt --bert_config_file=bert/model/bert_config.json \
--do_train=True --do_eval=True --do_export=False \
\
--learning_rate=1e-6 \
--init_checkpoint=gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model11/model.ckpt-500000 \
--output_dir=gs://zpp-bucket-1920/tpu-fine-tune/models/lr/lr-6 \
--use_tpu=True --tpu_name=bert3 --num_train_epochs=3.0 \
--train_file=gs://zpp-bucket-1920/tpu-fine-tune/data/preprocessed/train_with_mask.tf_record \
--eval_file=gs://zpp-bucket-1920/tpu-fine-tune/data/preprocessed/valid_with_mask.tf_record