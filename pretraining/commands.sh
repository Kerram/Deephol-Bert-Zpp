#!/usr/bin/env bash

gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/thms_ls.train .
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/vocab_thms_ls.txt .
mkdir ./shards
split -a 4 -l 5000 -d thms_ls.train ./shards/shard_
git clone https://github.com/Kerram/bert.git
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/create_pretraining_data.py bert/
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/tokenization.py bert/
gsutil -m cp -R gs://zpp-bucket-1920/bert-bucket-golkarolka/pretraining_data .
# This one is broken, but its purpose was to save now checkpoint to our bucket (I hope so).
#gsutil -m cp -r bert-bucket-golkarolka/bert_model bert-bucket-golkarolka/pretraining_data gs://zpp-bucket-1920
