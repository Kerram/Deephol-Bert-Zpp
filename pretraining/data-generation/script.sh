for f in `ls ./shards/`; do python3 ./bert/create_pretraining_data.py --input_file=./shards/$f --output_file=train_dir/$f.tfrecord --vocab_file=vocab.txt; done
