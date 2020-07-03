import os


class BertConfig:
    def __init__(self,
                 bert_folder,
                 voc_size,
                 vocab_thms_file_path,
                 vocab_filename,
                 max_seq_length,
                 masked_lm_prob,
                 max_predictions,
                 pretraining_dir,
                 bucket_name,
                 gcp_model_dir,
                 model_dir,
                 train_batch_size,
                 eval_batch_size,
                 train_steps,
                 save_checkpoints_steps,
                 num_tpu_cores,
                 bert_config_file,
                 use_tpu,
                 tpu_name,
                 tpu_zone,
                 project):
        self.bert_folder = bert_folder
        self.VOC_SIZE = voc_size
        self.vocab_thms_file_path = vocab_thms_file_path
        self.VOC_FNAME = vocab_filename
        self.MAX_SEQ_LENGTH = max_seq_length
        self.MASKED_LM_PROB = masked_lm_prob
        self.MAX_PREDICTIONS = max_predictions
        self.PRETRAINING_DIR = pretraining_dir
        self.BUCKET_NAME = bucket_name
        self.GCP_MODEL_DIR = gcp_model_dir
        self.MODEL_DIR = model_dir
        self.TRAIN_BATCH_SIZE = train_batch_size
        self.EVAL_BATCH_SIZE = eval_batch_size
        self.TRAIN_STEPS = train_steps
        self.SAVE_CHECKPOINTS_STEPS = save_checkpoints_steps
        self.NUM_TPU_CORES = num_tpu_cores
        self.bert_config_file = bert_config_file
        self.USE_TPU = use_tpu
        self.TPU_NAME = tpu_name
        self.TPU_ZONE = tpu_zone
        self.PROJECT = project
        self.LEARNING_RATE = 2e-5
        self.BUCKET_PATH = "gs://{}".format(self.BUCKET_NAME)
        self.BERT_GCS_DIR = "{}/{}".format(self.BUCKET_PATH, self.GCP_MODEL_DIR)
        self.DATA_GCS_DIR = "{}/{}".format(self.BUCKET_PATH, self.PRETRAINING_DIR)
        self.VOCAB_FILE = os.path.join(self.BERT_GCS_DIR, self.VOC_FNAME)
        self.CONFIG_FILE = os.path.join(self.BERT_GCS_DIR, self.bert_config_file)
