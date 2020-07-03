mkdir augmented
mkdir raw

gsutil cp gs://zpp-bucket-1920/data-extraction-csv/test.csv augmented/
gsutil cp gs://zpp-bucket-1920/data-extraction-csv/train.csv augmented/
gsutil cp gs://zpp-bucket-1920/data-extraction-csv/valid.csv augmented/

gsutil cp gs://zpp-bucket-1920/deephol-data-json/test.json raw/
gsutil cp gs://zpp-bucket-1920/deephol-data-json/train.json raw/ 
gsutil cp gs://zpp-bucket-1920/deephol-data-json/valid.json raw/
