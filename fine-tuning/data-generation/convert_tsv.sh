python3 convert_to_tfrecord.py --configuration_dir=../../configuration --file_name=test --set_type=test & \
python3 convert_to_tfrecord.py --configuration_dir=../../configuration --file_name=valid --set_type=valid & \
python3 convert_to_tfrecord.py --configuration_dir=../../configuration --file_name=valid_mini --set_type=valid & \
python3 convert_to_tfrecord.py --configuration_dir=../../configuration --file_name=half_train --set_type=train & \
python3 convert_to_tfrecord.py --configuration_dir=../../configuration --file_name=train --set_type=train
