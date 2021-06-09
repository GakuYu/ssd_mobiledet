#!/bin/bash

DATA_PATH="test-data"
DATA_YEAR="VOC2007"
LABEL_PATH="$DATA_PATH/pascal_label_map.pbtxt"
OUTPUT_PATH="$DATA_PATH/tfrecord"

if [ ! -d "$OUTPUT_PATH" ]; then
        mkdir "$OUTPUT_PATH"
fi

SET_LIST=("train" "val" "trainval" "test")

for set_name in ${SET_LIST[@]};
do
        echo "create $set_name.tfrecord"
        python create_pascal_tf_record.py \
                --data_dir="$DATA_PATH" \
                --set=$set_name \
                --year="$DATA_YEAR" \
                --output_path="$OUTPUT_PATH/$DATA_YEAR-$set_name.record" \
                --label_map_path="$LABEL_PATH"

done
