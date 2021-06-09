## SSD MobileDet
基于 [tensorflow models](https://github.com/tensorflow/models) 的目标检测模型 SSD MobileDet 训练与部署
[TensorFlow 1 Detection Model Zoo Mobile models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models)
[ssdlite mobiledet coco model](http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz)

### tensorflow models 源码获取
    git clone --depth=1 https://github.com/tensorflow/models.git

### Docker 训练环境搭建
    docker build -f Dockerfile -t tf_object_detection:1.15.2-gpu .
    docker run -dit --gpus all --name ssd_mobiledet -w /tf/ssd_mobiledet -p 6006:6006 -v $PWD:/tf/ssd_mobiledet tf_object_detection:1.15.2-gpu

### 训练数据准备
    # 使用 lableImg 标记数据，并存放到 {data_path} 文件夹下
    # test-data 文件夹为示例数据
    # test-data/images 为图片文件夹
    # test-data/annotations 为标注文件夹
    # test-data/pascal_label_map.pbtxt 为标签文件，id 从1开始(0 表示背景)，内容格式为
    item {
        id: 1
        name: 'aeroplane'
    }
    item {
        id: 2
        name: 'bicycle'
    }

### 数据转换
    # 修改 prepare_pascal.py 中的 data_path = 'test-data'
    # 运行 prepare_pascal.py 生成 pascal 数据
    docker exec -it ssd_mobiledet bash
    python prepare_pascal.py

### 生成 tfrecord 文件
    # 修改 create_pascal_tf_record.sh 中的 
    #   DATA_PATH="test-data"
    #   src_img_path = os.path.join(data_path, 'images')
    #   src_anno_path = os.path.join(data_path, 'annotations')
    # 运行 create_pascal_tf_record.sh 生成 tfrecord 文件
    docker exec -it ssd_mobiledet bash
    bash create_pascal_tf_record.sh

### 修改 pipeline.config 文件
    # 修改 label_map_path 和 input_path 中的路径
    train_input_reader: {
        label_map_path: "/tf/ssd_mobiledet/test-data/pascal_label_map.pbtxt"
        shuffle: true
        tf_record_input_reader {
            input_path: "/tf/ssd_mobiledet/test-data/tfrecord/VOC2007-train.record"
        }
    }
    eval_input_reader: {
        label_map_path: "/tf/ssd_mobiledet/test-data/pascal_label_map.pbtxt"
        shuffle: true
        num_epochs: 1
        tf_record_input_reader {
            input_path: "/tf/ssd_mobiledet/test-data/tfrecord/VOC2007-val.record"
        }
    }

### 运行训练
    docker exec -it ssd_mobiledet bash
    # 指定 gpu 运行，需要在命令前加入 CUDA_VISIBLE_DEVICES=gpu 序号，-1 表示不使用 gpu
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --pipeline_config_path=ssdlite_mobiledet_pipeline.config \
        --model_dir=output \
        --alsologtostderr

### TensorBoard
    tensorboard --logdir output --host 0.0.0.0

### 导出模型
    docker exec -it ssd_mobiledet bash
    # pb: 
    #   checkpoint  frozen_inference_graph.pb  model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta
    #   pipeline.config  saved_model/saved_model.pb saved_model/variables/
    python -m object_detection.export_inference_graph \
        --pipeline_config_path=ssdlite_mobiledet_pipeline.config \
        --trained_checkpoint_prefix=output/model.ckpt-0 \
        --output_directory=frozen_model

### 转换为 TFLite
    docker exec -it ssd_mobiledet bash
    # 生成 tflite graph pb
    # tflite graph: 
    #   tflite_graph.pb  tflite_graph.pbtxt
    python -m object_detection.export_tflite_ssd_graph \
        --pipeline_config_path=ssdlite_mobiledet_pipeline.config \
        --trained_checkpoint_prefix=output/model.ckpt-0 \
        --output_directory=frozen_model_tflite \
        --max_detections=10 \
        --add_postprocessing_op=true
    
    # 如果需要在手机端使用 gpu 则 quantize=False
    python converter_tflite.py \
        --pb_path=frozen_model_tflite/tflite_graph.pb \
        --save_dir=frozen_model_tflite \
        --quantize=True

### 推理
    docker exec -it ssd_mobiledet bash
    # 可以下载上文中提到的 ssdlite mobiledet coco model 进行测试
    wget http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz
    # 解压
    tar -xzvf ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz
    # 生成 pb 文件
    python -m object_detection.export_inference_graph \
        --pipeline_config_path=ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/pipeline.config \
        --trained_checkpoint_prefix=ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/model.ckpt-400000 \
        --output_directory=ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen_model
    # 修改 inference.py 中的 
    #   pb_path = 'ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen_model/frozen_inference_graph.pb'
    #   image_path = 'test-data/kite.jpg'
    #   label_map_path = 'test-data/mscoco_label_map.txt'
    #   result_img_path = 'test-data/result.jpg'
    # 运行
    CUDA_VISIBLE_DEVICES=-1 python inference.py
    # 检测结果将输出到 result_img_path 中

    # 生成 tflite graph pb 文件
    python -m object_detection.export_tflite_ssd_graph \
        --pipeline_config_path=ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/pipeline.config \
        --trained_checkpoint_prefix=ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/model.ckpt-400000 \
        --output_directory=ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen_model_tflite \
        --max_detections=100 \
        --add_postprocessing_op=true
    
    # 生成 tflite 文件
    python converter_tflite.py \
        --pb_path=ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen_model_tflite/tflite_graph.pb \
        --save_dir=ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen_model_tflite \
        --quantize=False
    # 修改 inference_tflite.py 中的 
    #   tflite_path = '/tf/ssd_mobiledet/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen_model_tflite/model.tflite'
    #   is_quantized = False
    #   image_path = 'test-data/kite.jpg'
    #   label_map_path = 'test-data/mscoco_label_map.txt'
    #   result_img_path = 'test-data/result_tflite.jpg'
    # 运行
    CUDA_VISIBLE_DEVICES=-1 python inference_tflite.py
    # 检测结果将输出到 result_img_path 中

