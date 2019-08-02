# YOLOv3-EfficientNet-EffYolo
Modification of YOLOv3 by applying EfficientNet as a backbone instead of Darknet53.



Usage
Installation
Pip
pip install -r requirements.txt
Conda
conda env create -f environment.yml
conda activate yolov3-tf2
Convert pre-trained Darknet weights


# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py


##Detection
# yolov3
python detect.py --image ./data/meme.jpg

# yolov3-tiny
python detect.py --weights ./checkpoints/yolov3-tiny.tf --tiny --image ./data/street.jpg

# webcam
python detect_video.py --video 0

# video file
python detect_video.py --video path_to_file.mp4 --weights ./checkpoints/yolov3-tiny.tf --tiny





reference


1.https://github.com/zzh8829/yolov3-tf2

2.https://arxiv.org/abs/1905.11946

3.https://github.com/qubvel/efficientnet

4.https://github.com/mingxingtan/efficientnet

5.https://github.com/mingxingtan/mnasnet

6.https://github.com/tsing-cv/EfficientNet-tensorflow-eager

7.https://www.reddit.com/r/MachineLearning/comments/bumjdc/r_efficientnet_rethinking_model_scaling_for
