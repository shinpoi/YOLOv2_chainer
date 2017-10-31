echo "Training Start"
rm ./log_*

echo "RUN: darknet19_train.py"
python darknet19_train.py >> log_yolov2_train.log

echo "RUN: darknet19_448_train.py" 
python darknet19_448_train.py >> log_darknet19_448_train.log

echo "RUN: yolov2_train.py"
python yolov2_train.py >> log_yolov2_train.log
