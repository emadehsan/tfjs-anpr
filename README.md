# Automatic Number Plate Recognition on web with Tensorflow.js

[STATUS: Model to detect number plates has been trained & can be [downloaded](./weights/). Next update will include running it on web using TF.js]

* These steps can be followed to train Yolo on any custom object detection dataset and run it on web
* Also, the provided model can be converted to be used on other platforms (mobile, desktop) too

This project uses [Darknet by AlexeyAB]() the framework behind [Yolo - Object detection framework]() to train a model on custom dataset to recognize objects of our choice.


Following steps describe how to train Yolo on your custom dataset. Must work for other Yolo models too.

(Note: There are a lot of small details in configuration and setup workflow that must be followed. I've tried to make this file as readable as possible. You are welcome to update it for better readablity.)

## 1. Dataset Setup
The dataset must be annotated with bounding boxes. You can [annotate your dataset using Yolo Mark](https://github.com/AlexeyAB/Yolo_mark).

Download your dataset and save under a dir `NumberPlates` (`/content/NumberPlates`)in our case. A directory 

Yolo requires that the data inside dataset dir (`NumberPlates`) must be in following format
```txt
image_1.jpg
image_1.txt
image_2.jpg
image_2.txt
image_3.jpg
image_3.txt
```

Where `image_1.jpg` is an image file... a photo of some object. Other image formats `png`, `jpeg` must also work.

*IMPORTANT*
The `image_1.txt` accompained by the image file must be of following format:
```<object-class> <x_center> <y_center> <width> <height>```

* `<object-class>` is id of the class that appears in `image_1.jpg` photo. If total number of classes (object types) that you want to detect is `numClasses`. Then `<object-class>` is between `0` and `numClasses - 1`.

* `<x_center> <y_center> <width> <height>` - float values relative to width and height of image, it can be equal from (0.0 to 1.0]

* For example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`

* Attention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner)


For example, for `image_1.jpg` you create `image_1.txt` containing:
```txt
1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
```


## 2. Project setup
Download Darknet by AlexeyAB
```sh
git clone https://github.com/AlexeyAB/darknet
```

Compile the project to use GPU. Original instructions: [ [Windows](widows instructions), [Linux](linux instructions) ]
```sh
cd darknet
make GPU=1
```

## 3. Configure for training
The steps are copied and adjusted from [steps described in darknet](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects).

### 3.1. Make a copy of tiny yolo config file 
Our config (cfg) file will be called `yolov4-tiny-obj.cfg`
```sh
cp darknet/cfg/yolov4-tiny-custom.cfg darknet/cfg/yolov4-tiny-obj.cfg
```

Do the following changes to our `darknet/cfg/yolov4-tiny-obj.cfg`

1. Set `subdivisions=16`
2. Set `max_batches=2000`
3. Set `steps=4800,5400`
4. Set `classes=1` in all 2 `[yolo]` layes
5. Change `[filters=255]` to filters=(classes + 5)x3 in the 2 `[convolutional]` before each [yolo] layer, keep in mind that it only has to be the last `[convolutional]` before each of the `[yolo]` layers. In our case `filters=18` since `(1+5)*3 = 18`

Or you can use the `yolov4-tiny-obj.cfg` file from this repository and place it under `darknet/cfg/`

### 3.2. Create `obj.names` file
Create `obj.names` in the directory `darknet/build/darknet/x64/data/` with objects names - each in new line. This is the title tht will be displayed over the bounding boxes when this object is detected during inference. In our case, the only object type is NumberPlate. 
```
echo "NumberPlate" > darknet/build/darknet/x64/data/obj.names
```

(Try to avoid adding any newline to this file. Can cause some bugs down the line)

### 3.3. Create `obj.data` file
Create `obj.data` in the directory `darknet/build/darknet/x64/data/`. (Here classes = number of objects in our dataset).
```sh
touch darknet/build/darknet/x64/data/obj.data
```

Add the following content in this file using vim or some text editor

```txt
classes = 1
train  = /content/darknet/build/darknet/x64/data/train.txt
valid  = /content/darknet/build/darknet/x64/data/test.txt
names = /content/darknet/build/darknet/x64/data/obj.names
backup = /content/darknet/backup/
```

To save yourself from headache, specifiy absolute paths to files and folders even though the darknet project doesn't instruct to do so.

### 3.4. Put image files in `darknet/build/darknet/x64/data/obj/`
Our images & labels are currenly in `/content/NumberPlates`. Copy or move them to `/content/darknet/build/darknet/x64/data/obj/`.

```sh
mkdir -p /content/darknet/build/darknet/x64/data/obj/
cp /content/NumberPlates/* /content/darknet/build/darknet/x64/data/obj/
```

### 3.5. Create `train.txt` file in `darknet/build/darknet/x64/data/`
And add (absolute) filenames of your images to it. (Even though we copied our dataset to `darknet/build/darknet/x64/data/obj/`, providing absolute paths will save you from debugging path issues).

```txt
/content/darknet/build/darknet/x64/data/obj/image_1.jpg
/content/darknet/build/darknet/x64/data/obj/image_2.jpg
/content/darknet/build/darknet/x64/data/obj/image_3.jpg
.
.
.
```

[You can use a little python script to read all `.jpg` files from dataset dir and put their absolute paths in `train.txt`]

### 3.6. Download pre-trained weights
Download pre-trained weights for the convolutional layers of the Yolo model you want to use and put to the directory `darknet/build/darknet/x64/`

```sh
cd /content/darknet/build/darknet/x64/
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
```

## 4. Train on your data
Start the training process. Specifiy correct (relative/absolute) path to `obj.data`, our customized configuration file: `yolov4-tiny-obj.cfg`, and weights of yolo tiny pre-trained model `yolov4-tiny.conv.29`.

```sh
cd /content/darknet
./darknet detector train build/darknet/x64/data/obj.data cfg/yolov4-tiny-obj.cfg build/darknet/x64/yolov4-tiny.conv.29
```

### (Optional) If using Google Colab for training
Before starting the training process with above command, mount Google Drive to your Colab so that you could save your model to Google Drive after training. Because your progress (trained model) will be lost if there is no activity after training completes (which can take more than an our on a GPU)

1. Mount Google drive with:
```sh
from google.colab import drive
drive.mount('/content/gdrive')
```
(Provide auth token after following described steps).

2. Now train your mode with command described in above section
```sh
!./darknet detector train ...
```

3. Save all weights in `/content/darknet/backup/` to your specified directory
```sh
!cp /content/darknet/backup/* /content/gdrive/My\ Drive/Colab\ Notebooks/
```

IMPORTANT: After you run the train command (step 2), also click to run the cell that copies your model to Google Drive (step 3). This way, the copy command will run right after your training command completes. Make sure your Google Drive folder path is correct.


## 5. Convert model to work with TensorFlow.js
To run the inferences on the web, we need to convert our custom yolo model's weights to a format that TensorFlow.js recognizes.
