# Face Recognition application

This package helps to train, test and execute Facenet neuronal network for Face identification.
The implementation of this software is based on [D. Sandberg, Facenet](https://github.com/davidsandberg/facenet), [Facenet with keras](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) and this two papers [“FaceNet: A Unified Embedding for Face Recognition and Clustering.”](https://arxiv.org/abs/1503.03832) and  [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)

## Configuration
To install just run
```shell script
$ python3 -m venv .venv #For virtual environment
$ source .venv/bin/activate
(.venv) $ pip install -r requirements.txt
```

## Usage
To create multiple trained image datasets from `DATASET_N`, just run:
```python
from of.faces import FaceRecognition

server_faces = FaceRecognition()

DATASET_1 = '/image-dataset/100/'
DATASET_2 = '/image-dataset/200/'
DATASET_3 = '/image-dataset/300/'

server_faces.create_dataset(directory=DATASET_1, revision='testing0001', recursive=True)
server_faces.create_dataset(directory=DATASET_2, revision='testing0002', recursive=True)
server_faces.create_dataset(directory=DATASET_3, revision='testing0003', recursive=True)

server_faces.save_trained_model(revision='testing0001', output_filename_model='embedded_file_model_100.pckl')
server_faces.save_trained_model(revision='testing0002', output_filename_model='embedded_file_model_200.pckl')
server_faces.save_trained_model(revision='testing0003', output_filename_model='embedded_file_model_300.pckl')
```

This returns embedded files with labels and trained models for an easy distribution. And can be loaded just like:
```python
from of.faces import FaceRecognition
totem_faces = FaceRecognition()

totem_faces.load_trained_model(input_filename_model='embedded_file_model_200.pckl')
```
Next, start detecting faces
```python
from of.faces import FaceDetection
video_detection = FaceDetection(camera_id=N)  # /dev/videoN
video_detection.prepare(size=(1280, 720))

def detected_face(frame, box):
    """
    callback with cv2 frame and (x, y, width, height) tuple box
    """

    pass

video_detection.start(callback=detected_face, stream_callback=stream_frame, use_thread=False)
```
and on another thread, get the streamed video with:
```python
video_detection.get_stream()
```
This will return a `multipart/x-mixed-replace`
And then, just start identifying
```python
raw_image, predicted_name, confidence = totem_face.recognize(raw_image=frame)
```

That's all <3