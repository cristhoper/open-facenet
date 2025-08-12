import os
from sys import stderr

import magic
import numpy
import cv2
import pickle

from shutil import rmtree

from time import time

from keras.backend import clear_session
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer, LabelEncoder
from tensorflow.compat.v1 import get_default_graph

from .inception_resnet_v1 import *

FACENET_DS = 'DS_Model'  # https://github.com/davidsandberg/facenet
FACENET_HT = 'HT_Model'  # https://github.com/nyoki-mtl/keras-facenet


def find_euclidean_distance(sample_detected, sample):
    def l2_normalize(x):
        return x / numpy.sqrt(numpy.sum(numpy.multiply(x, x)))

    print("{}\n{}", sample_detected, sample)
    sample_detected = l2_normalize(sample_detected)
    sample_rep = l2_normalize(sample)
    euclidean_distance = sample_detected - sample_rep
    euclidean_distance = numpy.sum(numpy.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = numpy.sqrt(euclidean_distance)
    return euclidean_distance


class FaceRecognition(object):
    """
    Class to list images with faces, extract faces, load dataset from faces,
    """

    SUPPORTED_IMAGES = ['image/png', 'image/jpeg', 'image/x-portable-pixmap']

    MODELS_NAME = [FACENET_DS, FACENET_HT]
    RECOGNIZE_THRESHOLD = 99.999
    RECOGNIZE_THRESHOLD_LOW = 85.0

    def __init__(self, model_name=FACENET_DS, use_default_graph=False):
        self.use_default_graph = use_default_graph
        if model_name not in self.MODELS_NAME:
            raise AttributeError("Please select right model")
        self.model_name = model_name
        self.input_encoder = Normalizer(copy=False, norm='l2')  # quadratic
        self.output_encoder = LabelEncoder()
        self.emb_model = SVC(kernel='linear', probability=True)
        self.detector = MTCNN()

        self.graph = None
        if use_default_graph:
            clear_session()

        if model_name == FACENET_HT:
            self.facenet_model = load_model(os.path.dirname(os.path.realpath(__file__)) + '/model/facenet_keras.h5')
        elif model_name == FACENET_DS:
            self.facenet_model = InceptionResNetV1()
            self.facenet_model.load_weights(os.path.dirname(os.path.realpath(__file__)) + '/model/facenet_weights.h5')

        if use_default_graph:
            self.graph = get_default_graph()

        self.mime = magic.Magic(mime=True)
        self.data = None
        self.curr_dir_dataset = None

        self.current_dataset_faces, self.current_dataset_labels = None, None

    def extract_face(self, raw_image=None, filename=None, required_size=(160, 160), to_file=False, margin=0):
        if filename is not None:
            image = cv2.imread(filename)
        elif raw_image is not None:
            image = raw_image
        else:
            return None

        if self.use_default_graph:
            with self.graph.as_default():
                results = self.detector.detect_faces(image)
        else:
            results = self.detector.detect_faces(image)
        len_res = len(results)

        if len_res == 1:
            x1, y1, width, height = results[0]['box']
            x1, y1 = abs(x1), abs(y1)
            face = image[y1 - margin:y1 + height + margin, x1 - margin:x1 + width + margin]
            face_array = cv2.resize(face, required_size)
            if not to_file:
                return face_array
            else:
                cv2.imwrite(filename + ".2.png", face_array)
        elif len_res > 1:
            print("We only support one face per image. {} faces detected. Check {}".format(len_res, filename),
                  file=stderr)
        return None

    def get_embedding(self, face_pixels):
        def __predict(__samples):
            y_hat = self.facenet_model.predict(__samples)
            __embedded = y_hat[0]
            return __embedded

        embedded = 0.0
        try:
            face_pixels = face_pixels.astype('float32')
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean) / std
            samples = numpy.expand_dims(face_pixels, axis=0)
            if self.use_default_graph:
                with self.graph.as_default():
                    embedded = __predict(samples)
            else:
                embedded = __predict(samples)

        except Exception as err:
            print("get_embedding {}".format(err), file=stderr)
        return embedded

    def __load_faces__(self, directory, recursive=False, limit=None, keep_files=True, dir_len=0):
        def _inner_load(inner_path):
            file_type = self.mime.from_file(inner_path)
            if file_type not in self.SUPPORTED_IMAGES:
                return None
            face = self.extract_face(filename=inner_path)
            if face is not None:
                return face, filename, inner_path
            else:
                return None, filename, inner_path
        faces = list()
        labels = list()

        total_images = 0
        directory_listed = os.listdir(directory)
        for filename in directory_listed:
            _path = directory + '/' + filename

            if recursive:
                if not os.path.isdir(_path):
                    continue
                __path = os.listdir(_path)
                if len(__path) < dir_len:
                    continue
                datas = []
                for _f in __path:
                    _filename = _path + '/' + _f
                    _data = _inner_load(_filename)
                    if _data is not None:
                        datas.append(_data)
                if len(datas) >= dir_len:
                    total_images_nested = len(datas)
                    for data in datas:
                        if data[0] is None and not keep_files:
                            total_images_nested -= 1
                            os.remove(data[2])
                    if limit is None:
                        for data in datas:
                            faces.append(data[0])
                            labels.append(data[1])
                            total_images += 1
                    elif total_images_nested >= limit:
                        nested_images = 0
                        for data in datas:
                            if nested_images < limit and data[0] is not None:
                                faces.append(data[0])
                                labels.append(data[1])
                                total_images += 1
                                nested_images += 1
                            elif data[0] is not None:
                                dest_file = './testing-datasets/'+data[2]
                                dir_dest = os.path.dirname(dest_file)
                                file_dest = os.path.basename(dest_file)
                                os.makedirs(dir_dest, exist_ok=True)
                                os.rename(data[2], dir_dest + "/" + file_dest)
                    elif not keep_files:
                        rmtree(_path)
            else:
                data = _inner_load(_path)
                total_images += 1
                if data[0] is not None:
                    faces.append(data[0])
                    labels.append(data[1])

        print("registered {} of {} faces".format(len(faces), total_images))
        return faces, labels

    @staticmethod
    def mutate_dataset(directory):
        directory_listed = os.listdir(directory)
        for filename in directory_listed:
            _path = directory + '/' + filename

            if not os.path.isdir(_path):
                continue
            __path = os.listdir(_path)

            for _f in __path:
                _filename = _path + '/' + _f
                img_original = cv2.imread(_filename)
                img_flipped = cv2.flip(img_original.copy(), 1)
                h, w = img_original.shape[:2]
                center = (w / 2, h / 2)
                M_15 = cv2.getRotationMatrix2D(center, 15, 1.0)
                M_345 = cv2.getRotationMatrix2D(center, -15, 1.0)
                img_15deg = cv2.warpAffine(img_original, M_15, (h, w))
                img_345deg = cv2.warpAffine(img_original, M_345, (h, w))
                img_f_15deg = cv2.warpAffine(img_flipped, M_15, (h, w))
                img_f_345deg = cv2.warpAffine(img_flipped, M_345, (h, w))
                cv2.imwrite(_filename + '.flip_rot_15.png', img_f_15deg)
                cv2.imwrite(_filename + '.flip_rot_345.png', img_f_345deg)
                cv2.imwrite(_filename + '.rot_15.png', img_15deg)
                cv2.imwrite(_filename + '.rot_345.png', img_345deg)
                cv2.imwrite(_filename + '.flipped.png', img_flipped)

    def clear_dataset(self, dataset_name=None):
        if dataset_name is None:
            return
        inner_dataset_path = '/tmp/' + __name__ + 'collection-dataset-embedding-{}.npz'.format(dataset_name)
        if os.path.exists(inner_dataset_path):
            self.data = None
            self.current_dataset_faces = None
            self.current_dataset_labels = None
            os.remove(inner_dataset_path)

    def attach_embedding_to_dataset(self, dataset_name=None, embedding_data=None, embedding_label=None):
        inner_dataset_path = '/tmp/' + __name__ + 'collection-dataset-embedding-{}.npz'.format(dataset_name)
        if os.path.exists(inner_dataset_path):
            self.data = numpy.load(inner_dataset_path)
            self.current_dataset_faces = self.data['arr_0'].tolist()
            self.current_dataset_labels = self.data['arr_1'].tolist()
        else:
            self.current_dataset_faces = list()
            self.current_dataset_labels = list()

        if embedding_data is not None:
            self.current_dataset_faces.append(embedding_data)
            self.current_dataset_labels.append(embedding_label)
        else:
            print("attach_hash_to_dataset. hash_signature must not be None", file=stderr)
            return None

        dataset_name = '/tmp/' + __name__ + 'collection-dataset-embedding-{}.npz'.format(dataset_name)
        numpy.savez_compressed(dataset_name,
                               numpy.asarray(self.current_dataset_faces),
                               numpy.asarray(self.current_dataset_labels))

    def attach_face_to_dataset(self, dataset_name=None, image_buffer=None, image_label=None, data_augmentation=False):
        inner_dataset_path = '/tmp/' + __name__ + 'collection-dataset-embedding-{}.npz'.format(dataset_name)
        if os.path.exists(inner_dataset_path):
            self.data = numpy.load(inner_dataset_path)
            self.current_dataset_faces = self.data['arr_0'].tolist()
            self.current_dataset_labels = self.data['arr_1'].tolist()
        else:
            self.current_dataset_faces = list()
            self.current_dataset_labels = list()

        if image_buffer is not None:
            nparr = numpy.fromstring(image_buffer, numpy.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
            if data_augmentation:
                img_batch = []
                imf_flipped = cv2.flip(img_np.copy(), 1)
                h, w = img_np.shape[:2]
                center = (w / 2, h / 2)
                m_15 = cv2.getRotationMatrix2D(center, 15, 1.0)

                img_batch.append(img_np)
                img_batch.append(imf_flipped)
                img_batch.append(cv2.warpAffine(img_np, m_15, (h, w)))
                img_batch.append(cv2.warpAffine(imf_flipped, m_15, (h, w)))
                for img_i in img_batch:
                    face_pixels_i = self.extract_face(raw_image=img_i)
                    if face_pixels_i is None:
                        continue
                    face_embedded_i = self.get_embedding(face_pixels_i)
                    self.current_dataset_faces.append(face_embedded_i)
                    self.current_dataset_labels.append(image_label)
            else:
                face_pixels = self.extract_face(raw_image=img_np)
                face_embedded = self.get_embedding(face_pixels)
                self.current_dataset_faces.append(face_embedded)
                self.current_dataset_labels.append(image_label)

        dataset_name = '/tmp/' + __name__ + 'collection-dataset-embedding-{}.npz'.format(dataset_name)
        numpy.savez_compressed(dataset_name,
                               numpy.asarray(self.current_dataset_faces),
                               numpy.asarray(self.current_dataset_labels))

    def create_dataset(self, directory=None, revision='0', recursive=False, limit=None, keep_files=True, dir_len=0):
        if not directory:
            raise AttributeError("Directory must be defined")
        self.curr_dir_dataset = directory
        faces, labels = self.__load_faces__(directory, recursive, limit, keep_files, dir_len)
        faces_emb = list()
        total, i = len(faces), 0
        for face_pixels in faces:
            if face_pixels is None:
                continue
            i += 1
            print("\rEmbedding {}%".format(100*i//total), end="", flush=True)
            face_embedding = self.get_embedding(face_pixels)
            faces_emb.append(face_embedding)

        dataset_name = '/tmp/' + __name__ + 'collection-dataset-embedding-{}.npz'.format(revision)
        numpy.savez_compressed(dataset_name,
                               numpy.asarray(faces_emb),
                               numpy.asarray(labels))
        print("\nCollection zipped!")

    def save_trained_model(self, revision='0', output_filename_model=None):
        t = time()

        try:
            self.data = numpy.load('/tmp/' + __name__ + 'collection-dataset-embedding-{}.npz'.format(revision))
        except FileNotFoundError as err:
            print("save_trained_model {}".format(err), file=stderr)
            return

        to_match_faces, to_match_labels = self.data['arr_0'], self.data['arr_1']

        # normalize dataset faces
        to_match_faces = self.input_encoder.transform(to_match_faces)

        # label encoder
        self.output_encoder.fit(to_match_labels)
        to_match_labels = self.output_encoder.transform(to_match_labels)

        # fit model
        self.emb_model.fit(to_match_faces, to_match_labels)
        self.__save_fit_model(revision, len(to_match_labels), output_filename_model)
        print('saved in {}'.format(time()-t))

    def __save_fit_model(self, revision, collection_size, output_file_model=None):
        filename = output_file_model if output_file_model else 'compressed_datasets/emb-model-{}.pckl'.format(revision)
        with open(filename, 'wb') as f:
            pickle.dump((self.emb_model, self.output_encoder, collection_size), f)

    def load_trained_model(self, revision=None, input_filename_model=None):
        if input_filename_model is not None:
            filename = input_filename_model
        elif revision is not None:
            filename = 'compressed_datasets/emb-model-{}.pckl'
        else:
            raise AttributeError("revision, must be defined")
        with open(filename.format(revision), 'rb') as f:
            self.emb_model, self.output_encoder, collection_size = pickle.load(f)
            print("Collection elements: {}".format(collection_size))

    def get_model_labels(self):
        if self.output_encoder is not None:
            return self.output_encoder.classes_
        return None

    def model_scores(self, directory, confidence_threshold=RECOGNIZE_THRESHOLD,
                     file_directory_index_file=None, debug=False, no_samples=False):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        samples = 0
        not_in_samples = 0

        if file_directory_index_file is None:
            dirlist = os.listdir(directory)
        else:
            dirlist = []
            with open(file_directory_index_file, 'r') as label_file:
                for index, line in enumerate(label_file):
                    d = line.rstrip().split(' ')
                    dirlist.append({"label": d[0], "file": d[1]})

        for element in dirlist:

            if isinstance(element, str):
                inside_path = directory + '/' + element
                image_file_list = os.listdir(inside_path)
                label = element
            elif isinstance(element, dict):
                inside_path = directory
                image_file_list = [element.get('file', '')]
                label = element.get('label')
            else:
                continue

            if not os.path.isdir(inside_path):
                continue

            for filename in image_file_list:
                try:
                    image_filename = inside_path + '/' + filename
                    file_type = self.mime.from_file(image_filename)
                    if file_type not in self.SUPPORTED_IMAGES:
                        continue
                    frame = self.extract_face(filename=image_filename)
                    if frame is None:
                        continue

                    face_array, name, confidence = self.recognize(frame)
                    samples += 1

                    if debug:
                        print("Testing {}. Detected {} with {}".format(label, name, confidence), file=stderr)
                    if label in self.output_encoder.classes_:
                        if confidence > confidence_threshold:
                            if label == name:
                                true_positives += 1
                            else:
                                false_positives += 1
                    else:
                        not_in_samples += 1
                        if confidence > confidence_threshold:
                            false_negatives += 1
                except Exception as err:
                    print("model_scores {}".format(err), file=stderr)

        if no_samples and not_in_samples == samples:
            return None
        if true_positives + false_negatives == 0 or true_positives + false_positives == 0:
            return {
                "with_confidence": confidence_threshold,
                "accuracy": "bad score",
                "samples": samples,
                "samples_in_model": samples - not_in_samples,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }

        recall = true_positives / (true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)

        return {
            "with_confidence": confidence_threshold,
            "accuracy": true_positives / (true_positives + false_positives + false_negatives),
            "precision": precision,
            "recall":  recall,
            "f1_score":  2*precision*recall/(precision+recall),
            "samples": samples,
            "samples_in_model": samples - not_in_samples,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    def recognize(self, raw_image):
        # use just one face and check if prediction is valid or close to others.
        def __get_predict(__sample):
            __to_match = self.emb_model.predict(__sample)
            __prob = self.emb_model.predict_proba(__sample)
            return __to_match, __prob
        try:
            face_array = cv2.resize(raw_image, (160, 160)).copy()
        except Exception as err:
            print("recognize {}".format(err), file=stderr)
            face_array = None
        if face_array is None:
            return None, None, None

        sample_embedding = self.get_embedding(face_array)
        input_sample = numpy.expand_dims(sample_embedding, axis=0)

        if self.use_default_graph:
            with self.graph.as_default():
                y_hat_to_match, y_hat_prob = __get_predict(input_sample)
        else:
            y_hat_to_match, y_hat_prob = __get_predict(input_sample)

        max_len = len(y_hat_to_match)
        if max_len > 1:
            print(y_hat_to_match)
        class_index = y_hat_to_match[0]
        confidence = y_hat_prob[0, class_index] * 100
        predicted_name = self.output_encoder.inverse_transform(y_hat_to_match)
        name = predicted_name[0]
        return face_array, name, confidence
