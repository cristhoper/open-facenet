import threading
from queue import Queue
from time import time

import cv2
from skimage.metrics import structural_similarity as compare_ssim

from mtcnn.mtcnn import MTCNN

OFFSET = 100
HISTOGRAM_THRESHOLD = 0.9
MARGIN = 32


class FaceDetection(object):
    use_haar = False

    GROUP_FACES_SIZE = 5

    def __init__(self, camera_id=0, use_haar=False, use_histogram=True, camera_id2=None):
        self.stream_lock = threading.Lock()
        self.use_histogram = use_histogram
        self.use_haar = use_haar
        if self.use_haar:
            casc_path1_0 = "haarcascade_frontalface_default.xml"
            casc_path1_1 = "haarcascade_frontalface_alt.xml"
            casc_path1_2 = "haarcascade_frontalface_alt2.xml"
            casc_path1_3 = "haarcascade_frontalface_alt_tree.xml"
            casc_path2_0 = "haarcascade_eye.xml"
            casc_path2_1 = "haarcascade_eye_tree_eyeglasses.xml"

            cascade_face0 = cv2.CascadeClassifier()
            cascade_face1 = cv2.CascadeClassifier()
            cascade_face2 = cv2.CascadeClassifier()
            cascade_facetree = cv2.CascadeClassifier()
            cascade_eye = cv2.CascadeClassifier()
            cascade_eyetree = cv2.CascadeClassifier()

            cascade_face0.load(cv2.data.haarcascades + casc_path1_0)
            cascade_face1.load(cv2.data.haarcascades + casc_path1_1)
            cascade_face2.load(cv2.data.haarcascades + casc_path1_2)
            cascade_facetree.load(cv2.data.haarcascades + casc_path1_3)
            cascade_eye.load(cv2.data.haarcascades + casc_path2_0)
            cascade_eyetree.load(cv2.data.haarcascades + casc_path2_1)

            self.cascades_fullfaces = [cascade_face0, cascade_face1, cascade_face2, cascade_facetree]
            self.cascades_fastfaces = [cascade_face0, cascade_face1]
            self.cascades_eyes = [cascade_eye, cascade_eyetree]

        self.detector = MTCNN()

        self.start_flag = True
        self.video_capture = cv2.VideoCapture(camera_id)
        self.camera_id2 = camera_id2
        if camera_id2 is not None:
            self.video_capture2 = cv2.VideoCapture(camera_id2)
        self.internal_lock = threading.RLock()
        self.__current_frame = None

    def prepare(self, size=(640, 480), fps=8):
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # self.video_capture.set(cv2.CAP_PROP_FOCUS, 3)
        self.video_capture.set(cv2.CAP_PROP_FPS, fps)

        if self.camera_id2 is not None:
            self.video_capture2.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.video_capture2.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self.video_capture2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            # self.video_capture2.set(cv2.CAP_PROP_FOCUS, 3)
            self.video_capture2.set(cv2.CAP_PROP_FPS, fps)

    def show_video(self):
        while True:
            ret, frame = self.video_capture.read()
            ret2, frame2 = self.video_capture2.read()
            cv2.imshow("video", frame)
            k = cv2.waitKey(60) & 0xff
            if k == 27 or key == ord("q"):
                break
        cv2.destroyAllWindows()

    def start(self, callback, use_thread=True):
        """
        Start face detection. Trigger callback at any detection
        :param callback: function that will receive image frame with face detected, array with coordinates
        from original frame and elapsed time since frame captured.
        :param use_thread: True if you want to return callback in a running thread. Use only if you are aware of it.
        :return: None
        """

        print('Start face detection')
        if self.use_haar:
            self.start_with_haar(callback, use_thread)
        else:
            self.start_with_mtcnn(callback, use_thread)

    @property
    def jpg_frame(self):
        with self.stream_lock:
            (s_flag, encoded_image) = cv2.imencode(".jpg", self.__current_frame)
            if s_flag:
                return encoded_image

    @property
    def png_frame(self):
        with self.stream_lock:
            (s_flag, encoded_image) = cv2.imencode(".png", self.__current_frame)
            if s_flag:
                return encoded_image

    @property
    def current_frame(self):
        with self.stream_lock:
            return self.__current_frame

    def preview_mtcnn(self):
        while True:
            ret, frame = self.video_capture.read()
            ret2, frame2 = self.video_capture2.read()

            try:
                faces1 = self.detector.detect_faces(frame2)
                faces2 = self.detector.detect_faces(frame)
            except Exception as err:
                faces = []
                print(err)

            if len(faces1 + faces2) > 0:
                for face in faces1:
                    (x, y, w, h) = face['box']
                    if h < OFFSET:
                        continue
                    text = "{}x{}".format(w,h)
                    keypoints = face['keypoints']
                    for e in keypoints:
                        cv2.circle(frame, keypoints[e], 3, (0, 0, 255))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    yt = y - 10 if y - 10 > 10 else y + 10
                    cv2.putText(frame, text, (x, yt),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                for face in faces2:
                    (x, y, w, h) = face['box']
                    if h < OFFSET:
                        continue
                    text = "{}x{}".format(w, h)
                    keypoints = face['keypoints']
                    for e in keypoints:
                        cv2.circle(frame2, keypoints[e], 3, (0, 255, 0))
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    yt = y - 10 if y - 10 > 10 else y + 10
                    cv2.putText(frame2, text, (x, yt), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            mixed_frame = cv2.addWeighted(frame, 0.5, frame2, 0.5, 0)
            cv2.imshow("Frame", mixed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
        self.video_capture.release()
        self.video_capture2.release()

    def start_with_mtcnn(self, callback, use_thread=False):
        """
        Start face detection with MTCNN. Trigger callback at any detection
        :param callback: function that will receive image frame with face detected, array with coordinates
        from original frame and elapsed time since frame captured.
        :param use_thread: True if you want to return callback in a running thread. Use only if you are aware of it.
        :return: None
        """
        __running = True
        last_gray = None

        while __running:
            self.internal_lock.acquire()
            __running = self.start_flag
            self.internal_lock.release()

            # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            with self.stream_lock:
                self.__current_frame = frame.copy()
            elapsed_time = time()
            try:
                faces = self.detector.detect_faces(frame)
            except Exception as err:
                faces = []
                print(err)

            if len(faces) > 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.use_histogram:
                    similarity = self.compare_frames(gray, last_gray)
                    last_gray = gray

                    if similarity > HISTOGRAM_THRESHOLD:
                        continue

                for face in faces:
                    (x, y, w, h) = face['box']
                    if h < OFFSET:
                        continue

                    inner_frame_orig = frame[y - MARGIN:y + h + MARGIN, x - MARGIN:x + w + MARGIN]
                    elapsed_time = time() - elapsed_time
                    callback_args = inner_frame_orig, face['box'], elapsed_time
                    self.__execute_callback(callback, *callback_args, threaded=use_thread)

    def start_with_haar(self, callback, use_thread=True):
        """
        Start face detection with HAAR cascades. Trigger callback at any detection
        :param callback: function that will receive image frame with face detected, array with coordinates
        from original frame and elapsed time since frame captured.
        :param use_thread: True if you want to return callback in a running thread. Use only if you are aware of it.
        :return: None
        """
        __running = True
        last_grayhist = None
        while __running:
            self.internal_lock.acquire()
            __running = self.start_flag
            self.internal_lock.release()
            elapsed_time = time()
            # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            with self.stream_lock:
                self.__current_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayhist = gray.copy()

            cv2.equalizeHist(gray, grayhist)
            faces = self.haar_parallel_detection(
                cascades=self.cascades_fastfaces,
                inner_frame=grayhist)

            for face in faces:
                (x, y, w, h) = face
                if h < OFFSET:
                    break

                eyes = self.haar_detection(self.cascades_eyes[0], grayhist[y:y + h, x:x + w])
                if len(eyes) > 0:

                    similarity = self.compare_frames(grayhist, last_grayhist)
                    last_grayhist = grayhist

                    if similarity > HISTOGRAM_THRESHOLD:
                        break

                    inner_frame_orig = frame[y:y + h, x:x + w]
                    elapsed_time = time() - elapsed_time
                    callback_args = inner_frame_orig, face, elapsed_time
                    self.__execute_callback(callback, *callback_args, threaded=use_thread)

    @staticmethod
    def __execute_callback(callback, *args, threaded=False):
        if callback.__code__.co_argcount == len(args):
            if threaded:
                try:
                    th = threading.Thread(target=callback, args=args)
                    th.daemon = True
                    th.start()
                except Exception as err:
                    print(err)
            else:
                callback(*args)
        else:
            raise AttributeError("Missing parameters on callback function")

    def stop(self):
        """
        Clean stop and release resources.
        :return: Start flag set to False.
        """
        self.internal_lock.acquire()
        self.start_flag = False
        self.internal_lock.release()
        self.video_capture.release()
        cv2.destroyAllWindows()
        return self.start_flag

    def haar_parallel_detection(self, cascades, inner_frame):
        """
        Find face using multiple HAAR cascades, running in parallel. Stops when one thread found a face.
        :param cascades: array with loaded cascades.
        :param inner_frame: frame with face to find
        :return: frame array with faces detected.
        """
        haar_threads = []
        q = Queue(maxsize=len(cascades))
        i = 0
        for cascade in cascades:
            th_id = i
            i += 1
            _frame = inner_frame.copy()
            _th = threading.Thread(target=self.haar_detection, args=(cascade, _frame, q, th_id))
            _th.setName("haar_parallel_detection-{}".format(i))
            haar_threads.append((th_id, _th))
            del _th
        th_ids = []
        for t_id, ht in haar_threads:
            ht.daemon = True
            th_ids.append(t_id)
            ht.start()

        __faces = []
        done = False
        count = 0
        while not done:
            response = q.get()
            count = count + 1 if response.get('th_id') in th_ids else count
            if response.get('detected'):
                q.task_done()
                __faces = response.get('objects')
                break
            done = count == len(th_ids)
        return __faces

    def haar_serial_detection(self, cascades, inner_frame):
        """
        Find face using nested cascades.
        :param cascades: array of cascades
        :param inner_frame: frame with face to find
        :return: frame face.
        """
        _frame = inner_frame.copy()
        for index in range(len(cascades)):
            detection = self.haar_detection(cascades[index], _frame)
            if index < len(detection):
                for (_x, _y, _w, _h) in detection:
                    self.haar_detection(cascades[index + 1, _frame[_y: _y + _h, _x: _x + _w]])
            if len(detection) > 0:
                return detection
        return None

    @staticmethod
    def haar_detection(classifier, inner_frame, queue=None, thread_id=None):
        """
        Detects objects based on classifier from the input image.
        .   of rectangles.

        :param classifier: Set the specific classifier
        :param inner_frame: frame to analyze.
        :param queue: Used for multithreading process.
        :param thread_id: Used for multithreading process.
        :return: array of frames with found elements.
        """
        objects_detected = classifier.detectMultiScale(inner_frame,
                                                       scaleFactor=1.1,
                                                       minNeighbors=2,
                                                       minSize=(40, 40),
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        if queue:
            try:
                data = {
                    'th_id': thread_id,
                    'objects': objects_detected,
                    'detected': len(objects_detected) > 0
                }
                queue.put(data)
            except Exception as err:
                print(err)
            return None
        return objects_detected

    @staticmethod
    def compare_frames(current_frame, earlier_frame):
        """

        :param current_frame:
        :param earlier_frame:
        :return:
        """
        score = 0.0
        compare_frame = current_frame.copy()
        if earlier_frame is not None and compare_frame is not None:
            (score, diff) = compare_ssim(compare_frame, earlier_frame, full=True)
        return score


class FaceTools(object):

    def __init__(self):
        self.detector = MTCNN()

    def face_on_image(self, filename):
        frame = cv2.imread(filename)
        faces = self.detector.detect_faces(frame)
        for face in faces:
            (x, y, w, h) = face['box']
            x, y = abs(x), abs(y)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('test', frame)
        cv2.imshow('test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    @staticmethod
    def save_face(frame, filepath):
        cv2.imwrite(filepath, frame)

    def show_frame(self, frame, tag='test'):
        cv2.imshow(tag, frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
