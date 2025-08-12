from of.faces import FaceDetection

video_detection = FaceDetection(camera_id=0)  # /dev/videoN
video_detection.prepare(size=(1280, 720))


def detected_face(frame, box, elapsed_time):
    print("doing: {}".format(elapsed_time))


video_detection.start(callback=detected_face, use_thread=False)
