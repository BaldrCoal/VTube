import cv2
import mediapipe
import numpy as np

cap = cv2.VideoCapture(1)
face_recognizer = mediapipe.solutions.face_mesh.FaceMesh()
render = mediapipe.solutions.drawing_utils
SCALE = 3
OFFSET_Y = -650
OFFSET_X = -550

while True:
    _, img = cap.read()
    avatar_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    avatar_img[:,:,1] = 255
    result = face_recognizer.process(img)
    if result.multi_face_landmarks:
        for landmark in result.multi_face_landmarks:
            # render.draw_landmarks(image=avatar_img, landmark_list=landmark,
            #                       connections=mediapipe.solutions.face_mesh.FACEMESH_TESSELATION,
            #                       landmark_drawing_spec=None,
            #                       connection_drawing_spec=mediapipe.solutions.drawing_styles.
            #                       get_default_face_mesh_tesselation_style())
            for i in range(len(landmark.landmark)):
                print(landmark.landmark[i])
                pixel = render._normalized_to_pixel_coordinates(landmark.landmark[i].x, landmark.landmark[i].y,
                                                                avatar_img.shape[1], avatar_img.shape[0])

                cv2.circle(avatar_img, (int(pixel[0]*SCALE+OFFSET_Y), int(pixel[1]*SCALE+OFFSET_X)), 2, (255, 255, 255))

    cv2.imshow("av_img", avatar_img)
    cv2.imshow("img", img)
    cv2.waitKey(1)
