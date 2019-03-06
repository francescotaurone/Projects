import time
from keras.models import load_model
import tensorflow as tf
from sklearn import svm
import glob
import numpy as np
import os
import cv2
from sklearn.externals import joblib


BUFFERSIZE = 20  # The size of the Buffer for the history of recognitions results
MIN_FACE_BUFF = 3  # Min number of times to appear in the most recent piece of the buffer for being valid
CHECK_BUFF = 6  # Size of cut buffer for analysis of the most recent ones, multiplied by the number of faces
PROB_THRESHOLD = 0.3  # Threshold on the probability of the image to belong to where the descriptor is pointing
PADDING_MAX = 50  # Increases the size of the window detected as being face by the detector
PADDING = 0  # This variable is meant to increase the size of the window found by descriptor around each face, if needed
FLUSH_BUFFER = 20  # Number of frames without detection of faces after which the buffer is flushed

# The following commented functions were used for previous implementations, such as with the euclidean distance for
# comparing descriptors. Here are just reported for reference
"""
def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_to_encoding_face(file, FRmodel)

    for (name, db_enc) in database.items():
        text_file.write("%s \t" % name)
    text_file.write("\n")
    return database
    
def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_identity(img, database, FRmodel, x, y, w, h):

    min_dist_identity = 100
    person = None
    padding_used= PADDING_MAX
    #Here looks at the minimum distances from the database images
    for (name, db_enc) in database.items():
        padding = PADDING_MAX
        min_dist_padding = 100
        min_padding=PADDING_MAX
        while padding > 0:

            x1 = x - padding
            y1 = y - padding
            x2 = x + w + padding
            y2 = y + h + padding

            part_image = get_box(img, x1, y1, x2, y2)
            encoding = img_to_encoding(part_image, FRmodel)



        #   Compute L2 distance between the target "encoding" and the current "emb" from the database.
            dist = np.linalg.norm(db_enc - encoding)

        #   print('distance for %s is %s' % (name, dist))

        #   If this distance is less than the min_dist, then set min_dist to dist, and identity to name
            if dist < min_dist_padding:
                min_dist_padding = dist
                min_padding = padding

            padding -= 10

        text_file.write("%s \t" % min_dist_padding)
        if min_dist_padding < min_dist_identity:
                min_dist_identity = min_dist_padding
                person = name
                padding_used = min_padding

    #   This number is empirical

    if min_dist_identity > 2:
        person = 'Unknown'
    else:
        person = str(person)
    text_file.write("%s \n" % person)
    return person, padding_used

"""


# This part of code is meant to return the descriptor of an image
def img_to_encoding(image, model):
    image = cv2.resize(image, (96, 96))
    img = image[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


# get_box cuts the region of interest
def get_box(img, x1, y1, x2, y2):

    height, width, channels = img.shape
    part_image = img[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

    return part_image


# This function  was implemented by template, has to be kept because it is needed for loading the neural network model
def triplet_loss(y_true, y_pred, alpha=0.3):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


# Given the image path, this function returns the relative descriptor
def img_to_encoding_face(image_path, FRmodel):

    img1 = cv2.imread(image_path, 1)
    if img1 is None:
        return -1
    face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    faces = face_cascade1.detectMultiScale(gray, 1.3, 5)
    part_image = None
    for (x, y, w, h) in faces:

        x1 = x - PADDING
        y1 = y - PADDING
        x2 = x + w + PADDING
        y2 = y + h + PADDING
        part_image = get_box(img1, x1, y1, x2, y2)

    # Here a control on the number of faces found
    if len(faces) != 1:
        return None
    else:
        encoding = img_to_encoding(part_image, FRmodel)
        return encoding


# Here is for building the database of descriptors that will be used by prepare_svm() for making the svm model
def prepare_database_for_svm():
    encoding_array = []
    name_array = []
    # load all the images of individuals to recognize into the database
    for file in glob.glob("images_video/*"):
        img_name = os.path.splitext(os.path.basename(file))[0]
        # this is for getting rid of the index of the image, getting only ID
        (identity, image_number) = img_name.split('_')
        print('%s is being processed.' % img_name)
        # To be noted that also the images in the database are cropped before encoding
        encoding = img_to_encoding_face(file, FRmodel)
        if encoding is None:  # This is to ensure that one and only one face was detected in the database image
            continue
        # the encoding returned has an extra dimension with no information that in the following is eliminated
        encoding_array.append(encoding[0, :])
        name_array.append(identity)
    return encoding_array, name_array


# This is where the descriptors are associated with classes and the SVM model is trained
def prepare_svm(encoding_array, name_array):

    model = svm.SVC(probability=True)
    lin_enc = np.array(encoding_array, dtype=np.float)
    # This is because during the building process of the encodings an additional dimension is
    # added, that now needs to be removed
    # lin_enc = lin_enc[:, 0, :]
    model.fit(lin_enc, name_array)
    # now the model is saved in the file system
    filename = 'FaceDatabase.sav'
    joblib.dump(model, filename)
    return model


# Here the identity of the captured frame is predicted based on the previously built SVM model
def find_identity_svm(model, img, FRmodel, x, y, w, h):

    x1 = x - PADDING
    y1 = y - PADDING
    x2 = x + w + PADDING
    y2 = y + h + PADDING

    part_image = get_box(img, x1, y1, x2, y2)
    encoding = img_to_encoding(part_image, FRmodel)
    identity = model.predict(encoding)
    prob = model.predict_proba(encoding)[0, :]
    array = model.classes_

# The following is for finding the index in the array of classes, so to threashold the relative probability of the
# instance being in the class. A cycle is implemented for simplicity, but for more complex id a more sophisticated
# search method is suggested
    i = 0
    while True:
        n = array[i]

        if int(n) == int(identity):
            break
        i += 1

    if prob[i] < PROB_THRESHOLD:
        identity[0] = '0'  # This zero corresponds to an Unknown
    return identity


# In this function the original frame is drown with names and rectangles for the faces identified
def draw_image(img, x, y, w, h, identity, padding_draw, original_guess):
    # Here the Ditionary.txt is used for associating id numbers to names
    d = {}
    with open("./Dictionary.txt") as dictionary:
        for line in dictionary:
            (key, val) = line.split()
            d[int(key)] = val
    print(d.get(int(identity)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    x1 = x - padding_draw
    y1 = y - padding_draw
    x2 = x + w + padding_draw
    y2 = y + h + padding_draw

    # The unknown id case is differentiated so to draw the frame with a different color
    if identity[0] == '0':
        frame = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, 'Unknown', (x1, y1), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # This next line is used to evaluate grafically the effectiveness of the buffer for recognized id, drawing the
        # original guess rather then the final guess at the bottom of the rectangle
        cv2.putText(frame, d.get(int(original_guess)), (x1, y2), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        frame = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, d.get(int(identity)), (x1, y1), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, d.get(int(original_guess)), (x1, y2), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    return frame


# Here the main function called after initialization
def webcam_face_recognizer(svm_model):
    # Two windows might be opened for comparing the original stream and the drawn one
    cv2.namedWindow("WEBCAM")
    cv2.namedWindow("WEBCAM_original")
    vc = cv2.VideoCapture(0)
    # This is the buffer used for making the identity prediction more robust.
    # It is initialized with zeros that correspond to unknowns
    recent_id = ['-1'] * BUFFERSIZE
    # This is for counting the images with no faces detected, so to understand changes of scene, with different subjects
    empty_frame = 0
    while vc.isOpened():

        start = time.time()

        _, frame = vc.read()
        """
        # In case of rotation needed
        num_rows, num_cols = frame.shape[:2]

        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 90, 1)
        frame = cv2.warpAffine(frame, rotation_matrix, (num_cols, num_rows))
        """
        img = frame
        original_img = img.copy()  # This copy is used for comparying the original stream with the elaborated one
        img_to_draw = img.copy()  # This copy is used for drawing and
        waiting_coord = []
        waiting_name = []
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        """
        # Code for equalization, just commented cause worsening performances
        img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_y_cr_cb)

        # Applying equalize Hist operation on channels.
        y_eq = cv2.equalizeHist(y)
        cr_eq = cv2.equalizeHist(cr)
        cb_eq = cv2.equalizeHist(cb)

        img_y_cr_cb_eq = cv2.merge((y_eq, cr_eq, cb_eq))
        img = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
        """
        """DETECTION"""
        # A more complete detection algorithm is being developed by others at the moment
        face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # The grey image is used only for detection, and the grey version is of the original image
        faces = face_cascade1.detectMultiScale(gray, 1.3, 5)
        # it returns the positions of detected faces as Rect(x,y,w,h)
        faces_number = len(faces)
        """RECOGNITION"""
        # The most recent set of recognition is considered for each image
        top = recent_id[:faces_number * CHECK_BUFF]
        # print(top)
        # This eliminates the -1 to ensure that even if the top lost is full of those, given that they do not
        # represent a real identity, they are not seen as the most occurring entry
        while '-1' in top:
            top.remove('-1')
        # A copy "top_copy" is made since one is modified in the process, the other has to reflect
        #  the original portion of Recent_id
        top_copy = top
        for (x, y, w, h) in faces:

            identity = find_identity_svm(svm_model, img, FRmodel, x, y, w, h)
            # An empty dimension is eliminated from identity
            identity = identity[0]

            # The array is updated with the guesses identity, but this is not the final result of the
            # recognition process.

            recent_id.insert(0,  identity)
            recent_id.pop()

            # Spurious and erroneous results are filtered out by checking how many times the identity has been found
            # in the recent past. A threshold is set for this purpose
            times = top.count(identity)
            if times >= MIN_FACE_BUFF:
                img_to_draw = draw_image(img_to_draw, x, y, w, h, str(identity), PADDING_MAX, identity)
                text_file.write("%s \t" % str(identity[0]))  # For storing the results
                # Zeros are not removed since multiple unknown identities might be present in the image
                if identity != '0':
                    while identity in top_copy:
                        # the confimed matches are removed so that the discarted guesses cannot choose among those,
                        # since are already present in the image, and a person cannot appear in two different positions
                        # in the image
                        top_copy.remove(identity)
            else:
                # If it was an isolated result, then it's probable that the real identity of the face was detected in
                # the recent past. Therefore it has to wait for checking which are the recent identities that are not
                # at the momement present in the image
                waiting_coord.append((x, y, w, h))
                waiting_name.append(identity)
        # Here deletes all the Unknown elements in the array, since we don't want those to affect the assignment of the
        # most probable identity
        i = 0
        while i < len(top_copy):
            top_copy[i] = int(top_copy[i])
            if top_copy[i] == 0:
                del top_copy[i]
            i += 1
        # print(top_copy)
        # For each of the identities under the threshold, the most common identity left in the array of identities is
        # going to be picked and assigned. This solution is not robust when various faces are in the waiting_coord,
        # therefore here it's assumed that the SVM has a good reliability
        i = 0
        for (x, y, w, h) in waiting_coord:

            id_possible = max(set(top_copy), key=top_copy.count, default='0')  # default 0 corresponds to unknown
            img_to_draw = draw_image(img_to_draw, x, y, w, h, str(id_possible), PADDING_MAX, str(waiting_name[i]))

            text_file.write("%s \t" % str(id_possible))  # For storing the results
            while identity in top_copy:
                # Removing the already aassigned entries ensures that no identity is assigned twice in the same image
                top_copy.remove(identity)
            i += 1

        if len(faces) != 0:
            text_file.write("\n")
            empty_frame = 0
        else:
            empty_frame += 1
        #  This action is for making sure that if the sujects of the video change, they are not recognized as
        # the previous subjects just because the buffer is full of the previous results
        if empty_frame == FLUSH_BUFFER:
            recent_id = ['-1'] * BUFFERSIZE
            empty_frame = 0
        waiting_coord.clear()
        key = cv2.waitKey(100)
        done = time.time()
        elapsed = done - start

        # Here the frame rate is printed at the top left corner of the image
        cv2.putText(img_to_draw, str(round(1/elapsed, 1)), (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("WEBCAM", img_to_draw)
        cv2.imshow("WEBCAM_original", original_img)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("WEBCAM")
    cv2.destroyWindow("WEBCAM_original")


if __name__ == "__main__":
    """INITIALIZATION"""
    # Here the model for the neural network is loaded
    FRmodel = load_model('net_model.h5', custom_objects={'triplet_loss': triplet_loss})
    # This file has been used to analyze the results
    text_file = open("Results.txt", "w")
    # Load the SVM model if it exists, otherwise build it
    if os.path.isfile('./FaceDatabase.sav'):
        clf = joblib.load('./FaceDatabase.sav')
    else:
        (encoding_svm, name_svm) = prepare_database_for_svm()
        clf = prepare_svm(encoding_svm, name_svm)
    # After initialization, the main function is called
    webcam_face_recognizer(clf)
    text_file.close()
