import numpy as np
import glob as gb
from matplotlib import pyplot
from PIL import Image
from numpy import asarray, expand_dims
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from numpy.linalg import norm
import math



###extract_face use for loading image, detect face, resize the face and extract face array
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # detect the face
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None
    
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


if __name__ == "__main__":

    # Example: You can adapt these paths to your data
    Dem_facePath = gb.glob(r"C:\path_to\dataset\Democrat\*.jpg")
    Rep_facePath = gb.glob(r"C:\path_to\dataset\Republican\*.jpg")

###create tran and test sataset
#Dem_face_train, Dem_face_test = train_test_split(Dem_facePath, test_size=0.2, random_state=100 )
#Rep_face_train, Rep_face_test = train_test_split(Rep_facePath, test_size=0.2, random_state=100 )

 # Use VGG model and get feature vectors
    def get_feature_vectors(image_paths):
    
        model = VGGFace(model='resnet50', include_top=False, pooling='avg')
        feature_list = []

        for fig in image_paths:
            pixels = extract_face(fig)
            if pixels is None:
                
                continue
            pixels = pixels.astype('float32')
            samples = expand_dims(pixels, axis=0)
            samples = preprocess_input(samples, version=2)
            yhat = model.predict(samples)
            feature_list.append(yhat[0])
        return feature_list

    # Generate vectors
    Dem_vec_man = get_feature_vectors(Dem_facePath)
    Rep_vec_man = get_feature_vectors(Rep_facePath)

    # Compute average vectors
    Dem_avg_vec = np.mean(Dem_vec_man, axis=0) if len(Dem_vec_man) > 0 else None
    Rep_avg_vec = np.mean(Rep_vec_man, axis=0) if len(Rep_vec_man) > 0 else None

    # Save average vectors to text files
    if Dem_avg_vec is not None:
        np.savetxt(r'C:\path_to\save\Dem_avg_man.txt', Dem_avg_man, delimiter=',')
    if Rep_avg_vec is not None:
        np.savetxt(r'C:\path_to\save\Rep_avg_man.txt', Rep_avg_man, delimiter=',')
   

    print("Average vectors have been generated and saved.")