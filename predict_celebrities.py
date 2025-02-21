import numpy as np
import glob as gb
import pandas as pd
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
    
    pixels = pyplot.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return asarray(image)


#  Compute the angle (in degrees) between two vectors based on cosine similarity.
 
def Cosine_cal(Vec, Vec_ref):
 
    cosine = np.dot(Vec, Vec_ref) / (norm(Vec) * norm(Vec_ref))
    phi_radian = math.acos(cosine)
    phi_degree = math.degrees(phi_radian)
    return phi_degree

if __name__ == "__main__":

    # 2.1 Load the average vectors previously generated
    Dem_avg_vec = np.loadtxt(r'C:\path_to\save\Dem_avg_vec.txt', delimiter=',')
    Rep_avg_vec = np.loadtxt(r'C:\path_to\save\Rep_avg_vec.txt', delimiter=',')

##list of celebrities
celebrities= gb.glob(r"C:\path_to\Celebrities\Celeb\*.jpeg")
# create a vggface model
model = VGGFace(model='resnet50', include_top=False, pooling='avg')

sample_vec = []
orientation_pred = []
pol_score = []

for fig_path in celebrities:
    pixels = extract_face(fig_path)
    if pixels is None:
        print(f"Warning: No face found for {fig_path}. Skipping.")
        continue

    pixels = pixels.astype('float32')
    samples = expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    yhat = model.predict(samples)
    vec = yhat[0]

    # Distances from Dem/Rep average (man)
    dist_dem = np.linalg.norm(vec - Dem_avg_vec)
    dist_rep = np.linalg.norm(vec - Rep_avg_vec)
    score = dist_dem - dist_rep

    # If score > 0 => "Republican" else => "Democrat"
    if score > 0:
        orientation_pred.append("Conservative")
    else:
        orientation_pred.append("Liberal")

    pol_score.append(score)
    sample_vec.append(fig_path)

    # Save results to CSV
    df_celeb = pd.DataFrame({
        'Filename': sample_vec,
        'Political ori': orientation_pred,
        'score': pol_score
    })
    df_celeb.to_csv('political_orientation_predictions_celebrities.csv', index=False)