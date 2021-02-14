import flask
from werkzeug.utils import secure_filename
from Face_dectection import faceDetection
from CNN import CNNModel
from ImageProcess import ImageProcess
from SVM import SVM
from flask_ngrok import run_with_ngrok
import cv2


cnn = CNNModel()
svm = SVM()

app = flask.Flask(__name__)
app.secret_key="key"
run_with_ngrok(app)
@app.route('/', methods = ['GET', 'POST'])
def requestCheck():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            return "No file part"
        file = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return 'No selected file'
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(filename)

            fd= faceDetection()
            rot_img = fd.face_detection(filename)
            if(str(rot_img)!="NO"):
                denomination = cnn.Denomation_Detector(filename)
                if(denomination=="1000"):
                    ip= ImageProcess(denomination)
                    features = ip.features_extraction(filename)
                   
                    prediction = svm.predict(features)
                    return "The Currency Note Denomination is "+denomination+" and it is "+prediction
                else:
                    return "The Denomination is "+denomination    
            else:
                return "No Face is Detected"


if __name__ == '__main__':
    app.run()