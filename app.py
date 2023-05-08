from fastai.vision.all import *
from flask import Flask, jsonify, Response, request

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './images/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

learn = load_learner('petClassifierModel.pkl')

categories = learn.dls.vocab

def classify_image(img):
    print(img)
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def hello():
    return 'Hello World!'

@app.route('/isalive')
def is_alive():
    print('/isalive request')
    status_code = Response(status=200)
    return status_code

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print('got request')
        try:
            file = request.files.getlist('image')[0]
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            predictions = classify_image(f'./images/{filename}')
            print(predictions)
            # format predictions / sort by accuracy
            return jsonify({
                "predictions": predictions
            })
        except:
            return jsonify({'msg': "no image detected"})