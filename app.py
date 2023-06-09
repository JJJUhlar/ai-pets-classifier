from fastai.vision.all import load_learner, Learner
from flask import Flask, jsonify, Response, request, render_template
from werkzeug.utils import secure_filename
import os

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
def index():
    return render_template('index.html')

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

            for pet in predictions.keys():
                predictions[pet] = format(float(predictions[pet]), 'f')
            
            sorted_predictions = sorted(predictions.items(), key=lambda x: float(x[1]), reverse=True)
            sorted_predictions = {pet: '{:.2%}'.format(float(prob)) for pet, prob in sorted_predictions}

            return render_template('result.html', predicted_class=sorted_predictions)

        except:
            return jsonify({'msg': "no image detected"})
        
if __name__ == '__main__':
    app.run()
