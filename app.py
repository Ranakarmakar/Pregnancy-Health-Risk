from flask import Flask, render_template, request
import os
import numpy as np
from keras.models import load_model

app = Flask(__name__)
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
port = int(os.getenv('PORT'))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class_names = ["High Risk", "Low Risk", "Mid Risk"]

model = load_model('model_.h5')


@app.route('/', methods=['GET'])
def hello_world():
    return render_template("home.html")


@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        N = request.form["N"]
        P = request.form["P"]
        K = request.form["K"]
        Temperature = request.form["Temperature"]
        humidity = request.form["humidity"]
        ph = request.form["ph"]
        rainfall = request.form["rainfall"]
        lst = [N, P, K, Temperature, humidity, ph, rainfall]
        for i in range(0, len(lst)):
            lst[i] = int(lst[i])
        prediction = model.predict([lst])
        result = "Recommended Crop is {} with {:.2f}% Confidence. ".format(class_names[np.argmax(prediction)],
                                                                           100 * np.max(prediction))
    else:
        result = " "

    return render_template("mai.html", prediction=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
    # app.run(debug=True)
