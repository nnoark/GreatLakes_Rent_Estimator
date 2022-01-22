from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('./models/model.pkl','rb'))

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    result = model.predict(to_predict)
    return result[0]

@app.route('/')
def index():     
    return render_template('index.html')
 
@app.route('/result', methods = ['POST','GET'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        prediction = ValuePredictor(to_predict_list)                 
        return render_template("prediction.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)