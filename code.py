from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import csv, nltk, pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    cv = CountVectorizer()
    
    ans = [str(x) for x in request.form.values()]
    ans = ans[5:]
    
    sample = cv.transform(ans)

    output = model.predict(sample)
    print(output)
    return render_template('index.html', prediction_text = 'emotion: {}'.format(output))


if __name__ == '__main__':
    app.run()