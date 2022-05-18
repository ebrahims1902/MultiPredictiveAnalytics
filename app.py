from flask import Flask,render_template,request
import pickle
import numpy as np
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer


# Load Pickle
tfidf = pickle.load(open('./models/vectorizer.pkl', 'rb'))
sms_model = pickle.load(open('./models/sms_model.pkl', 'rb'))
cct_model = pickle.load(open('./models/credit_card_model.pkl', 'rb'))
gld_model = pickle.load(open('./models/gold_model.pkl', 'rb'))
cal_model = pickle.load(open('./models/calories_model.pkl', 'rb'))

# Mail Prediction
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

app=Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/about', methods=['GET','POST'])
def about():
    return render_template('about.html')

@app.route('/projects', methods=['GET','POST'])
def projects():
    return render_template('index.html')

@app.route('/skills', methods=['GET','POST'])
def skills():
    return render_template('index.html')

@app.route('/teams', methods=['GET','POST'])
def teams():
    return render_template('index.html')

# ------------------------ SMS Prediction API -------------------------
@app.route('/projectsms', methods=['GET','POST'])
def projectsmsprediction():
    return render_template('smsprediction.html')

# @app.route('/projectsmscode', methods=['GET','POST'])
# def projectsmspredictioncode():
#     return render_template('smspredictioncode.html')

# projects routes

@app.route('/predictsms', methods=['GET','POST'])
def predictsms():
    # 1. preprocess
    message = request.form['message']
    # input_sms = np.array(string_features)
    transformed_sms = transform_text(message)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    prediction = sms_model.predict(vector_input)
    return render_template('smspredictionresult.html', pred=prediction)


# -------------- Credit Card Transaction Prediction API --------------------

@app.route('/projectcct', methods=['GET','POST'])
def projectcctprediction():
    return render_template('cctprediction.html')

# @app.route('/projectcctcode', methods=['GET','POST'])
# def projectcctpredictioncode():
#     return render_template('cctpredictioncode.html')

@app.route("/predictcct", methods = ["GET", "POST"])
def predictcct():
    float_features = [[np.float64(x) for x in request.form.values()]]
    final = np.array(float_features)
    prediction = cct_model.predict(final)
    return render_template('cctpredictionresult.html', pred=prediction)
      
# -------------- Gold Prediction API ------------------------------------

@app.route('/projectgld', methods=['GET','POST'])
def projectgldprediction():
    return render_template('gldprediction.html')

# @app.route('/projectgldcode', methods=['GET','POST'])
# def projectgldpredictioncode():
#     return render_template('gldpredictioncode.html')

@app.route("/predictgld", methods = ["GET", "POST"])
def predictgld():
    float_features = [[np.float64(x) for x in request.form.values()]]
    final = np.array(float_features)
    prediction = gld_model.predict(final)
    return render_template('gldpredictionresult.html', pred=prediction)

# --------------- Calories Prediction API -------------------------------

@app.route('/projectcal', methods=['GET','POST'])
def projectcalprediction():
    return render_template('calprediction.html')

# @app.route('/projectcalcode', methods=['GET','POST'])
# def projectcalpredictioncode():
#     return render_template('calpredictioncode.html')

@app.route("/predictcal", methods = ["GET", "POST"])
def predictcal():
    float_features = [[np.float64(x) for x in request.form.values()]]
    final = np.array(float_features)
    prediction = cal_model.predict(final)
    return render_template('calpredictionresult.html', pred=prediction)        

# ---------------------- Route's For Team --------------------------
@app.route('/about-ebrahim', methods=['GET','POST'])
def ebrahim():
    return render_template('ebrahim.html')

@app.route('/about-shahul', methods=['GET','POST'])
def shahul():
    return render_template('shahul.html')

@app.route('/about-nijam', methods=['GET','POST'])
def nijam():
    return render_template('nijam.html')

@app.route('/about-mukesh', methods=['GET','POST'])
def mukesh():
    return render_template('mukesh.html')

if __name__=='__main__':
    app.run()