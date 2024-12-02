from django.shortcuts import render
from django.http import HttpResponse
import joblib
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import sklearn
from django.views.decorators.cache import never_cache


nlp = spacy.load("en_core_web_sm")
# Load preprocessing pipeline
with open('./spammsgdetector/models/preprocessing_pipeline.pkl', 'rb') as pipeline_file:
    preprocessing_pipeline = joblib.load(pipeline_file)

# Load trained model
with open('./spammsgdetector/models/Logistic_Regression_model.pkl', 'rb') as model_file:
    trained_model = joblib.load(model_file)


def text_preprocessing(msg):
    m = msg.lower()
    m = re.sub(r"[^a-zA-Z\s]|'", '', m)
    m = re.sub(r'http\S+', '', m).strip()
    m = re.sub(r'https\S+', '', m).strip()
    m = re.sub(r'\s+', ' ', m).strip()
    stp_wrds = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    doc = nlp(m)
    lemmas = [token.lemma_ for token in doc if token.text not in stp_wrds and token.text not in string.punctuation]
    return ' '.join(lemmas)


def predict_message(message):
    # Ensure the input is in the expected format (e.g., a list or dataframe)
    if message=="":
        return None
    message = text_preprocessing(message)
    input_data = [message]  # Assuming the pipeline expects a list
    processed_data = preprocessing_pipeline.transform(input_data)  # Preprocess the message
    prediction = trained_model.predict(processed_data)  # Get the prediction
    # Optional: Return a user-friendly result (e.g., "Spam" or "Not Spam")
    
    result = {}
    if prediction[0] == 1:
        result['res'] = "Spam"
        result['proba'] = round(trained_model.predict_proba(processed_data)[0][1],2)
    else:
        result['res'] = "Not spam"
        result['proba'] = round(trained_model.predict_proba(processed_data)[0][0],2)
    return result



def welcome(Request):
    return HttpResponse("Hello world!")

# Create your views here.

@never_cache
def detect_spam(request):
    message = ""
    prediction = {'res': 'N/A', 'proba': 'N/A'}
    if request.method == "POST":
        message = request.POST.get('message', '')
        prediction = predict_message(message)  # Call your model's prediction function
    return render(request, 'index.html', {
        'message': message,
        'prediction': prediction['res'] if prediction else 'N/A',
        'proba': prediction['proba'] if prediction else 'N/A',
    })

