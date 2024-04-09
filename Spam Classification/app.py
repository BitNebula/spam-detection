import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load your trained models
tfidf = pickle.load(open('C:\\Users\\aniru\Downloads\\Spam Classification\\tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('C:\\Users\\aniru\Downloads\\Spam Classification\\spam_classifier_model.pkl', 'rb'))

# Initialize the NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

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

    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

def main():
    st.title("SPAM EMAIL CLASSIFIER")

    input_text = st.text_area("Enter the email")

    if st.button('Detect'):
        transformed_input = transform_text(input_text)
        vectorized_input = tfidf.transform([transformed_input])
        result = model.predict(vectorized_input)[0]

        if result == 1:
            st.header("Spam")
        else:
            st.header("Genuine")

if __name__ == '__main__':
    main()




