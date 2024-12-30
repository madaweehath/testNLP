import streamlit as st
import joblib
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np # linear algebra
import nltk
nltk.download('punkt_tab')
word2vec_model = Word2Vec.load('/word2vec_model.bin')  #Word2Vec vectorizer

loaded_svm_model = joblib.load('/svm_model.pkl')
loaded_logistic_model = joblib.load('/logisticReg_model.pkl')

# 111111111111111111111111111
arabic_stopwords = [
    "في", "من", "على", "إلى", "عن", "و", "لا", "ما", "لم", "لن", "إن", "أن", "لكن", "بل", "أو",
    "إما", "ف", "ثم", "حين", "عند", "حيث", "كيف", "كما", "أين", "متى", "هنا", "هناك", "مع", "كل",
    "هذا", "هذه", "ذلك", "تلك", "إذ", "إذا", "إلا", "أي", "أيضا", "أحد", "بعض", "كلما", "بين",
    "حتى", "أمام", "إليكم", "إليكما", "إليك", "أنت", "أنتما", "أنتم", "أنتن", "أنا", "نحن",
    "هو", "هي", "هما", "هم", "هن", "كان", "كانت", "يكون", "يكونون", "تكون", "تكونين", "نكون",
    "كانوا", "أكون", "يكن", "لم يكن", "ليس", "ليسوا", "ليست", "لن يكون", "قد", "لقد", "سوف",
    "وما", "وهذا", "وهذه", "وهل", "هناك", "به", "بهذا", "بهذه", "بهم", "بها", "بذلك", "بما",
    "لذلك", "لما", "معه", "معها", "معهم", "فيه", "فيها", "منه", "منها", "منهم", "عليه", "عليها",
    "عليهم", "إليه", "إليها", "إليهم", "إليك", "لك", "لكم", "له", "لها", "لهم", "لهذا", "لهذه",
    "إذن", "بينما", "أيها"
]
def preprocessText(text):
    # إزالة التشكيل
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

    # إزالة أي شيء ليس حرفًا عربيًا أو مسافةوالرموز التعبيرية
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)

    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()

    # إزالة كل ما هو غير الأحرف
    text = re.sub(r'[^\w\s]', '', text)
    # Tokeniz The Sentence into tokens
    Tokens = word_tokenize(text)

    Tokens = [word for word in Tokens if word not in arabic_stopwords and len(word) > 1]

    PreprocessedText = ' '.join(Tokens)

    return PreprocessedText
# 2222222222222222222222222222
def generate_average_word2vec(text, model, vector_size=100):
    tokens = text.split()  # Tokenize the preprocessed text
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)  # Average the word vectors
    else:
        return np.zeros(vector_size)  # Return a zero vector if no words are in the model






    # App title
st.title("Text Processing App")

    # Text input from the user
user_input = st.text_area("Enter your text below:")
    # Process text when the button is clicked
if st.button("Process Text"):
    if user_input.strip():
            # Call the processing function
        text=[user_input]
        preprocess = [preprocessText(text1) for text1 in text]
        vector = np.array([generate_average_word2vec(text, word2vec_model) for text in preprocess])
        svm_prediction = loaded_svm_model.predict(vector)[0]
        logistic_prediction = loaded_logistic_model.predict(vector)[0]
        st.write("prediction1"+svm_prediction )
        st.write("prediction2"+logistic_prediction )
    else:
        st.warning("Please enter some text to process.")
