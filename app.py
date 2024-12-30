import streamlit as st
import joblib
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np # linear algebra
from sklearn.decomposition import PCA
import nltk
nltk.download('punkt_tab')
word2vec_model = Word2Vec.load('word2vec_model.bin')  #Word2Vec vectorizer
# word2vec_model2 = Word2Vec.load('word2vec_model.bin.syn1neg.npy')  #Word2Vec vectorizer
# word2vec_model3 = Word2Vec.load('word2vec_model.bin.wv.vectors.npy')  #Word2Vec vectorizer

loaded_svm_model = joblib.load('svm_model.pkl')
loaded_logistic_model = joblib.load('logisticReg_model.pkl')

text=[ "الاقتصاد يتاثر بالاحداث العالمية"]
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
        # x=np.mean(vectors, axis=0)
        # x.reshape(1, -1)
        return np.mean(vectors, axis=0)  # Average the word vectors
    else:
        # x = np.zeros(vector_size)
        # x.reshape(1, -1)
        return np.zeros(vector_size)  # Return a zero vector if no words are in the model
# 3333333333333333333333333333333
def select_twenty_feature(vector):
    pca = PCA(n_components=20)
    x=vector.reshape(1,-1)
    print(x.shape)
    x2=pca.fit_transform(x)
    # x=pca.fit_transform(vector)
    return x2

# pca = PCA(n_components=20)
# reduced_embeddings = pca.fit_transform(doc_vector_matrix)
# preprocess = preprocessText(text)
# vector =  np.array([generate_average_word2vec(preprocess , word2vec_model)])
preprocess = [preprocessText(article) for article in text]
vector = np.array([generate_average_word2vec(article, word2vec_model) for article in preprocess])

# text_vector = select_twenty_feature(vector)
# print(text_vector.shape)
print(vector.shape)
svm_prediction = loaded_svm_model.predict(vector)[0]
logistic_prediction = loaded_logistic_model.predict(vector)[0]

print(svm_prediction )
print(logistic_prediction )

st.write('Hello world!')


# def main():
#     # App title
#     st.title("Text Processing App")

#     # Text input from the user
#     user_input = st.text_area("Enter your text below:")

#     # Process text when the button is clicked
#     if st.button("Process Text"):
#         if user_input.strip():
#             # Call the processing function
#             processed_text = process_text(user_input)
#             st.subheader("Processed Text:")
#             st.write(processed_text)
#         else:
#             st.warning("Please enter some text to process.")

# if __name__ == "__main__":
#     main()

# st.write('Hello world!')
