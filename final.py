import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# reading the stop words list with pickle
with open ('stop_words.ob', 'rb') as fp:
    domain_stop_word = pickle.load(fp)

# read data file
file_path = 'diseases_with_description.csv'
df = pd.read_csv(file_path)
#print(df.head())

def clean_text_func(text):
    
    """ this function clean & pre-process the data  """

    text = str(text)
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    final_text = ""
    for x in text.split():
        if x not in domain_stop_word:
            final_text = final_text + x  +" "
    return final_text

df['Description'] = df['Description'].apply(lambda x: clean_text_func(x))


# WORDS EMBEDDING
cv = CountVectorizer(stop_words="english")
cv_tfidf = TfidfVectorizer(stop_words="english")

X = cv.fit_transform(list(df.loc[:, 'Description' ]))
X_tfidf = cv_tfidf.fit_transform(list(df.loc[:, 'Description' ]))


df_cv = pd.DataFrame(X.toarray() , columns=cv.get_feature_names_out())
df_tfidf = pd.DataFrame(X_tfidf.toarray() , columns=cv_tfidf.get_feature_names_out())

#print(df_cv.shape)
cosine = lambda v1 , v2 : dot(v1 , v2) / (norm(v1) * norm(v2))
# Implementing Logistic Regression
X_train = df.Description
y_train = df.D_Name

cv1 = CountVectorizer()
X_train_cv1 = cv1.fit_transform(X_train)
pd_cv1 = pd.DataFrame(X_train_cv1.toarray(), columns=cv1.get_feature_names_out())

lr = LogisticRegression()
lr.fit(X_train_cv1, y_train)
import matplotlib.pyplot as plt
import numpy as np

# Calculate cosine similarity scores for each chapter
cosine_scores_cv = []
cosine_scores_tfidf = []
user_input = input("Detail your symptoms:\n")
new_text = [user_input]
new_text_cv = cv.transform(new_text).toarray()[0]
new_text_tfidf = cv_tfidf.transform(new_text).toarray()[0]

for chapter_number in range(int(df.shape[0])):
    cosine_scores_cv.append(cosine(df_cv.iloc[chapter_number], new_text_cv))
    cosine_scores_tfidf.append(cosine(df_tfidf.iloc[chapter_number], new_text_tfidf))

# Plot the cosine similarity scores
plt.figure(figsize=(10, 6))
plt.plot(range(int(df.shape[0])), cosine_scores_cv, label='Cosine CV')
plt.plot(range(int(df.shape[0])), cosine_scores_tfidf, label='Cosine TFIDF')
plt.xlabel('Chapter Number')
plt.ylabel('Cosine Similarity Score')
plt.title('Cosine Similarity Scores for Each Chapter')
plt.legend()
plt.show()

# Initialize
# Determine class counts
class_counts = Counter(y_train)

# Determine the maximum number of splits based on the filtered data
#max_splits = min(5, len(valid_classes))
max_splits = min(5, len(set(y_train)))
# ...
# ...

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import precision_score, recall_score, f1_score

# ...

if max_splits >= 2:
    # ...

    if all(count >= max_splits for count in class_counts.values()):
        # ...

        if len(X_train_filtered) > 0:  # Check if there are samples available
            for train_index, test_index in sss.split(X_train_filtered, y_train_filtered):
                # ...

            # Calculate average evaluation metrics across folds
             avg_precision = np.mean(precision_scores)
             avg_recall = np.mean(recall_scores)
             avg_f1 = np.mean(f1_scores)

            # Print the average evaluation metrics
             print("Average Precision:", avg_precision)
             print("Average Recall:", avg_recall)
             print("Average F1 Score:", avg_f1)
        else:
            print("No samples available in the filtered dataset.")
    else:
        print("Not enough samples available for each class for cross-validation.")
else:
    print("Not enough classes with samples for cross-validation.")






import gradio as gr

def predict_disease_description(symptoms):
    cleaned_text = clean_text_func(symptoms)
    X_test_cv3 = cv1.transform([cleaned_text])
    y_pred_cv3 = lr.predict(X_test_cv3)
    return y_pred_cv3[0]
# Create the Gradio interface
description_input = gr.inputs.Textbox(label='Detail your symptoms:')
#output_label = gr.outputs.Label()
output_label = gr.outputs.Label(num_top_classes=3)
gr.Interface(fn=predict_disease_description, inputs=description_input, outputs=output_label, title='Health Specialist Predictor', description='Enter the patient\'s symmptoms to get a suggested healtcare specialist.').launch()

