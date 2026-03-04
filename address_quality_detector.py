import numpy as np
import pandas as pd
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accurcy_score, f1_score, confusiin_matrix, roc_curve, auc
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer
import fasttext
from huggingface_hub import hf_hub_download

import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(BASE_DIR, "model_address_quality_detector.bin")

# print("Script location:", BASE_DIR)
# print("Model path:", model_path)
# print("Exists?:", os.path.exists(model_path))

model_path = hf_hub_download(
    repo_id="swatigupta110/address-quality-detector-private",
    filename="model_address_quality_detector.bin",  # change if filename differs
    token=os.getenv("HF_TOKEN")
)

model = fasttext.load_model(model_path)

# Load model
# model = fasttext.load_model("model_address_quality_detector.bin")

def combine_features(row):
  combined=" "
  combined+=row['address']+" " if row['address']!="" else ""
  combined+=row['address_line1']+" " if row['address_line1']!="" else ""
  combined+=row['landmark']+" " if row['landmark']!="" else ""
  combined+=row['city']+" " if row['city']!="" else ""
  combined+=row['state']+" " if row['state']!="" else ""
  return combined

tokenizer = RegexpTokenizer(r'\w+')


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

def replace_numbers_with_token(tokens):
    return ['<number>' if re.fullmatch(r'\b\d+\b|\d+[a-zA-Z]*|[a-zA-Z]+\d+', token) else token for token in tokens]

lancaster_stemmer = LancasterStemmer()

def apply_lancaster_stemming(tokens):
       return [lancaster_stemmer.stem(token) for token in tokens]

def join_tokens(tokens):
    return ' '.join(tokens)


# 'model_address_quality_detector.bin'


def process_and_predict(new_data):
    # Fill missing values
    new_data = new_data.fillna("")

    # Apply the preprocessing steps
    new_data['combined_features'] = new_data.apply(combine_features, axis=1)                              # Combine the columns
    new_data['regexp_tokens'] = (new_data['combined_features'].str.lower()).apply(tokenizer.tokenize)     # Apply Regexp Tokenizer
    new_data['filtered_tokens'] = new_data['regexp_tokens'].apply(remove_stopwords)                       # Remove Stopwords
    new_data['filtered_tokens'] = new_data['filtered_tokens'].apply(replace_numbers_with_token)           # Replace numbers with <number>
    new_data['filtered_tokens'] = new_data['filtered_tokens'].apply(apply_lancaster_stemming)             # Apply Lancaster Stemming
    new_data['joined_tokens'] = new_data['filtered_tokens'].apply(join_tokens)                            # Join the filtered text
    # model = fasttext.load_model(model_path)                                                               # Load the trained FastText model
    predicted_label = []                                                                                  # Make predictions on the new data
    probability = []
    for text in new_data['joined_tokens']:
        label, prob = model.predict(text)
        predicted_label.append(label[0].replace('__label__', ''))
        probability.append(prob[0])
    new_data['predicted_quality'] = predicted_label                                                       # Add predicted_label and probability to the new data DataFrame
    new_data['probability'] = probability
    return new_data                                                                                       # Return new_data



# '''
# # Manually input data
# data = {
#     "address": ["#14/2, 2nd floor, 2nd main, 9th cross bmk layout, vittal nagar, chamrajpet",
#                 "(saraimeer road) , fariha chowk, fariha",
#                 ", nagbal district : srinagar,",
#                 ", nh-17b, zuarinagar,",
#                 "my house"
#                 ],
#     "address_line1": ["#14/2,",
#                       "azizia hijama & homeopathic clinic",
#                       "madina complex nagbal 90ft",
#                       "355 dh-5 bits goa campus",
#                       "near water tank"
#                       ],
#     "landmark": ["near azadnagar police outpost",
#                  "opposite nayra petrol pump",
#                  " near petrol pump",
#                  "",
#                  "near mandir"
#                  ],
#     "city": ["bangalore",
#              "azamgarh",
#              "srinagar",
#              "san",
#              "satna"
#              ],
#     "state": ["karnataka",
#               "uttar pradesh",
#               "jammu and kashmir",
#               "goa",
#               "madhya pradesh"
#               ]
# }


# new_data = pd.DataFrame(data)
# result = process_and_predict(new_data)
# print(result)

# '''