import numpy as np
from tensorflow.keras import Sequential, Model, Input, models
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score

# =============================================================== #
#    Read training data and prepare input and output arrays
# =============================================================== #

dataset = pd.read_csv('data/dataset.csv')

#filter languages 
#languages from different alphabet groups (referred to as dif_group). Accuracy: 0.998% 
dataset_1 = dataset[dataset['language'] =='English']
dataset_2 = dataset[dataset['language'] =='Hindi']
dataset_3 = dataset[dataset['language'] =='Arabic']

#languages from the same alphabet group (referred to as same_group). Accuracy: 0.772% 
# dataset_1 = dataset[dataset['language'] =='Portuguese']
# dataset_2 = dataset[dataset['language'] =='Romanian']
# dataset_3 = dataset[dataset['language'] =='Spanish']

dataset = pd.concat([dataset_1, dataset_2, dataset_3])

languages = dataset['language'].unique() #get unique language lablels
#sort labels so they always get same indexes 
languages = sorted(list(languages))
# print('Unique languages:', languages)

X = dataset['Text']
y = dataset['language']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #specific random_state to always get same split results 


# print('length of training sentences: ', len(X_train))
# print('length of testing sentences: ', len(X_test))


#count most frequent words in each input sentence
vectorizer1 = CountVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1, 1), max_features=2000)
words_features_train = vectorizer1.fit_transform(X_train).toarray()
words_features_test = vectorizer1.fit_transform(X_test).toarray()

#count most frequent ngrams in each input sentence
vectorizer2 = CountVectorizer(strip_accents='unicode', analyzer='char', ngram_range=(1, 4), max_features=2000)
ngram_features_train = vectorizer2.fit_transform(X_train).toarray()
ngram_features_test = vectorizer2.fit_transform(X_test).toarray()

#Build a dictionary to map unique vocabulary chars and target labels to their indices
output_label_index = dict(zip(languages, range(len(languages))))


#output labels are converted to one-hot vectors
output_labels = [to_categorical(output_label_index[label], num_classes=len(languages)) for label in y_train]
test_output_labels = [to_categorical(output_label_index[label], num_classes=len(languages)) for label in y_test]

#normalize input vectors to be between 0 and 1
train_data_words_unigrams = words_features_train.astype('float32') / words_features_train.max()
train_data_char_unigrams = ngram_features_train.astype('float32') / ngram_features_train.max()
test_data_words_unigrams = words_features_test.astype('float32') / words_features_test.max()
test_data_char_unigrams = ngram_features_test.astype('float32') / ngram_features_test.max()

# print(train_data_words_unigrams[:1])
# print(test_data_char_unigrams[:1])



# =============================================================== #
#              Define The Main model
# =============================================================== #


#the main model layers
words_feature_inputs = Input(shape = (len(words_features_train[0]),), name='words_feature_inputs')
ngram_feature_inputs = Input(shape = (len(ngram_features_train[0]),), name='ngram_feature_inputs')
concatenated_input = concatenate([words_feature_inputs, ngram_feature_inputs])

dense_layer1 = Dense(512, activation='relu')(concatenated_input)
dense_layer2 = Dense(256, activation='relu')(dense_layer1)
dense_layer3 = Dense(128, activation='relu')(dense_layer2)
model_outputs = Dense(len(languages), activation='softmax')(dense_layer3)

#the complete model 
model = Model([words_feature_inputs, ngram_feature_inputs], model_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True)


# =============================================================== #
#                       Train The model
# =============================================================== #


model.fit(
    [train_data_words_unigrams, train_data_char_unigrams], np.array(output_labels),
    epochs=40,
    batch_size=256,
    verbose = 1,
    shuffle=True,
    validation_data=([test_data_words_unigrams, test_data_char_unigrams], np.array(test_output_labels))
    # validation_split=0.2
)
# Save all models
model.save("model/trained_model_dif_groups.h5")


# ==============load the trained autoencoder=============
# model = models.load_model("model/trained_model_dif_groups.h5")

reverse_output_label_index = dict((i, label) for label, i in output_label_index.items())

predictions = model.predict([test_data_words_unigrams, test_data_char_unigrams])
predictions = np.argmax(predictions, axis=-1)
predicted_labels = [reverse_output_label_index[pred] for pred in predictions]

index = 0
for sentence in X_test[:10]:
    print("-" * 50)
    print(sentence)
    print(predicted_labels[index])
    index+=1

accuracy = accuracy_score(y_test, predicted_labels)
print('accuracy:', accuracy)
