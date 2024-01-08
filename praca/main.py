import os
import pickle
import pandas as pd
import stopwords
import tensorflow as tf
import numpy as np
import re
import unidecode
import morfeusz2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, \
    Dropout, MaxPooling1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from keras_tuner import Hyperband
from tensorflow.keras.callbacks import ReduceLROnPlateau
from itertools import product
import sys
import signal


class TextClassifier:

    def __init__(self, category_col, text_cols, min_category_count=20, max_epochs=50, batch_size=32,
                 save_folder='model_save',
                 save_name='classifier', retrain=True):
        self.category_col = category_col
        self.text_cols = text_cols
        self.min_category_count = min_category_count
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.save_folder = save_folder
        self.save_name = save_name
        self.retrain = retrain
        self.model_path = os.path.join(save_folder, save_name, 'model.h5')
        self.preprocessing_path = os.path.join(save_folder, save_name, 'preprocessing.pkl')
        self.encoder_path = os.path.join(save_folder, save_name, 'encoder.pkl')
        self.morf = morfeusz2.Morfeusz()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    @staticmethod
    def signal_handler(sig, frame):
        print("Przechwycenie sygnału: ", sig)
        tf.keras.backend.clear_session()
        print("Zasoby GPU zwolnione poprawnie.")
        sys.exit(0)

    def load_and_preprocess_data(self, filepath):
        data = pd.read_csv(filepath)
        category_counts = data[self.category_col].value_counts()
        categories_to_keep = category_counts[category_counts >= self.min_category_count].index
        filtered_data = data[data[self.category_col].isin(categories_to_keep)]
        filtered_data = filtered_data.dropna(subset=self.text_cols)
        combined_text = filtered_data[self.text_cols].agg(' '.join, axis=1)
        filtered_data['combined_text'] = combined_text.str.replace(r'[^\w\s]+', '', regex=True).str.lower()
        return filtered_data

    @staticmethod
    def preprocess_text(text, method):
        morf = morfeusz2.Morfeusz()
        if method['use_lemmatization']:
            analysis = morf.analyse(text)
            text = ' '.join([word[2][1].split(':')[0] for word in analysis])

        if method['remove_punctuation']:
            text = re.sub(r'[^\w\s]+', '', text)

        if method['remove_diacritics']:
            text = unidecode.unidecode(text)

        if method['lowercase']:
            text = text.lower()

        if method['remove_stopwords']:
            text = ' '.join([word for word in text.split() if word not in stopwords.words('polish')])

        if method['remove_single_characters']:
            text = ' '.join([word for word in text.split() if len(word) > 1])

        return text

    def preprocess_data(self, data, method):
        return np.array([self.preprocess_text(text, method) for text in data])

    def build_branch1(self, hp, input_text):
        max_features = hp.Int('max_features', min_value=20000, max_value=50000, step=1000)
        output_mode = 'tf-idf'

        vectorize_layer_tfidf = TextVectorization(max_tokens=max_features, output_mode=output_mode, standardize=None)
        vectorize_layer_tfidf.adapt(self.X_train.to_numpy())
        x = vectorize_layer_tfidf(input_text)

        for i in range(hp.Int('num_dense_layers_branch1', 1, 3)):
            x = Dense(
                units=hp.Int('dense_units_branch1_' + str(i), min_value=64, max_value=192, step=8),
                activation=hp.Choice('dense_activation_branch1_' + str(i), ['relu', 'tanh', 'sigmoid']),
                kernel_regularizer=l2(hp.Float('l2_dense_branch1_' + str(i), 1e-8, 1e-4, sampling='LOG'))
            )(x)
            x = Dropout(hp.Float('dropout_dense_branch1_' + str(i), 0.0, 0.7, step=0.1))(x)

        return x

    def build_branch2(self, hp, input_text):
        max_features_branch2 = hp.Int('max_features_branch2', min_value=20000, max_value=50000, step=1000)
        embedding_dim = hp.Int('embedding_dim', min_value=200, max_value=600, step=8)

        vectorize_layer_int = TextVectorization(max_tokens=max_features_branch2, output_mode='int', standardize=None)
        vectorize_layer_int.adapt(self.X_train.to_numpy())
        x = vectorize_layer_int(input_text)

        x = Embedding(max_features_branch2, embedding_dim)(x)
        dropout_rate = hp.Float('dropout_rate_embedding', 0.0, 0.7, step=0.1)
        x = Dropout(dropout_rate)(x)

        for j in range(hp.Int('num_conv1d_layers_branch2', 1, 2)):
            x = Conv1D(
                filters=hp.Int('filters_branch2_' + str(j), min_value=32, max_value=256, step=8),
                kernel_size=hp.Int('kernel_size_branch2_' + str(j), min_value=4, max_value=8, step=1),
                activation='relu',
                kernel_regularizer=l2(hp.Float('l2_conv1d_branch2_' + str(j), 1e-8, 1e-4, sampling='LOG'))
            )(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(hp.Float('dropout_conv1d_branch2_' + str(j), 0.0, 0.7, step=0.1))(x)

        x = GlobalMaxPooling1D()(x)

        return x

    def build_model(self, hp):
        input_text = Input(shape=(1,), dtype=tf.string)
        branch1 = self.build_branch1(hp, input_text)
        branch2 = self.build_branch2(hp, input_text)
        combined = Concatenate()([branch1, branch2])
        output = Dense(self.y_train_categorical.shape[1], activation='softmax')(combined)
        model = Model(inputs=input_text, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def perform_tuning(self, data, labels, directory, project_name):
        tuner = Hyperband(
            self.build_model,
            objective='val_accuracy',
            max_epochs=self.max_epochs,
            factor=3,
            seed=42,
            directory=directory,
            project_name=project_name
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5)
        tuner.search(data, labels, epochs=self.max_epochs, validation_split=0.2, callbacks=[reduce_lr, stop_early],
                     batch_size=self.batch_size)
        return tuner.get_best_hyperparameters(num_trials=1)[0]

    def perform_initial_tuning(self, data, labels):
        return self.perform_tuning(data, labels, 'keras_tuning', 'initial_tuning')

    def perform_final_tuning(self, data, labels):
        return self.perform_tuning(data, labels, 'keras_tuning', 'final_tuning')

    def fit(self, filepath):
        if os.path.exists(self.model_path) and os.path.exists(self.preprocessing_path) and os.path.exists(
                self.encoder_path) and not self.retrain:
            print("Wczytywanie zapisanych modelu, preprocessingu i enkodera...")
            self.best_model = tf.keras.models.load_model(self.model_path)
            with open(self.preprocessing_path, 'rb') as f:
                self.best_method = pickle.load(f)
            with open(self.encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)
        else:
            data = self.load_and_preprocess_data(filepath)
            X_train, X_test, y_train, y_test = train_test_split(data['combined_text'], data['category'], test_size=0.2,
                                                                random_state=42)
            self.X_train = X_train

            self.encoder = LabelEncoder()
            y_train_encoded = self.encoder.fit_transform(y_train)
            y_test_encoded = self.encoder.transform(y_test)
            y_train_categorical = np_utils.to_categorical(y_train_encoded)
            self.y_train_categorical = y_train_categorical
            y_test_categorical = np_utils.to_categorical(y_test_encoded)

            best_hps = self.perform_initial_tuning(X_train.to_numpy(), y_train_categorical)

            options = {
                "remove_punctuation": [True, False],
                "lowercase": [True, False],
                "use_lemmatization": [True, False],
                "remove_diacritics": [True, False],
                "remove_stopwords": [True, False],
                "remove_single_characters": [True, False]

            }
            preprocessing_methods = [dict(zip(options, combination)) for combination in product(*options.values())]
            best_method = None
            best_val_accuracy = 0

            for method in preprocessing_methods:
                print(f"Testowanie metody preprocessingu: {method}")
                X_train_processed = self.preprocess_data(X_train, method)
                model = self.build_model(best_hps)
                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5)
                history = model.fit(X_train_processed, self.y_train_categorical, epochs=self.max_epochs,
                                    validation_split=0.2,
                                    callbacks=[reduce_lr, stop_early], batch_size=self.batch_size)
                val_accuracy = max(history.history['val_accuracy'])
                print(f"Wynik dokładności dla metody preprocessingu {method}: {val_accuracy:.4f}")

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_method = method

            print(f"Najlepsza metoda preprocessingu: {best_method}")
            print(f"Najwyższa dokładność walidacyjna: {best_val_accuracy:.4f}")

            self.best_method = best_method
            X_train_processed_final = self.preprocess_data(data, best_method)
            best_hps_final = self.perform_final_tuning(X_train_processed_final, self.y_train_categorical)
            model_final = self.build_model(best_hps_final)
            X_train_processed_final = self.preprocess_data(X_train, best_method)
            X_test_processed_final = self.preprocess_data(X_test, best_method)

            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5)
            model_final.fit(X_train_processed_final, self.y_train_categorical, epochs=self.max_epochs,
                            validation_split=0.1,
                            callbacks=[reduce_lr, stop_early], batch_size=self.batch_size)
            self.best_model = model_final
            test_loss, test_accuracy = model_final.evaluate(X_test_processed_final, y_test_categorical)
            print(f"Dokładność modelu na zbiorze testowym: {test_accuracy:.4f}")
            if not os.path.exists(os.path.join(self.save_folder, self.save_name)):
                os.makedirs(os.path.join(self.save_folder, self.save_name))
            self.best_model.save(self.model_path)
            with open(self.preprocessing_path, 'wb') as f:
                pickle.dump(self.best_method, f)
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(self.encoder, f)
            print(f"Model, preprocessing i encoder zapisane w: {os.path.join(self.save_folder, self.save_name)}")

    def predict(self, examples):
        examples_processed = self.preprocess_data(examples, self.best_method)
        predictions = self.best_model.predict(examples_processed)
        return self.encoder.inverse_transform(np.argmax(predictions, axis=1))


text_cols = ["label_longdescription", "label_shortdescription"]
classifier = TextClassifier(category_col="category", text_cols=text_cols)
classifier.fit('allegro.csv')
