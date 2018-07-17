import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

MODEL_NAME = 'dnn_estimator_hub'
MODE_DIR = os.path.join(os.getcwd(), MODEL_NAME)

PREDICTION_DICT = {b'0': 'neutral',  b'1': 'sad',
                   b'2': 'happy',  b'3': 'anger'}

embedded_text_feature_column = hub.text_embedding_column(
        key="sentence",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 500],
        feature_columns=[embedded_text_feature_column],
        n_classes=4,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.003),
        model_dir=MODE_DIR)

single_text_df = pd.DataFrame({'sentence': ["i am so happy"]})
predict_input_fn = tf.estimator.inputs.pandas_input_fn(single_text_df, shuffle=False)

prediction = list(estimator.predict(input_fn=predict_input_fn))[0]['classes'][0]

print(PREDICTION_DICT[prediction])