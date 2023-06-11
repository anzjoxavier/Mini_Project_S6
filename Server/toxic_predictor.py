import tensorflow as tf
import pandas as pd
from keras.layers import TextVectorization
MAX_FEATURES = 200000
df =  pd.read_csv(r'E:/ANZ JO/Mini Project/Sarcasm Analyser/toxicity.csv')
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
X = df['comment_text']
vectorizer.adapt(X.values)
def predict_toxicity(s):
    vectorized_comment = vectorizer([s])
    model=load_model()
    results = model.predict(vectorized_comment)
    text = {}
    for idx, col in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult']):
        text[col]=1 if str(results[0][idx]>0.5)=='True' else 0
    return text


def load_model():
    model_saved = tf.keras.models.load_model('E:/ANZ JO/Mini Project/Sarcasm Analyser/toxic_model')
    print("Done Loading")
    return model_saved



if __name__=="__main__":
    # load_model()
    print(predict_toxicity('hey i freaken hate you! I will kill you'))
