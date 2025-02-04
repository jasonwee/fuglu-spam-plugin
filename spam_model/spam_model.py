import tensorflow as tf

from tflite_model_maker import (
    ExportFormat,
    model_spec,
    text_classifier,
)
from tflite_model_maker.text_classifier import DataLoader



assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')

#data_file = tf.keras.utils.get_file(fname='comment-spam.csv', origin='https://raw.githubusercontent.com/jasonwee/MyMachineLearningArtificialIntelligence/refs/heads/master/courses/build_a_comment_spam_machine_learning_model/lmblog_comments.csv', extract=False)
data_file = "./comment-spam.csv"

spec = model_spec.get('average_word_vec')
spec.num_words = 2000
spec.seq_len = 20
spec.wordvec_dim = 7

data = DataLoader.from_csv(
    filename=data_file,
    text_column='commenttext',
    label_column='spam',
    model_spec=spec,
    delimiter=',',
    shuffle=True,
    is_training=True)

train_data, test_data = data.split(0.9)

# Build the model
model = text_classifier.create(train_data,
                               model_spec=spec,
                               epochs=50, 
                               validation_data=test_data)

model.export(export_dir='./output/')
export_detail=False
if export_detail:
    model.export(export_dir='./output/', export_format=[ExportFormat.LABEL, ExportFormat.VOCAB])
