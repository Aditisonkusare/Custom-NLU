import pandas as pd
import mysql.connector
import os
from class_custom_train1 import CustomBertModel
from class_predefined import OpenIEExtractor
import pandas as pd

os.environ['CORENLP_HOME'] = '/home/smitesh22/.stanfordnlp_resources/stanford-corenlp-4.5.3'

connection = mysql.connector.connect(
    host="localhost",
    user="Admin",
    password="Admin",
    database="name_entity_recognition"
)

cursor = connection.cursor()

df = pd.read_csv("NER_utf8.csv")
pretrained_model = OpenIEExtractor()
finetuned_model = CustomBertModel(model_name="bert-base-uncased", num_labels=4, checkpoint_path='version_0/version_0/checkpoints/epoch=1-step=356.ckpt')

def reverse_sentence(df) -> list():
    reversed_sentences = []

    for sentences in df.Sentence:
        words = sentences[:len(sentences)-1].split(" ")[::-1]
        reversed_sentences.append(" ".join(words)+".")
    return reversed_sentences

reverse_sentences = reverse_sentence(df)

df['reverse_sentences'] = reverse_sentences   

pretrained_model = OpenIEExtractor()
finetuned_model = CustomBertModel(model_name="bert-base-uncased", num_labels=4, checkpoint_path='version_0/version_0/checkpoints/epoch=1-step=356.ckpt')

for index, row in df.iterrows():
    openie_prediction = pretrained_model.extract_relations(row.reverse_sentences)
    custom_bert_prediction = finetuned_model.predict_relation(row.reverse_sentences)
    finetune_accuracy = 0
    pretrain_accuracy = 0
    if custom_bert_prediction == row.manual_answer:
        finetune_accuracy = 1

    if row.pretrained_prediction_cleaned == row.manual_answer:
        pretrain_accuracy = 1

    insert_query = "INSERT INTO SENTENCE_PREDICTION (input_text, finetune_prediction, pretrained_prediction, pretrained_prediction_cleaned, finetune_accuracy, pretrained_accuracy, sentence_labels) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    values = (row.reverse_sentences, custom_bert_prediction, openie_prediction, row.pretrained_prediction_cleaned, finetune_accuracy, pretrain_accuracy, row.manual_answer)
    cursor.execute(insert_query, values)

connection.commit()
connection.close()

