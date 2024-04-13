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

if connection.is_connected():
    print('Connected Successfully')
else:
    print('Failed to connect')
    exit()


table_exists_query = "SHOW TABLES LIKE 'SENTENCE_PREDICTION'"
cursor.execute(table_exists_query)

if not cursor.fetchone():
    table_creation_query = """
       CREATE TABLE SENTENCE_PREDICTION (
            id INT AUTO_INCREMENT PRIMARY KEY,
            input_text VARCHAR(255),
            finetune_prediction VARCHAR(255),
            pretrained_prediction VARCHAR(255),
            pretrained_prediction_cleaned VARCHAR(255),
            finetune_accuracy INT,
            pretrained_accuracy INT,
            sentence_labels TEXT
        );
        """
    
    cursor.execute(table_creation_query)
    cursor.fetchall()
    
    print("Predictions Table created")
    connection.commit()
else:
    print("Table exists")


df = pd.read_csv("NER_utf8.csv")

pretrained_model = OpenIEExtractor()
finetuned_model = CustomBertModel(model_name="bert-base-uncased", num_labels=4, checkpoint_path='version_0/version_0/checkpoints/epoch=1-step=356.ckpt')

for index, row in df.iterrows():
    openie_prediction = pretrained_model.extract_relations(row.Sentence)
    custom_bert_prediction = finetuned_model.predict_relation(row.Sentence)
    finetune_accuracy = 0
    pretrain_accuracy = 0
    if custom_bert_prediction == row.manual_answer:
        finetune_accuracy = 1

    if row.pretrained_prediction_cleaned == row.manual_answer:
        pretrain_accuracy = 1

    insert_query = "INSERT INTO SENTENCE_PREDICTION (input_text, finetune_prediction, pretrained_prediction, pretrained_prediction_cleaned, finetune_accuracy, pretrained_accuracy, sentence_labels) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    values = (row.Sentence, custom_bert_prediction, openie_prediction, row.pretrained_prediction_cleaned, finetune_accuracy, pretrain_accuracy, row.manual_answer)
    cursor.execute(insert_query, values)

connection.commit()
connection.close()

