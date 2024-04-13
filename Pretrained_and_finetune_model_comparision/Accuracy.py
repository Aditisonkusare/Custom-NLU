import pandas as pd
import mysql.connector

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

cursor = connection.cursor()

sql_query = "SELECT * FROM SENTENCE_PREDICTION;"
cursor.execute(sql_query)

records = cursor.fetchall()

columns = [desc[0] for desc in cursor.description]

df = pd.DataFrame(records, columns=columns)

print("Finetune accuracy", df.finetune_accuracy.sum()/df.finetune_accuracy.count())
print("Pretrained accuracy", df.pretrained_accuracy.sum()/df.pretrained_accuracy.count())

sql_query = "SELECT * FROM SENTENCE_PREDICTION;"

df = pd.read_sql(sql_query, connection)

connection.close()

df.to_csv('output.csv', index=False)