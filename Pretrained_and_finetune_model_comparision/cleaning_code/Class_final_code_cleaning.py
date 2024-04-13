# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:42:36 2023


"""

import re
import pandas as pd

class CleaningCSV:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, encoding='latin-1')
        #self.df['processed text'] = ''
        self.df['cleaned text'] = ''

    def remove_dot_between_tags(self, text):
        regex_pattern = r'\[E\d\](.*?)\[/E\d\]'
        find_text_between_tags = re.findall(regex_pattern, text)

        for index, word in enumerate(find_text_between_tags):
            if '.' in word:
                word = word.replace('.', ' ')
                text = text.replace(find_text_between_tags[index], word)
        return text

    def remove_adjacent_dots(self, text):
        try:
            dot_indices = [i for i, char in enumerate(text) if char == '.']
            for i in range(len(dot_indices) - 1):
                index1 = dot_indices[i]
                index2 = dot_indices[i + 1]
                distance = index2 - index1

                if distance <= 20:
                    text = text[:index1] + ' ' + text[index1 + 1:index2] + ' ' + text[index2 + 1:]
                elif index1 < 30:
                    text = text[:index1] + ' ' + text[index1 + 1:]

        except:
            pass
        return text

    def add_dot_at_end(self, text):
        if not text.endswith('.'):
            text += '.'
        return text

    def split_sentences(self, text):
        # Split the list of sentences into separate sentences
        split_sentences_list = re.split(r'(?<=[.!?])\s+', text)
        return split_sentences_list

    def take_sentence_with_two_entities(self, text, tags):
        # Selecting the sentences which contain both tags
        selected_sentences = [s for s in text if all(tag in s for tag in tags)]
        cleaned_sentence1 = ' '.join(selected_sentences)
        return cleaned_sentence1

    def process_text(self):
        for index, row in self.df.iterrows():
            text = row['sentences']
            text = self.add_dot_at_end(self.remove_adjacent_dots(self.remove_dot_between_tags(text)))
            split_sentences_list = self.split_sentences(text)
            tags_to_select = ['[E1]', '[E2]']
            cleaned_text = self.take_sentence_with_two_entities(split_sentences_list, tags_to_select)
            self.df.at[index, 'cleaned text'] = cleaned_text

    def save_processed_data(self, output_file):
        self.df.to_csv(output_file, index=False)

    def keep_unique_sentences(self,column_name):
        self.df['temp_sentence'] = self.df['sentences'].apply(lambda x: re.sub(r'\W', '', x.lower()))
        self.df['is_duplicate'] = self.df.duplicated(subset=['temp_sentence'], keep='first')
        df_unique = self.df[~self.df['is_duplicate']]
        df_unique = df_unique.drop(columns=['temp_sentence', 'is_duplicate'])
        return df_unique




obj = CleaningCSV('data.csv')
obj.process_text()
obj.save_processed_data('Cleaned sentences.csv')
df_unique = obj.keep_unique_sentences('sentences')
df_unique.to_csv('Cleaned sentences.csv', index=False)

