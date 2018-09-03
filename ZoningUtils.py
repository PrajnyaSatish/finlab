#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 01:04:06 2018

@author: prajnya
"""

import string
import re
import math
from collections import Counter


def get_lines_in_box(box, all_lines_in):
    selected_lines = []
    for line in all_lines_in:
        if line['l']>=box['l']-20 and line['r']<=box['r']+20 and line['t']>=box['t']-20 and line['b']<=box['b']+20:
            selected_lines.append(line)
    return selected_lines        

def get_words_in_box(box, all_words_in):
    selected_words = []
    for word in all_words_in:
        if word['l']>=box['l'] and word['r']<=box['r'] and word['t']>=box['t'] and word['b']<=box['b']:
            selected_words.append(word)
    return selected_words  

def get_all_words(curr_page):
    """ Function simply returns all the words with their features appended 
    to the list all_words_in_page.
    Inputs : curr_page - the dict format of the current word. """
    all_words_in_page = []
    for lines in curr_page['page_lines']:
        for ind_outer, word in enumerate(lines['words']):
            if lines[str(ind_outer)]['word'] not in string.punctuation:
                all_words_in_page.append(lines[str(ind_outer)])
    return all_words_in_page

def sort_two_d(work_page):
    """Sort words from top to bottom and then from left to right."""
    all_words_in_page = get_all_words(work_page)
    ## Perform a top-down sort of all the words in a page. This will allow us to easily
    ## split based on the space between consecutive lines.
    sorted_ud = sorted(all_words_in_page, key=lambda k: ("b" not in k, k.get('b', None)))
    ## Separate the lines in eaxh page of the xml file base on the distance between the line and
    ## It's previous line.
    prev_line_bottom = 0
    line_split = []
    new_line = []
    for word in sorted_ud:
        space_from_prev_line = word['t'] - prev_line_bottom
        if space_from_prev_line > 18:
            line_split.append(new_line)
            new_line = [word]
            prev_line_bottom = word['b']
        else:
            new_line.append(word)
    else:
        line_split.append(new_line)

    sorted_lr = [sorted(each_line, key=lambda k: ("l" not in k, k.get('l', None))) \
                 for each_line in line_split]
    sorted_lr = [line_e for line_e in sorted_lr if line_e]
    return sorted_lr

def select_words(box, words_in_page):
    """ Select the words from the entire page that fit into the specific box."""
    selected_lines = []
    if isinstance(words_in_page[0], list):
        for line in words_in_page:
            selected_words = []
            line_top = min(line, key=lambda k: ('t' not in k, k.get('t', None)))['t']
            line_b = max(line, key=lambda k: ('b' not in k, k.get('b', None)))['b']
            if line_top >= box['t'] and line_b <= box['b']:
                for word in line:
                    if word['l'] >= box['l'] and word['r'] <= box['r']:
                        selected_words.append(word)
            if selected_words:
                selected_lines.append(selected_words)
    else:
        for word in words_in_page:
            if word['t'] >= box['t'] and word['b'] <= box['b'] and word['l'] >= box['l'] and word['r'] <= box['r']:
                selected_lines.append(word)
    return selected_lines

def sort_linewise(list_of_words):
    sorted_ud = sorted(list_of_words, key=lambda k: ("b" not in k, k.get('b', None)))
    prev_line_bottom = 0
    line_split = []
    new_line = []
    for word in sorted_ud:
        space_from_prev_line = word['t'] - prev_line_bottom
        if space_from_prev_line > 20:
            line_split.append(new_line)
            new_line = [word]
            prev_line_bottom = word['b']
        else:
            new_line.append(word)
    else:
        line_split.append(new_line)
    
    sorted_lr = [sorted(each_line, key=lambda k: ("l" not in k, k.get('l', None))) \
                 for each_line in line_split]
    sorted_lr = [line_e for line_e in sorted_lr if line_e]
    return sorted_lr

def find_expected_col_num(info, p_info):
    select_cols = info[info['Columns'].notnull() & \
                     (info['Microfiche1'] <= int(p_info['microficheName'])) & \
                     (int(p_info['microficheName']) <= info['Microfiche2']) & \
                     (info['Year'] == int(p_info['year']))]
    if len(select_cols) == 1:
        num_cols_expected = int(select_cols['Columns'])
    elif len(select_cols) >= 2:
        if(int(select_cols.iloc[0]['Microfiche2'])==int(p_info['microficheName']) \
           and 0 < int(p_info['pageNum']) <= int(select_cols.iloc[0]['Page2'])):
            num_cols_expected = select_cols.iloc[0]['Columns']
        elif(int(select_cols.iloc[1]['Microfiche2'])==int(p_info['microficheName']) \
           and 0 < int(p_info['pageNum']) <= int(select_cols.iloc[1]['Page2'])):
            num_cols_expected = select_cols.iloc[1]['Columns']
        else:
            if isinstance(select_cols[-1:]['Columns'], int):
                num_cols_expected = int(select_cols[-1:]['Columns'])
            else:
                num_cols_expected = None
    elif len(select_cols) == 0:
        num_cols_expected = 1
    else:
        num_cols_expected = int(select_cols[-1:]['Columns'])
    return num_cols_expected


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)

def text_to_char_vec(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    string = ''.join(words)
    return Counter(string)


def cosine_similarity(moodys_string, to_check_string, sim_level='word'):
    if sim_level == 'char':
        vector1 = text_to_char_vec(moodys_string)
        vector2 = text_to_char_vec(to_check_string)
        
    if sim_level=='word':
        vector1 = text_to_vector(moodys_string)
        vector2 = text_to_vector(to_check_string)

    cosine_result = get_cosine(vector1, vector2)
    return cosine_result

cosine_similarity("Hi!", "Hi I'm here")
