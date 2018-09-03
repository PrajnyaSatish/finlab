#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 18:28:22 2018

@author: prajnya
"""
import re

def file_reader(filename):
    try:
        with open(filename, encoding='utf-8') as read_file:
            file_content = read_file.readlines()
    except UnicodeDecodeError:
        with open(filename, encoding='utf-16-le') as read_file:
            file_content = read_file.readlines()
    return file_content


year = str(1928)
filename = 'xml_output_201802/Industrials_{0}_raw_image.XML'.format(year)
file_content = file_reader(filename)

def get_microfiche_name(str_in):
    """ Returns year, microfiche name and filename """
    split_string = str_in.split("\"")
    if ".tif" in split_string[1]:
        split_again = split_string[1].split("/")
        year = split_again[-3][-4:]
        microfiche = split_again[-2]
        file_name = split_again[-1]
        page_num = file_name[-8:-4]
        dict_info = {'year': year, 'microficheName': microfiche,
                     'fileName': file_name, 'pageNum': page_num}
        return dict_info
    return ''

def get_properties(line):
    prop_dict = {}
    stripped = re.sub(r'[<>|\n]', '', line)
    wspace_split = stripped.split(' ')[1:]
    for el in wspace_split:
        if '=' in el:
            prop, val = el.split('=')
            val = val.strip('"')
            if val.isdigit():
                prop_dict[prop] = int(val)
            else:
                prop_dict[prop] = str(val)
    return prop_dict

#page_all_words = []
#page_count = -1
#page_details = []
#for line in file_content:
#    if line.startswith('<source '):
#        page_count += 1
#        microfiche_name = get_microfiche_name(line)
#        page_dict = {'pageCount':page_count, 
#                     'microfiche_details':microfiche_name, 
#                     'line_info': []}
#        page_line_properties = [get_properties(sel_line) for sel_line in page_all_words]
#        if page_line_properties:
#            page_details[-1]['line_info'] = page_line_properties
#        page_lines = []
#        page_details.append(page_dict)
#    if line.startswith('<ln '):
#        page_lines.append(line)
#page_line_properties = [get_properties(sel_line) for sel_line in page_all_words]
#if page_line_properties:
#    page_details[-1]['line_info'] = page_line_properties
# In[]:
page_all_words = []
for line in file_content[:100]:
    if line.startswith('<wd '):
        page_all_words.append(line)
        
def parse_word_details(page_all_words):
    for line in page_all_words:
        line = line.rstrip()
        split_1 = re.split('(</wd>)|(</run>)', line)
        split_2 = re.split(r'[<>|\n]', split_1[0])
        word = split_2[-1]
        word_feats = split_2[1]
        word_props = get_properties(word_feats)
        word_props['word'] = word
    return word_props        
        
        
        
        
        
        
        