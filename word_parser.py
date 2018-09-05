#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:16:32 2018

@author: prajnya
"""

# In[0]:
import re

# In[]:
## Change the current working directory
#os.chdir('/home/prajnya/Desktop/fin_lab_workstation')
#
#
#list_info_file = 'Data/list_sections20180318.xlsx'
#xl = pd.ExcelFile(list_info_file)
#
#list_info = xl.parse(xl.sheet_names[0])

# In[]

def file_reader(filename):
    try:
        with open(filename, encoding='utf-8') as read_file:
            file_content = read_file.readlines()
    except UnicodeDecodeError:
        with open(filename, encoding='utf-16-le') as read_file:
            file_content = read_file.readlines()
    return file_content

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

def get_microfiche_name(str_in):
    """ Returns year, microfiche name and filename """
    split_string = str_in.split("\"")
    if ".tif" in split_string[1]:
        split_again = split_string[1].split("/")
        year = split_again[-3][10:14]
        microfiche = split_again[-2][:4]
        file_name = split_again[-1]
        page_num = file_name[-8:-4]
        dict_info = {'year': year, 'microficheName': microfiche,
                     'fileName': file_name, 'pageNum': page_num}
        return dict_info
    return ''

def parse_word_details(page_all_words):
    all_word_feats = []
    for line in page_all_words:
        line = line.rstrip()
        split_1 = re.split('(</wd>)|(</run>)', line)
        split_2 = re.split(r'[<>|\n]', split_1[0])
        word = split_2[-1]
        word = word.replace('&apos;', "'")
        word = word.replace('&amp;', "&")
        word_feats = split_2[1]
        word_props = get_properties(word_feats)
        word_props['word'] = word
        all_word_feats.append(word_props)
    return all_word_feats

def parse_line_details(file_content):
    page_lines = []
    page_count = -1
    page_details = []
    page_all_words = []
    for line in file_content:
        if line.startswith('<source '):
            page_count += 1
            microfiche_name = get_microfiche_name(line)
            page_dict = {'pageCount':page_count, 
                         'microfiche_details':microfiche_name, 
                         'line_info': [],
                         'word_info': []}
            page_line_properties = [get_properties(sel_line) for sel_line in page_lines]
            page_word_properties = parse_word_details(page_all_words)
            if page_line_properties:
                page_details[-1]['line_info'] = page_line_properties
            if page_word_properties:
                page_details[-1]['word_info'] = page_word_properties
            page_lines = []
            page_all_words = []
            page_details.append(page_dict)
        if line.startswith('<ln '):
            page_lines.append(line)
        if line.startswith('<wd '):
            page_all_words.append(line)
    # Finally:
    page_line_properties = [get_properties(sel_line) for sel_line in page_lines]
    page_word_properties = parse_word_details(page_all_words)
    if page_line_properties:
        page_details[-1]['line_info'] = page_line_properties
    if page_word_properties:
        page_details[-1]['word_info'] = page_word_properties
    return page_details

# In[]:
def write_to_json(all_pages, year):
    """Write the file to json file so that you don't need to run it each time the program loads.
    It's only for convenient code manipulation and must be commented while running on all the files.
    """
    import json

    write_file = 'Data/parsed_word_file_{0}.json'.format(year)
    with open(write_file, 'w') as w_f:
        json.dump(all_pages, w_f, indent=4)
    
# In[]:  
def parser(filename):
    """ Main parser that converts XML document into a dictionary of words and their features for 
    each page."""
    try:
        with open(filename, encoding='utf-8') as read_file:
            file_content = read_file.readlines()
    except UnicodeDecodeError:
        with open(filename, encoding='utf-16-le') as read_file:
            file_content = read_file.readlines()
    all_pages = parse_line_details(file_content)
    return all_pages

# In[]: 
if __name__ == "__main__":
    year = str(input("Enter the year you want to work on: "))
#    filename = 'xml_output_201802/Industrials_{0}_raw_image.XML'.format(year)
    filename = 'Data/Brightness_70/Moodys{0}.XML'.format(year)
#    filename = 'Data/Brightness_70/Industrial048.XML'
    all_pages = parser(filename)
    write_to_json(all_pages, year)
