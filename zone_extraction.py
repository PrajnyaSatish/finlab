#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:07:47 2018

@author: prajnya
"""

# In[]:

import os
import glob
import pandas as pd
import json

from ImagePlot import draw_rectangles
from ZoningUtils import find_expected_col_num
from OneColTemplate import one_col_page
from TwoColTemplate import two_col_page
from ThreeColTemplate import three_col_page


# In[]:

def template(page, num_cols_expected, year):
    minor_splits = []
    if  1920 <= int(year) <= 1929:
        if num_cols_expected == 1:
            OneColSplitter = one_col_page(page)
            major_blocks = OneColSplitter.get_major_blocks(gap_threshold=95)
            title_splits = OneColSplitter.title_splits_1_col(major_blocks)
            title_splits = OneColSplitter.close_gaps(title_splits)
            vertical_splits_out = OneColSplitter.vertical_splits_one_col(title_splits)
            vertical_splits = OneColSplitter.find_gap_intersections(vertical_splits_out)
            minor_splits = OneColSplitter.minor_horizontal_splits(vertical_splits)
        elif num_cols_expected == 2:
            TwoColSplitter = two_col_page(page)
            boxes = TwoColSplitter.partition_2_col()
            vertical_splits = TwoColSplitter.find_gap_intersections(boxes)
            title_splits = TwoColSplitter.title_splits_2_col(vertical_splits)
            minor_splits = TwoColSplitter.minor_horizontal_splits(title_splits)
        elif num_cols_expected == None:
            return []
        return minor_splits
    if 1940 <= int(year) < 2000:
        if num_cols_expected == 1:
            OneColSplitter = one_col_page(page)
            major_blocks = OneColSplitter.get_major_blocks(gap_threshold=55)
            closed_blocks = OneColSplitter.close_gaps(major_blocks)
            major_blocks = OneColSplitter.remove_moodys_string(closed_blocks)
            title_splits = OneColSplitter.title_splits_3_1_col(major_blocks)
            hor_splits = OneColSplitter.three_one_separation(title_splits)
            vertical_splits = OneColSplitter.vertical_splits_three_col(hor_splits)
            minor_splits = OneColSplitter.minor_horizontal_splits(vertical_splits)
            minor_splits = OneColSplitter.close_gaps(minor_splits)
            minor_splits = OneColSplitter.readjust_boxes(minor_splits)
            return minor_splits
        elif num_cols_expected == 3:
            ThreeColSplitter = three_col_page(page)
            major_blocks = ThreeColSplitter.get_major_blocks(gap_threshold=55)
            closed_blocks = ThreeColSplitter.close_gaps(major_blocks)
            major_blocks = ThreeColSplitter.remove_moodys_string(closed_blocks)
            vertical_splits = ThreeColSplitter.partition_3_col(major_blocks)
            minor_splits = ThreeColSplitter.minor_horizontal_splits(vertical_splits)
            minor_splits = ThreeColSplitter.readjust_boxes(minor_splits)
        elif num_cols_expected == None:
            return []
        return minor_splits
        
    
# In[]:
#page_num = 2046 
#year = str(1962)
#info = pd.read_excel(open('list_sections20180613.xlsx','rb'), sheet_name=0)
#with open('Pages/page_{0}_1962.json'.format(page_num+1), 'r') as rf:
#    working_page = json.load(rf)
#p_info = working_page['microfiche_details']
#num_cols_expected = find_expected_col_num(info, working_page['microfiche_details'])
#boxes_out = template(working_page, num_cols_expected, year)
#draw_rectangles([{'color':'violet', 'word_list': working_page['word_info']},
#                  {'color': 'seagreen', 'word_list': boxes_out}], \
#    'Data/Lines_1962/Line_Info_{0}'.format(page_num+1))

# In[]:
year = str(1983)

info = pd.read_excel(open('list_sections20180613.xlsx','rb'), sheet_name=0)

directory = 'Data'
parsed_word_file = glob.glob(os.path.join(directory,'*{0}*.json'.format(year)))[0]

with open(parsed_word_file) as pf:
    all_pages = json.load(pf)

selected_pages = list(range(50))
#selected_pages = [635, 644]
for page in selected_pages:
#    print(all_pages[page]['microfiche_details']['fileName'], end='  ')
    num_cols_expected = find_expected_col_num(info, all_pages[page]['microfiche_details'])
    boxes_out = template(all_pages[page], num_cols_expected, year)
    draw_rectangles([{'color':'violet', 'word_list': all_pages[page]['word_info']},
                      {'color': 'seagreen', 'word_list': boxes_out}], \
    'Data/Lines_1983/Line_Info_{0}'.format(all_pages[page]['pageCount']+1))
    print("Page {0} done".format(page+1)) 

