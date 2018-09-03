#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:57:47 2018

@author: prajnya
"""
import os
import glob
import re
import pathlib
import pandas as pd
import json

from ZoningUtils import find_expected_col_num
from word_parser import parser
from OneColTemplate import one_col_page
from TwoColTemplate import two_col_page
from ThreeColTemplate import three_col_page

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

def write_string(file_name_dets, boxes_to_write):
    to_write_dict = []
    if boxes_to_write:
        for zone in boxes_to_write:
            to_write_dict.append({'Filename': file_name_dets['fileName'][:-4],
                                  'Num_Zones':len(boxes_to_write),
                                  'l':zone['l'], 't': zone['t'], 'r': zone['r'], 'b': zone['b']})
    else:
        to_write_dict.append({'Filename': file_name_dets['fileName'][:-4],
                                  'Num_Zones':0,
                                  'l':0, 't': 0, 'r': 0, 'b': 0})
    return pd.DataFrame.from_dict(to_write_dict)

if __name__ == '__main__':
    info = pd.read_excel(open('list_sections20180613.xlsx','rb'), sheet_name=0)
    year = str(input("Enter the year you want to work on: "))
    file_location = 'Data/WholeManuals/'
#    filenames = glob.glob(os.path.join(file_location,'Moodys{0}*.XML'.format(year)))
    filenames = ['Data/Brightness_70/Moodys1983.XML']
    for file_in in filenames:
        pd_table = pd.DataFrame()
        print('\n'+file_in)
        filename = re.split('/',file_in)[-1]
        all_pages = parser(file_in)
#        selected_pages = list(range(len(all_pages)))
        selected_pages = list(range(0, 10))
        for page_num in selected_pages:
            print(all_pages[page_num]['microfiche_details'])
            num_cols_expected = find_expected_col_num(info, all_pages[page_num]['microfiche_details'])
            
            boxes = template(all_pages[page_num], num_cols_expected, year)
            print(boxes)
            pd_table = pd_table.append(write_string(all_pages[page_num]['microfiche_details'], boxes))
            print("Page {0} done".format(page_num+1))
        outfile_location = './'
#        outfile_location = '/projects/asbe9894/Step2_WithUserZones/Input/UserZones/Run007/'
        ## Create directory if it doesn't exist
#        pathlib.Path(outfile_location).mkdir(exist_ok=True)
        ## Create a direcotry for each year and bightness selected
        directory = outfile_location
        outfile_name = filename.replace(".XML","")
        out_file_path = os.path.join(directory, outfile_name)
        pd_table.to_csv(out_file_path, columns=['Filename','Num_Zones','l','t','r','b'], index=False)
        os.chmod(out_file_path, 0o777)
