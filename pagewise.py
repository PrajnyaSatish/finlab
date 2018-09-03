#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:06:06 2018

@author: prajnya
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# In[]:
selected_page = 'page_20.json'
with open(selected_page, 'r') as rf:
    page_info = json.load(rf)
    
page_lines = page_info['page_lines']

# In[]:
def sort_two_d(work_page):
    """Sort words from top to bottom and then from left to right."""
    all_words_in_page = []

    for lines in work_page['page_lines']:
        for ind_outer, word in enumerate(lines['words']):
            all_words_in_page.append(lines[str(ind_outer)])
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
        if space_from_prev_line > 50:
            line_split.append(new_line)
            new_line = [word]
            prev_line_bottom = word['b']
        else:
            new_line.append(word)

    sorted_lr = [sorted(each_line, key=lambda k: ("l" not in k, k.get('l', None))) \
                 for each_line in line_split]
    sorted_lr = [line_e for line_e in sorted_lr if line_e]
    return sorted_lr


# In[]:
def mean_gap_between(horizontal_line_dict):
    """ Find mean gap between the horizontal lines."""
    empty_lines_list = []
    lines_list = []
    for el in horizontal_line_dict.items():
        if el[1] > 0:
            horizontal_line_dict[el[0]] = 0
            if lines_list:
                empty_lines_list.append(lines_list)
                lines_list = []
        else:
            lines_list.append(el[0])
            horizontal_line_dict[el[0]] = 1
    
    empty_line_sizes = [len(l) for l in empty_lines_list]
    mean = np.mean(empty_line_sizes)
    return mean, empty_lines_list


# In[]:

def left_right_bounds(left_lims, right_lims):
    left_freq_counts, bins = np.histogram(left_lims, bins=100, range=(min(left_lims), max(left_lims)))
    left_start_step = bins[(left_freq_counts.tolist()).index(max(left_freq_counts))]
    left_end_step = left_start_step+100
    fixed_left_lim = left_end_step
    for val in left_lims:
        if left_start_step <= val:
            if val < fixed_left_lim:
                fixed_left_lim = val

    right_freq_counts, bins = np.histogram(right_lims, bins=100, range=(min(right_lims), max(right_lims)+100))
    right_start_step = bins[(right_freq_counts.tolist()).index(max(right_freq_counts))]
    right_end_step = right_start_step+100
    fixed_right_lim = right_start_step
    for val in right_lims:
        if val <= right_end_step:
            if fixed_right_lim < val:
                fixed_right_lim = val
    return fixed_left_lim, fixed_right_lim

# In[]:
def select_words(box, words_in_page):
    selected_lines = []
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
    return selected_lines

# In[]:
    

def get_major_blocks(sorted_lr):
    """ ## First make the major horizontal splits so that where to split 
    vertically can be approximated.
    """
    hor_dict = {}
    left_lims = []
    right_lims = []
    top_most = []
    b_most = []
    details_out = []
    for print_line in sorted_lr:
        left_lims.append(min(print_line, key=lambda k: ('l' not in k, k.get('l', None)))['l'])
        right_lims.append(max(print_line, key=lambda k: ('r' not in k, k.get('r', None)))['r'])
        
        top_most.append(min(print_line, key=lambda k: ('t' not in k, k.get('t', None)))['t'])
        b_most.append(max(print_line, key=lambda k: ('b' not in k, k.get('b', None)))['b'])
    page_top = min(top_most)
    page_bottom = max(b_most) 
    
    for num in range(page_top, page_bottom+1):
        hor_dict[num] = 0
        
    for line in sorted_lr:
        for word in line:
            for wnum in range(word['t'], word['b']+1):
                hor_dict[wnum] += 1
    
    left_bound, right_bound = left_right_bounds(left_lims, right_lims)     
    mean_gap, empty_lines_list = mean_gap_between(hor_dict)
    
    top_bounds = [page_top]
    bottom_bounds = []
    for lines_set in empty_lines_list:
        if len(lines_set) > mean_gap:
            bottom_bounds.append(lines_set[-1])
            top_bounds.append(lines_set[0])
    bottom_bounds.append(page_bottom)
    for ind in range(len(top_bounds)):
        box_limits_dict = {'t': top_bounds[ind], 'b': bottom_bounds[ind],
                           'l': left_bound, 'r': right_bound,
                           'w': right_bound - left_bound, 
                           'h': bottom_bounds[ind] - top_bounds[ind]}
        lines_in_box = select_words(box_limits_dict, sorted_lr)
        details_out.append({'box_limits': box_limits_dict, 'lines_in_block': lines_in_box})
    return details_out

sorted_lr = sort_two_d(page_info)
major_h_boxes = get_major_blocks(sorted_lr)
        

# In[]:
#plt.figure(figsize=[25,10])
#x_axis = list(hor_dict.keys())
#y_axis = list(hor_dict.values())
#plt.bar(range(x_axis[0],x_axis[0]+len(x_axis)), y_axis)
#plt.savefig('page_19.jpg', bbox_inches='tight')
# In[]:
#import collections
#import matplotlib.pyplot as plt
#
#tab_dict = {}
#for el in leftmost:
#    if el not in tab_dict:
#        tab_dict[el]=1
#    else:
#        tab_dict[el] += 1
#od = dict(collections.OrderedDict(sorted(tab_dict.items(), key=lambda t: t[0])))
#x_axis = list(od.keys())
#print(x_axis)
#y_axis = list(od.values())
#print(y_axis)
#plt.figure(figsize=(20,10))
#plt.bar(range(len(x_axis)), y_axis)
#plt.xticks(range(len(x_axis)), x_axis) 
#plt.savefig('left axis.png', bbox_inches='tight')
    
    
# In[]:
