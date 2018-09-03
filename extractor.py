#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 18:23:11 2018

@author: prajnya
"""
# In[0]:
import json
import numpy as np
import pandas as pd
import copy

from ImagePlot import draw_rectangles

# In[]:

def sort_two_d(work_page):
    """Sort words from top to bottom and then from left to right."""
    all_words_in_page = []

    for lines in work_page['page_lines']:
        for ind_outer, word in enumerate(lines['words']):
            all_words_in_page.append(lines[ind_outer])
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
def select_words(box, words_in_page):
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
def get_text_limits(set_of_words):
    left_lims = []
    right_lims = []
    top_most = []
    b_most = []
    for line in set_of_words:
        left_lims.append(min(line, key=lambda k: ('l' not in k, k.get('l', None)))['l'])
        right_lims.append(max(line, key=lambda k: ('r' not in k, k.get('r', None)))['r'])
        
        top_most.append(min(line, key=lambda k: ('t' not in k, k.get('t', None)))['t'])
        b_most.append(max(line, key=lambda k: ('b' not in k, k.get('b', None)))['b'])
    page_top = min(top_most)
    page_bottom = max(b_most) 
    return left_lims, right_lims, page_top, page_bottom
    
    
    

# In[]:
    

def get_major_blocks(sorted_lr):
    """ ## First make the major horizontal splits so that where to split 
    vertically can be approximated.
    """
    hor_dict = {}
    details_out = []
    
    if sorted_lr:
        left_lims, right_lims, page_top, page_bottom = get_text_limits(sorted_lr)
        
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
                bottom_bounds.append(lines_set[0])
                top_bounds.append(lines_set[-1])
        bottom_bounds.append(page_bottom)
        for ind in range(len(top_bounds)):
            box_limits_dict = {'t': top_bounds[ind], 'b': bottom_bounds[ind],
                               'l': left_bound, 'r': right_bound,
                               'w': right_bound - left_bound, 
                               'h': bottom_bounds[ind] - top_bounds[ind],
                               'color': 'seagreen', 'num_col':'one'}
            lines_in_box = select_words(box_limits_dict, sorted_lr)
            details_out.append({'box_limits': box_limits_dict, 'lines_in_block': lines_in_box})
    return details_out

 

# In[]:
def select_words_in_box(orig_box, left_bound, right_bound):
    """ Returns only the words that fit in the new left and rigt bounds."""
    block_lines = []
    for line in orig_box:
        final_words_in_line = []
        for word in line:
            if word['l'] >= left_bound and word['r'] < right_bound:
                final_words_in_line.append(word)
        block_lines.append(final_words_in_line)
    return block_lines


# In[]:

def partition_3_columns(y_selected, box_limits):
    distance_1 = abs(y_selected[1][1]-y_selected[0][1])
    distance_2 = abs(y_selected[2][1]-y_selected[1][1])
    distance_3 = abs(box_limits['r'] - y_selected[2][1])
    block_width = box_limits['w']
    box_left = box_limits['l']
    one_third = round(block_width/3, 3)
    block_1_size_flag = one_third-250 < distance_1 < one_third+250
    block_2_size_flag = one_third-250 < distance_2 < one_third+250
    block_3_size_flag = one_third-250 < distance_3 < one_third+250
    first_block_start = box_left-100 < y_selected[0][1] < box_left+100
    if block_1_size_flag and block_2_size_flag and block_3_size_flag and first_block_start:
        box_1 = {'t': box_limits['t'], 'b': box_limits['b'],
                 'l': y_selected[0][1], 'r': y_selected[1][1],
                 'w': distance_1, 'h': box_limits['h'],
                 'color':'orange'}
        box_2 = {'t': box_limits['t'], 'b':box_limits['b'],
                 'l': y_selected[1][1], 'r': y_selected[2][1],
                 'w': distance_2, 'h': box_limits['h'],
                 'color':'orange'}
        box_3 = {'t': box_limits['t'], 'b': box_limits['b'],
                 'l': y_selected[2][1], 'r': box_limits['r'],
                 'w': distance_3, 'h': box_limits['h'],
                 'color':'orange'}
        return [box_1, box_2, box_3]
    return [box_limits]


# In[]:

def partition_2_columns(y_selected, box_limits, lines_in_block):
    gap_lower_limit = 20
    gap_upper_limit = 500
    gap_between_flag = False
    gap = 0
    block_width = box_limits['r'] - box_limits['l']
    box_left = box_limits['l']
    one_half = round(block_width/2, 3)
    if len(y_selected) > 1:
        y_sel = [y_selected[0]]
        y_sel_next = [el for el in y_selected if el[1] > one_half-800]
        y_sel.extend(y_sel_next)
        y_selected = y_sel
    if len(y_selected) > 1:
        block_2_left_lim = y_selected[1][1]
        block_1_left_lim = y_selected[0][1]
        distance_1 = abs(block_2_left_lim-y_selected[0][1])
        distance_2 = abs(box_limits['r']- block_2_left_lim)
        block_1_size_flag = one_half-800 < distance_1 < one_half+800
        if block_1_size_flag:
            block_1_lines = select_words_in_box(lines_in_block, block_1_left_lim, block_2_left_lim)
            right_lim_block_1 = max([line[-1]['r'] if line else 0 for line in block_1_lines])
            gap = block_2_left_lim - right_lim_block_1
            gap_between_flag = gap_lower_limit < gap < gap_upper_limit
        block_2_size_flag = one_half-800 < distance_2 < one_half+800
        first_block_start = box_left-100 < y_selected[0][1] < box_left+250
#        print("D1 = {0}, D2 = {1}".format(distance_1,distance_2))
#        print("One half = {0}, Left of Box = {1}".format(one_half, box_left))
#        print(y_selected, block_1_size_flag, block_2_size_flag, first_block_start)
        if block_1_size_flag and block_2_size_flag and first_block_start:
            box_1 = {'t': box_limits['t'], 'b': box_limits['b'],
                     'l': box_limits['l'], 'r': right_lim_block_1,
                     'w': distance_1, 'h': box_limits['h'],
                     'color':'brown',
                     'num_col': 'two'}
            box_2 = {'t': box_limits['t'], 'b': box_limits['b'],
                     'l': block_2_left_lim, 'r': box_limits['r'],
                     'w': distance_2, 'h': box_limits['h'],
                     'color':'brown',
                     'num_col': 'two'}
            return [box_1, box_2]
#    elif len(y_selected) == 1:
#        box_1 = {'t': box_limits['t'], 'b': box_limits['b'],
#                 'l': box_limits['l'], 'r': y_selected[0][1],
#                 'w': y_selected[0][1] - box_limits['l'], 'h': box_limits['h'],
#                 'color':'brown', 'num_col': 'two'}
#        box_2 = {'t': box_limits['t'], 'b': box_limits['b'],
#                 'l': y_selected[0][1], 'r': box_limits['r'],
#                 'w': box_limits['r'] - y_selected[0][1], 'h': box_limits['h'],
#                 'color':'brown', 'num_col': 'two'}
#        return [box_1, box_2]
    box_limits.update({'num_col':'one'})
    return [box_limits]

    
# In[]:
def get_all_words(curr_page):
    """ Function simply returns all the words with their features appended 
    to the list all_words_in_page.
    Inputs : curr_page - the dict format of the current word. """
    all_words_in_page = []
    for lines in curr_page['page_lines']:
        for ind_outer, word in enumerate(lines['words']):
                all_words_in_page.append(lines[ind_outer])
    return all_words_in_page


# In[]:
def check_for_3_columns(word_left_in_block, left_lims, right_lims, page_num, block_num, block):
    """ Function prepares all the variables necessary to check if there are three columns in the block.
    """
    binwidth_3 = 45 
    freq_thresh = 0.5
    num_lines_in_block = len(block['lines_in_block'])
    try:
        y_freq, y_ranges = np.histogram(word_left_in_block, \
                                            range=(left_lims, right_lims), \
                                            bins=np.arange(min(word_left_in_block), \
                                   max(word_left_in_block) + binwidth_3, binwidth_3))
    except:
        y_freq = [];y_ranges = []
    
    y_range_top_4 = sorted(zip(y_freq, y_ranges), reverse=True)[:4]
    y_range_sorted = sorted(y_range_top_4, key=lambda x: x[1])
    y_selected = [x for x in y_range_sorted if x[0] > (freq_thresh*num_lines_in_block)]
    if len(y_selected) >= 3:
        block = partition_3_columns(y_selected, block['box_limits'])
        return block
    return {}


# In[]:
    
def check_for_2_columns(word_left_in_block, left_lims, right_lims, page_num, block_num, block):
    binwidth_2 = 60
    freq_thresh = 0.51
    half_mark_variance = 200
    num_lines_in_block = len(block['lines_in_block']) 
    try:
        y_freq, y_ranges = np.histogram(word_left_in_block, \
                                    range=(left_lims, right_lims), \
                                    bins=np.arange(min(word_left_in_block), \
                           max(word_left_in_block) + binwidth_2, binwidth_2))
    except:
        y_freq = []; y_ranges = []
    y_range_top_4 = sorted(zip(y_freq, y_ranges), reverse=True)[:4]
    y_range_sorted = sorted(y_range_top_4, key=lambda x: x[1])
    y_selected = [x for x in y_range_sorted if x[0] >= (freq_thresh*num_lines_in_block)]
    if len(y_selected) >= 2:
        block = partition_2_columns(y_selected, block['box_limits'], block['lines_in_block'])
        return block
    elif len(y_selected) == 1:
        block_width = block['box_limits']['r'] - block['box_limits']['l']
        one_half = round(block_width/2, 3)
        block_half = block['box_limits']['l']+one_half
        half_mark_check = block_half - half_mark_variance < y_selected[0][1] < block_half + half_mark_variance
        if y_selected[0][0] > freq_thresh*num_lines_in_block and half_mark_check:
            block = partition_2_columns(y_selected, block['box_limits'], block['lines_in_block'])
            return block
    if num_lines_in_block > 30:
        freq_thresh_new = 0.35
        y_selected = [x for x in y_range_sorted if x[0] > (freq_thresh_new*num_lines_in_block)]
        if len(y_selected) >= 2:
            block = partition_2_columns(y_selected, block['box_limits'], block['lines_in_block'])
            return block
        elif len(y_selected) == 1:
            block_width = block['box_limits']['r'] - block['box_limits']['l']
            one_half = round(block_width/2, 3)
            block_half = block['box_limits']['l']+one_half
            half_mark_check = block_half - half_mark_variance < y_selected[0][1] < block_half + half_mark_variance
            if y_selected[0][0] > freq_thresh_new*num_lines_in_block and half_mark_check:
                block = partition_2_columns(y_selected, block['box_limits'], block['lines_in_block'])
                return block
    return {}


# In[]:
    
def check_table(block_lines, block_width):
    """ Check if a block is a block of tables or of text."""
#    average_words_per_line=24
#    total_num_words = 0
    ratio_threshold = 0.55
    actual_num_chars = 0
    all_char_ws = []
    num_lines_in_block = len(block_lines)
    for line in block_lines:
        cas = []
#        total_num_words += len(line)
        for word in line:
            if word['word']:
                actual_num_chars += len(word['word'])
                char_w = float(word['r']-word['l'])/len(word['word'])
                cas.append(round(char_w, 2))
        all_char_ws.extend(cas)
    average_char_width = np.mean(all_char_ws)
    expected_num_chars = (float(block_width)/average_char_width)*num_lines_in_block
#    expected_word_count = average_words_per_line*num_lines_in_block
    ratio = actual_num_chars/expected_num_chars
    if ratio < ratio_threshold:
        return True
    else:
        return False
        
# In[]:


def get_major_vertical_splits(working_page, page_num, current_working_page):

    box_limits_vertical_splits = []
    for block_num, block in enumerate(working_page):
        num_lines_in_block = len(block['lines_in_block'])
        
        ## If there are more than 1 lines in the block,
        if num_lines_in_block > 1:
            right_lims = max([line[-1]['r'] if line else 0 for line in block['lines_in_block']])
            left_lims = min([line[0]['l']  if line else 0 for line in block['lines_in_block']])
            word_left_in_block = [w['l'] for lb in block['lines_in_block'] for w in lb]
            block_width = block['box_limits']['r'] - block['box_limits']['l']
            table_check = check_table(block['lines_in_block'], block_width)
            if not table_check:
                block_check_3 = check_for_3_columns(word_left_in_block, left_lims, \
                                                    right_lims, page_num, block_num, block)
                if block_check_3:
                    box_limits_vertical_splits.extend(block_check_3)
                
                block_check_2 = check_for_2_columns(word_left_in_block, left_lims, right_lims, \
                                                    page_num, block_num, block)
                if block_check_2:
                    box_limits_vertical_splits.extend(block_check_2)
    
                else: box_limits_vertical_splits.append(block['box_limits'])
            
            #------------------------ *************** ---------------------------------#
            else: box_limits_vertical_splits.append(block['box_limits'])    
        else: box_limits_vertical_splits.append(block['box_limits'])
    ## Recheck so that boxes don't repeat.
    box_splits = []
    for box in box_limits_vertical_splits:
        if box not in box_splits:
            box_splits.append(box)
    return box_splits


    
    



# In[]:
def xml_out_string(boxes):
    header = '<?xml version="1.0" encoding="UTF-16"?>\n'
    zone_num = '<Zones znum="{0}">\n'.format(len(boxes))
    string_out = header+zone_num
    for box in boxes:
        bounding_box_string = "<BBox left=\"{0}\" top=\"{1}\" right=\"{2}\" bottom=\"{3}\"/>\n".format(\
                                            box['l'],box['t'], box['r'], box['b'])
        next_string_1 = "<Props type=\"flow\" clr=\"0xffffff\" chk_ctrl=\"0xb0000\" fm=\"1\" rm=\"13\" "
        next_string_2 = "filt=\"0x1f\" langs=\"0,-1\"/>\n"
        zone_string = "<Zone\n>"+bounding_box_string+next_string_1+next_string_2+"</Zone>\n"
        string_out += zone_string
    return string_out


# In[]:
def major_vertical_split_2_col(divided_page):
    final_boxes = []
    half_mark_variance = 200
    for ind, box in enumerate(divided_page):
        bounds = box['box_limits']
        if len(box['lines_in_block']) > 1:
            block_width = box['box_limits']['r'] - box['box_limits']['l']
            table_check = check_table(box['lines_in_block'], block_width)
            if table_check:
                bounds.update({'color':'seagreen', 'num_col':'one'})
                final_boxes.append(bounds)
                continue
            text_half = bounds['l'] + (float(bounds['r'] - bounds['l'])/2)
            vertical_splits = [text_half+el for el in list(range((-half_mark_variance), (half_mark_variance)))]
            intersect_count_list = []
            for split_line in vertical_splits:
                intersect_count = 0
                for line in box['lines_in_block']:
                    for word in line:
                        if word['l'] < split_line < word['r']:
                            intersect_count+=1
                intersect_count_list.append(intersect_count)
            masked_list = [el[1] for el in zip(intersect_count_list, vertical_splits) if 0<= el[0]<3]
            if masked_list:
                vertical_split_line = np.median(masked_list)
                box_1 = {'t': bounds['t'], 'b': bounds['b'],
                         'l': bounds['l'], 'r': vertical_split_line,
                         'w': vertical_split_line-bounds['l'], 'h': bounds['b']-bounds['t'],
                         'color':'yellow', 'num_col': 'two'}
                box_2 = {'t': bounds['t'], 'b': bounds['b'],
                         'l': vertical_split_line, 'r': bounds['r'],
                         'w': bounds['r']-vertical_split_line, 'h': bounds['b']-bounds['t'],
                         'color':'orange', 'num_col': 'two'}
                final_boxes.extend([box_1, box_2])
            else:
                bounds.update({'color':'seagreen', 'num_col':'one'})
                final_boxes.append(bounds)
                ## Recheck to make sure there is no overlap. If there is, split 
                ## the boxes at line which overlaps
        else:
            bounds.update({'color':'seagreen', 'num_col':'one'})
            final_boxes.append(bounds)
    return final_boxes

# In[]:
def find_gap_intersections(boxes_in, sorted_lr):
    actual_left = boxes_in[0]['l']
    actual_right = boxes_in[0]['r']
    new_boxes_m = []
    for box in boxes_in:
        if 'num_col' in box:
            if box['num_col'] == 'two' and box['l'] == actual_left:
                tops = []; bots = []
                left_box = box
                for box_in in boxes_in:
                    if box_in['r'] > left_box['r'] and box_in['r'] == actual_right and \
                    box_in['t']==left_box['t'] and box_in['b']==left_box['b']:
                        right_box = box_in
                        gutter = {'l': left_box['r'], 'r': right_box['l'],
                                  'w': right_box['l'] - left_box['l']}
                        for line in sorted_lr:
                            for word in line:
                                gutter_word = False
                                if word['t'] >= left_box['t'] and word['b'] <= left_box['b']:
                                    # Check if whole word is in gutter:
                                    if gutter['l'] < word['l'] and gutter['r'] > word['r']:
                                        gutter_word = True
                                    # Word starts after gutter but ends in box:
                                    elif gutter['l'] <= word['l'] < gutter['r'] and word['r'] > gutter['r']:
                                        gutter_word = True
                                    # Word starts in box but ends in gutter
                                    elif word['l'] < gutter['l'] and gutter['l'] < word['r'] < gutter['r']:
                                        gutter_word = True
                                    # word starts before guttre and ends after guttr.
                                    elif word['l'] < gutter['l'] and word['r'] > gutter['r']:
                                        gutter_word = True
                                if gutter_word:
                                    line_top = min(line, key=lambda k: ('t' not in k, k.get('t', None)))['t']
                                    line_bot = max(line, key=lambda k: ('b' not in k, k.get('b', None)))['b']
                                    tops.append(line_top)
                                    bots.append(line_bot)
                if tops:
                    new_boxes = [{'l': actual_left, 'r': actual_right,
                                  't': c_top, 'b': c_bot,
                                  'w': actual_right-actual_left, 'h': c_bot-c_top,
                                  'color': 'blue', 'num_col': 'one'} for (c_top,c_bot) in zip(tops, bots)]
                    new_boxes_m = [new_boxes[0]]
                    for b_num in list(range(1,len(new_boxes))):
                        if new_boxes[b_num]['t'] > new_boxes_m[-1]['t'] and new_boxes[b_num]['t'] < new_boxes[b_num-1]['b']+20:
                            new_boxes_m[-1]['b'] = new_boxes[b_num]['b']
                        else:
                            new_boxes_m.append(new_boxes[b_num])
    
    
    all_boxes_split = []
    to_split = copy.deepcopy(boxes_in)
    if new_boxes_m:
        for nb in new_boxes_m:
            for box in to_split:
                if box['t'] <= nb['t'] < box['b']:
                    box1 = copy.deepcopy(box); box2 = copy.deepcopy(box)
                    box1.update({'b':nb['t']})
                    box2.update({'t':nb['b']})
                    all_boxes_split.extend([box1, box2])
                else:
                    all_boxes_split.append(box)
            to_split = copy.deepcopy(all_boxes_split)
            all_boxes_split = []
    final_splits = to_split+new_boxes_m
    return final_splits


# In[]:
def minor_horizontal_splits(boxes, sorted_lr, all_words_in_page):
    """ Partition Step 3 : Horizontal splits in vertically split boxes"""
#    min_tab_width = 20
    final_zones = []
    zones_out = []
    delta_allowed_2_col = 8
    delta_allowed_1_col = 500
    all_heights = [max([w['b'] - w['t'] for w in line]) for line in sorted_lr]
    height_20_percentile = np.percentile(all_heights, 50)
    
    for box in boxes:
        horizontal_split_b = [box['t']]
        prev_line_l = box['l']
        words_b = sort_linewise(select_words(box, all_words_in_page))
        for line_num, line_l in enumerate(words_b):
            ## Case 1: Check for entirey capital string.
            line_str = ' '.join([w['word'] for w in words_b[line_num]])
            if line_str.isupper() and len(words_b) > 1:
                line_top = min(words_b[line_num], key=lambda k: ('t' not in k, k.get('t', None)))['t']
                horizontal_split_b.append(line_top)
            ## Case 2: Check for heights greater than some 80 percentile.
            line_height = max([w['b'] - w['t'] for w in line_l])
            if line_height > height_20_percentile:
                line_top = min(words_b[line_num], key=lambda k: ('t' not in k, k.get('t', None)))['t']
                horizontal_split_b.append(line_top)

            ## Case 3: Check indentation for two column images.
#            line_lefts = [line[0]['l'] for line in words_b]
#            for ln, l_l in enumerate(line_lefts):
#                if l_l >= prev_line_l+delta_allowed_2_col:
#                    if box['num_col'] == 'two':
#                        line_top = min(words_b[ln], key=lambda k: ('t' not in k, k.get('t', None)))['t']
#                        horizontal_split_b.append(line_top)
#                prev_line_l = l_l
        horizontal_split_b.append(box['b'])
        selected_zones = []
        for num in range(len(horizontal_split_b)-1):
            inner_zone = {'t': horizontal_split_b[num], 'b': horizontal_split_b[num+1],
                         'l': box['l'], 'r': box['r'],
                         'w': box['w'], 'h': horizontal_split_b[num+1]-horizontal_split_b[num],
                         'color':box['color'], 'num_col': box['num_col']}
            selected_zones.append(inner_zone)
        final_zones.extend(selected_zones)

    for n, zone in enumerate(final_zones):
        box_out = []
        zone_words = select_words(zone, all_words_in_page)
        if zone in box_out:
            continue
        if zone_words:
            box_out.append(zone)
        zones_out.extend(box_out)
    return zones_out

# In[]:
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


# In[]:
def merge_consecutive_tables(boxes, sorted_lr, all_words_in_page):
    for box in boxes:
        box_words = sort_linewise(select_words(box, all_words_in_page))
        if box_words:
            left_lims, right_lims, page_top, page_bottom = get_text_limits(box_words)
            box['l'] = min(left_lims)
            box['r'] = max(right_lims)
    
    boxes_td = sorted(boxes, key=lambda k: ("l" not in k, k.get('l', None)))
    boxes = sorted(boxes_td, key=lambda k: ("t" not in k, k.get('t', None)))
    
    merged = [boxes[0]]
    prev_box_words = sort_linewise(select_words(boxes[0], all_words_in_page))
    prev_box_width = boxes[0]['r'] - boxes[0]['l']
    prev_table_check = check_table(prev_box_words, prev_box_width)
    for box_num in range(1, len(boxes)):
        curr_box_words = sort_linewise(select_words(boxes[box_num], all_words_in_page))
        box_width = boxes[box_num]['r'] - boxes[box_num]['l']
        curr_table_check = check_table(curr_box_words, box_width)
        if prev_table_check and curr_table_check and boxes[box_num]['num_col']=='one':
            merged[-1]['b'] = boxes[box_num]['b']
            merged[-1]['l'] = min(merged[-1]['l'], boxes[box_num]['l'])
            merged[-1]['r'] = max(merged[-1]['r'], boxes[box_num]['r'])
        else:
            merged.append(boxes[box_num])
        prev_table_check = curr_table_check
    return merged

# In[]:
def template(working_page,page_num):
    xl = pd.ExcelFile('list_sections20180404.xlsx')
    info = xl.parse('Sheet1')
#    info = pd.read_excel(open('list_sections20180404.xlsx','rb'), sheet_name=0)
    p_info = working_page['microfiche']
    try:
        num_cols_expected = int(info[info['Columns'].notnull() & \
                     (info['Microfiche1'] <= int(p_info['microficheName'])) & \
                     (int(p_info['microficheName']) <= info['Microfiche2']) & \
                     (info['Year'] == int(p_info['year']))][-1:]['Columns'])
    except:
        num_cols_expected = 1
    all_words_in_page = get_all_words(working_page)
    sorted_lr = sort_two_d(working_page)
    if sorted_lr:
        divided_page = get_major_blocks(sorted_lr)
        if num_cols_expected == 2:
            major_vertical_splits = major_vertical_split_2_col(divided_page)
        else:
            major_vertical_splits = get_major_vertical_splits(divided_page, page_num, working_page)
        major_splits_again = find_gap_intersections(major_vertical_splits, sorted_lr)
    
    #    minor_splits = minor_horizontal_splits(major_vertical_splits, sorted_lr, all_words_in_page)
    #    merged_boxes = merge_consecutive_tables(minor_splits, sorted_lr, all_words_in_page)
        print("Page {0} done".format(page_num))
        return major_splits_again
    else:
        print("Page {0} empty".format(page_num))
        return []


def xml_out_string(boxes):
    header = '<?xml version="1.0" encoding="UTF-16"?>\n'
    zone_num = '<Zones znum="{0}">\n'.format(len(boxes))
    string_out = header+zone_num
    for box in boxes:
        bounding_box_string = "<BBox left=\"{0}\" top=\"{1}\" right=\"{2}\" bottom=\"{3}\"/>\n".format(\
                                            box['l'],box['t'], box['r'], box['b'])
        next_string_1 = "<Props type=\"flow\" clr=\"0xffffff\" chk_ctrl=\"0xb0000\" fm=\"1\" rm=\"13\" "
        next_string_2 = "filt=\"0x1f\" langs=\"0,-1\"/>\n"
        zone_string = "<Zone\n>"+bounding_box_string+next_string_1+next_string_2+"</Zone>\n"
        string_out += zone_string
    return string_out
        

# In[]: 
if __name__ == '__main__':
    import glob
    import os
    
    directory = 'Data'
    parsed_word_file = glob.glob(os.path.join(directory,'*1928*.json'))[0]
#    with open(parsed_word_file) as pf:
#        all_pages = json.load(pf)
#    selected_pages = [33,35]
#    for page_num in selected_pages:
#        boxes = template(all_pages[page_num])
#        outstring = xml_out_string(boxes)
