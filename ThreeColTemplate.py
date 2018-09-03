#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:04:32 2018

@author: prajnya
"""
import numpy as np
import copy
from pprint import pprint as pp

from ZoningCommons import get_text_bounds, mean_gap_between
from ZoningUtils import get_lines_in_box, get_words_in_box
from ZoningUtils import cosine_similarity

def redefined_line_bounds(lines_in_box):
    line_bounds = []
    if lines_in_box:
        for each_line in lines_in_box:
            line_info = {}
            line_info['l'] = min(each_line, key=lambda k: ("l" not in k, k.get('l', None)))['l']
            line_info['r'] = max(each_line, key=lambda k: ("r" not in k, k.get('r', None)))['r']
            line_info['t'] = min(each_line, key=lambda k: ("t" not in k, k.get('t', None)))['t']
            line_info['b'] = max(each_line, key=lambda k: ("b" not in k, k.get('b', None)))['b']
            line_bounds.append(line_info)
        return line_bounds
    else:
        return []
    
    
def check_all_num(words_in_line, num_percent=0.8):
    num_digits = 0
    total_chars = 0
    for word in words_in_line:
        for char in word['word']:
            if char.isdigit():
                num_digits+=1
        total_chars+=len(word['word'])
    if total_chars>0:
        if (num_digits/total_chars) > num_percent:
            return True
    return False

def check_table(words_in_block, block_width, num_lines_in_block):
    """ Check if a block is a block of tables or of text."""
#    average_words_per_line=24
#    total_num_words = 0
    ratio_threshold = 0.55
    actual_num_chars = 0
    all_char_ws = []
    cas = []
#        total_num_words += len(line)
    if num_lines_in_block > 0:
        for word in words_in_block:
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
        else: return False
    else: return False
    

def find_vertical_line(third_val, words_in_box):
    third_variance = 100
    vertical_splits = [third_val+el for el in list(range((-third_variance), (third_variance)))]
    intersect_count_list = []
    for split_line in vertical_splits:
        intersect_count = 0
        for line in words_in_box:
            if line['l'] < split_line < line['r']:
                intersect_count+=1
        intersect_count_list.append(intersect_count)
    masked_list = [el[1] for el in zip(intersect_count_list, vertical_splits) if 0<= el[0]<12]
    if masked_list:
        vertical_split_line = np.median(masked_list)
        return vertical_split_line
    else:
        masked_list = [el[1] for el in zip(intersect_count_list, vertical_splits) if 0<= el[0]<16]
        if masked_list:
            vertical_split_line = np.median(masked_list)
            return vertical_split_line
    print("I take issue here")
    return None
    


class three_col_page:
    """ Class that holds templates for three column pages"""
    def __init__(self, page):
        self.page = page
        self.lines_in_page = page['line_info']
        self.words_in_page = page['word_info']
        self.all_line_heights = [w['b'] - w['t'] for w in self.lines_in_page]
        self.left_indents = []

    def get_major_blocks(self, gap_threshold=80):
        """ ## First make the major horizontal splits so that where to split 
        vertically can be approximated.
        """
        hor_dict = {}
        details_out = []
        
        page_left, page_right, page_top, page_bottom = get_text_bounds(self.lines_in_page)
        for num in range(page_top, page_bottom+1):
            hor_dict[num] = 0
        for line in self.lines_in_page:
            for l_filled in range(line['t'], line['b']+1):
                hor_dict[l_filled] += 1
        mean_gap, empty_lines_list = mean_gap_between(hor_dict, gap_threshold)
        top_bounds = [page_top]
        bottom_bounds = []
        for lines_set in empty_lines_list:
            if len(lines_set) >= mean_gap:
                bottom_bounds.append(lines_set[0])
                top_bounds.append(lines_set[-1])
        bottom_bounds.append(page_bottom)
        for ind in range(len(top_bounds)):
            box_limits_dict = {'t': top_bounds[ind], 'b': bottom_bounds[ind],
                               'l': page_left, 'r': page_right,
                               'w': page_right - page_left, 
                               'h': bottom_bounds[ind] - top_bounds[ind],
                               'color': 'seagreen', 'num_col':'one'}
            details_out.append(box_limits_dict)
        return details_out
    
    def close_gaps(self, boxes_in):
        """Close white space gaps between boxes if any.
        Makes sure there s only a two pixel gap"""
        final_boxes = []
        for bbox in boxes_in:
            if bbox['b'] > bbox['t']:
                final_boxes.append(bbox)
        sorted_ud = copy.deepcopy(sorted(final_boxes, key=lambda k: ("t" not in k, k.get('t', None))))
        for bnum, box in enumerate(sorted_ud[1:], 1):
            if box['t']>sorted_ud[bnum-1]['b']+2:
                sorted_ud.append({'t':sorted_ud[bnum-1]['b']+2, 'b':box['t']-2,
                                  'l':box['l'],'r':box['r'],
                                  'h':(box['t']-2)-(sorted_ud[bnum-1]['b']+2),'w':box['w'],
                                  'color':'red','num_cols':'one'})
        final_sorted = sorted(sorted_ud, key=lambda k: ("t" not in k, k.get('t', None)))
        return final_sorted
    
    def find_moodys_box(self, box, num_lines):
        moodys_box = []
        moodys_string = 'MOODY\'S MANUAL OF INVESTMENTS'
#        if num_lines<=1:
        words_in_box = ' '.join([w['word'] for w in get_words_in_box(box, self.words_in_page)])
        if '&quot;' in words_in_box:
            words_in_box = words_in_box.replace("&quot;", "'")
        word_similarity = cosine_similarity(moodys_string, words_in_box)
        if word_similarity > 0.8:
            moodys_box = box
        else:
            char_similarity = cosine_similarity(moodys_string, words_in_box, 'char')
            if char_similarity >= 0.7:
                moodys_box = box
        return moodys_box
    
    def remove_moodys_string(self, boxes_in):
        moodys_box = []
        final_boxes = []
        for box in boxes_in:
            top_bounds = []
            bottom_bounds = []
            expected_num_lines = int(box['h']/np.mean(self.all_line_heights))
            lines_in_block = redefined_line_bounds(self.get_split_lines(box))
            if not moodys_box:
                moodys_box = self.find_moodys_box(box, expected_num_lines)
                if moodys_box:
                    final_boxes.append(box)
                    continue
            if lines_in_block and not moodys_box:
                sorted_lines = sorted(lines_in_block, key=lambda k: ("t" not in k, k.get('t', None)))
                first_line = sorted_lines[0]
                l_indent = first_line['l']-box['l']
                r_indent = box['r'] - first_line['r']
                large_lr_indent = (l_indent > 1000) or (r_indent > 1000)
                if large_lr_indent:
                    moodys_box = self.find_moodys_box(first_line, 1)
                    if moodys_box:
                        box.update({'t':first_line['b'], 'h':box['b']-first_line['b']})
                        final_boxes.append(box)
                        continue
            if not moodys_box:
                hor_dict = {}        
                for num in range(box['t'], box['b']+1):
                    hor_dict[num] = 0
                for line in lines_in_block:
                    for l_filled in range(line['t'], line['b']+1):
                        hor_dict[l_filled] += 1
                mean_gap, empty_lines_list = mean_gap_between(hor_dict, 75)
                for lines_set in empty_lines_list:
                    if len(lines_set) >= mean_gap:
                        bottom_bounds.append(lines_set[0])
                        top_bounds.append(lines_set[-1])
                if bottom_bounds:
                    for bot in bottom_bounds:
                        split_box = copy.deepcopy(box)
                        split_box.update({'b':bot, 'h':bot-split_box['t']})
                        box.update({'t':bot, 'h':box['b']-bot})
                        split_line_exect = int(split_box['h']/np.mean(self.all_line_heights))
                        moodys_box = self.find_moodys_box(split_box, split_line_exect)
                        if moodys_box:
                            final_boxes.append(split_box)
                            final_boxes.append(box)
                            break # Breaks only from inner loop
                        else:
                            final_boxes.append(split_box)
                    final_boxes.append(box)
                else:final_boxes.append(box)
            else:
                final_boxes.append(box)
        if not moodys_box:
            print("no moodys string\n********\n")
        # Only keep boxes below moody string
        if moodys_box:
            zones_out = []
            for sel_box in final_boxes:
                if sel_box['t'] > moodys_box['t']:
                    zones_out.append(sel_box)
        else:
            zones_out = final_boxes
        return zones_out
    
    
    def partition_3_col(self, boxes_in):
        """ When the page is expected to have three columns come here directly."""
        
        ### Rejoin all the blocks first so that there is only one left. Makes life easier
        sorted_td = copy.deepcopy(sorted(boxes_in, key=lambda k: ('t' not in k, k.get('t', None))))
        single_box = copy.deepcopy(sorted_td[0])
        single_box.update({'b':sorted_td[-1]['b'], 'h':sorted_td[-1]['b']-sorted_td[0]['t'],
                           'color':'seagreen'})
        one_third = (single_box['r']-single_box['l'])/3
        page_left = single_box['l']
        page_right = single_box['r']
        words_in_single_box = get_words_in_box(single_box, self.words_in_page)
        text_one_third = page_left + one_third
        text_two_third = page_right- one_third
        
        vertical_split_1= find_vertical_line(text_one_third, words_in_single_box)
        vertical_split_2 = find_vertical_line(text_two_third, words_in_single_box)
        final_boxes = []
        if vertical_split_1 and vertical_split_2:
            box_1 = {'t': single_box['t'], 'b': single_box['b'],
                     'l': page_left, 'r': vertical_split_1,
                     'w': vertical_split_1-page_left, 'h': single_box['b']-single_box['t'],
                     'color':'mediumvioletred', 'num_col': 'three'}
            box_2 = {'t': single_box['t'], 'b': single_box['b'],
                     'l': vertical_split_1, 'r': vertical_split_2,
                     'w': vertical_split_2-vertical_split_1, 'h': single_box['b']-single_box['t'],
                     'color':'mediumvioletred', 'num_col': 'three'}
            box_3 = {'t': single_box['t'], 'b': single_box['b'],
                     'l': vertical_split_2, 'r': page_right,
                     'w': page_right-vertical_split_2, 'h': single_box['b']-single_box['t'],
                     'color':'mediumvioletred', 'num_col': 'three'}
            final_boxes.extend([box_1, box_2, box_3])
        else:
            return [{'l': page_left, 't': single_box['t'], 'r': page_right, 'b': single_box['b'],
                                    'w': page_right-page_left, 'h': single_box['b']-single_box['t'],
                                    'color':'red', 'num_col': 'one'}]
        return final_boxes

    def create_new_boxes(self, top_bounds, bot_bounds, box):
        """ Given a set of top and bounds, split the outer box into a set of smaller ones"""
        final_splits = []
        new_boxes_m = []
        if top_bounds:
            new_boxes = [{'t': c_top, 'b': c_bot,
                          'l': box['l'], 'r': box['r'],
                          'w': box['r']-box['l'], 'h': c_bot-c_top,
                          'color': 'orange', 'num_col': 'three'} \
                for (c_top,c_bot) in zip(top_bounds, bot_bounds)]
            
            ## Makes sure that there is no smaller box in a larger box. If there is, merge them.
            new_boxes = sorted(new_boxes, key=lambda k: ("t" not in k, k.get('t', None)))
            new_boxes_m = [new_boxes[0]]
            for b_num in list(range(1,len(new_boxes))):
                if new_boxes[b_num]['t'] > new_boxes_m[-1]['t'] and \
                new_boxes[b_num]['t'] < new_boxes[b_num-1]['b']+20:
                    new_boxes_m[-1]['b'] = new_boxes[b_num]['b']
                else:
                    new_boxes_m.append(new_boxes[b_num])
                    
        if new_boxes_m:
            to_split = [copy.deepcopy(box)]
            for num, nbm in enumerate(new_boxes_m):
                all_boxes_split = []
                for outer_box in to_split:
                    if nbm['t'] == outer_box['t'] and nbm['b'] < outer_box['b']:
                        box1 = copy.deepcopy(outer_box); box2 = copy.deepcopy(outer_box)
                        box1.update({'b':nbm['b']})
                        box2.update({'t':nbm['b']})
                        all_boxes_split.extend([box1, box2])
                    elif outer_box['t'] < nbm['t'] < outer_box['b'] and outer_box['t'] < nbm['b'] < outer_box['b']:
                        box1 = copy.deepcopy(outer_box); box3 = copy.deepcopy(outer_box)
                        box1.update({'b':nbm['t']})
                        box3.update({'t':nbm['b']})
                        all_boxes_split.extend([box1, nbm, box3])
                    elif outer_box['t'] < nbm['t'] < outer_box['b'] and nbm['b'] == outer_box['b']:
                        box1 = copy.deepcopy(outer_box); box2 = copy.deepcopy(outer_box)
                        box1.update({'b':nbm['t']})
                        box2.update({'t':nbm['t']})
                        all_boxes_split.extend([box1, box2])
                    else:
                        all_boxes_split.append(outer_box)
                to_split = copy.deepcopy(all_boxes_split)
    
            final_splits = [bo for bo in all_boxes_split if bo['t'] != bo['b']]
        return final_splits
    
    def title_splits_3_col(self, boxes_in):
        """ Split based on titles in the box"""
        zones_out = []
        for box in boxes_in:
            if box['num_col']=='one':
                continue
            tops = []; bots = []
#            box_center = box['l']+(box_width/2)
            lines_in_box = redefined_line_bounds(self.get_split_lines(box))
            for line_num, line_l in enumerate(lines_in_box):
                words_in_line = get_words_in_box(line_l, self.words_in_page)
                line_str = ' '.join(w['word'] for w in words_in_line)
                ## Case 1: Check for entirely capital string.
                if line_str.isupper() and len(words_in_line) > 1:
                    if line_l['t'] not in tops:
                        tops.append(line_l['t'])
                        bots.append(line_l['b'])            
            new_boxes_m = self.create_new_boxes(tops, bots, box)
            if new_boxes_m:
                zones_out.extend(new_boxes_m)
            else:
                zones_out.append(box)
        zones_out = [box for box in zones_out if get_words_in_box(box, self.words_in_page)]
        return zones_out
    
    def get_tab_width(self):
        if self.left_indents:
            mean = np.mean(self.left_indents, axis=0)
            sd = np.std(self.left_indents, axis=0)
            without_outliers = [x for x in self.left_indents if (x > mean - (2*sd))]
            without_outliers = [x for x in without_outliers if (x < mean + (2*sd))]
            two_col_tab = np.mean(without_outliers)
        else:
            two_col_tab = 320
        return two_col_tab
    
    def get_split_lines(self, box):
        # Split words in box into lines.
        words_in_box = get_words_in_box(box, self.words_in_page)
        sorted_td = sorted(words_in_box, key = lambda k: ("t" not in k, k.get('t', None)))
        prev_line_bottom = 0
        line_split = []
        new_line = []
        for word in sorted_td:
            space_from_prev_line = word['t'] - prev_line_bottom
            if space_from_prev_line > 12:
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

    

    def right_gap_splits(self, box):
        """ Perform horizontal splits based on space at the end of a line when the 
        next line is a new paragraph."""
        max_word_width = np.percentile([w['r']-w['l'] for w in self.words_in_page], 60)
        new_boxes = []
#        words_in_block = get_words_in_box(box, self.words_in_page)
        lines_in_block = redefined_line_bounds(self.get_split_lines(box))
#        is_table = check_table(words_in_block, box['r']-box['l'], len(lines_in_block))
        if box['num_col']=='three':
            to_split = copy.deepcopy(box)
            sorted_ud = sorted(lines_in_block, key=lambda k: ("t" not in k, k.get('t', None)))
            for lnum, line in enumerate(sorted_ud[1:-1], 1):
                right_space = box['r']-line['r']
                if right_space > max_word_width:
                    new_b = {'t':to_split['t'], 'b': line['b'],
                                      'l':to_split['l'], 'r': to_split['r'],
                                      'w':to_split['r']-to_split['l'], 'h': line['b']-to_split['t'],
                                      'color':'orange', 'num_col': 'three'}
                    newb_lines = redefined_line_bounds(self.get_split_lines(new_b))
                    selected_lines = newb_lines[1:]
                    newb_words = []
                    for s_line in selected_lines:
                        newb_words.extend(get_words_in_box(s_line, self.words_in_page))
                    all_num = check_all_num(newb_words, num_percent=0.5)
                    if newb_words:
                        if not all_num:
                            new_boxes.append(new_b)
                            to_split.update({'t':line['b']})
                            self.left_indents.append(lines_in_block[lnum+1]['l']-box['l'])
                    else:
                        new_boxes.append(new_b)
                        to_split.update({'t':line['b']})
                        self.left_indents.append(lines_in_block[lnum+1]['l']-box['l'])
            new_boxes.append(to_split)
        else:
            new_boxes.append(box)
        return new_boxes
    
    def left_indent_split(self, boxes_in):
        """ Split the box if there is a left indent at the start of the box."""
        final_boxes = []
        for in_box in boxes_in:
            three_col_tab = self.get_tab_width()
            new_boxes=[]
            to_split = copy.deepcopy(in_box)
            lines_in_block = redefined_line_bounds(self.get_split_lines(in_box))
            is_table = check_table(get_words_in_box(in_box, self.words_in_page), 
                                   in_box['r']-in_box['l'],
                                   len(get_lines_in_box(in_box, lines_in_block)))
            if not is_table and in_box['num_col']=='three':
                lines_in_block = get_lines_in_box(in_box, self.lines_in_page)
                sorted_ud = sorted(lines_in_block, key=lambda k: ("t" not in k, k.get('t', None)))
                for lnum, line in enumerate(sorted_ud[1:-1], 1):
                    is_indent = 0.65*three_col_tab<abs(line['l']-in_box['l'])<3*three_col_tab
                    next_line_indent = lines_in_block[lnum+1]['l']<line['l']
                    if is_indent and next_line_indent:
                        self.left_indents.append(line['l']-in_box['l'])
                        new_boxes.append({'t':to_split['t'], 'b':line['t'],
                                   'l':to_split['l'], 'r': to_split['r'],
                                   'w':to_split['w'], 'h': to_split['h'],
                                   'color':'orange', 'num_col':to_split['num_col']})
                        to_split.update({'t':line['t']})
                new_boxes.append(to_split)
            else:
                new_boxes.append(in_box)
            final_boxes.extend(new_boxes)
        return final_boxes
    
    def merge_consecutive_tables(self,boxes_in):
        boxes_out = []
        if boxes_in:
            prev_box = copy.deepcopy(boxes_in[0])
            lines_in_box = redefined_line_bounds(self.get_split_lines(prev_box))
            prev_is_table = check_table(get_words_in_box(prev_box, self.words_in_page), prev_box['w'], len(lines_in_box))
            for box in boxes_in[1:]:
                curr_words = get_words_in_box(box, self.words_in_page)
                c_lines_in_box = redefined_line_bounds(self.get_split_lines(box))
                curr_is_table = check_table(curr_words, box['w'], len(c_lines_in_box))
                if prev_is_table and curr_is_table:
                    prev_box['b'] = box['b']
                else:
                    boxes_out.append(prev_box)
                    prev_box = copy.deepcopy(box)
                prev_is_table = copy.deepcopy(curr_is_table)
            boxes_out.append(prev_box)
            return boxes_out
        else:
            return []
                
            
    
    def minor_horizontal_splits(self, boxes_in):
        """ Split one column horizontal boxes into minor splits."""
        boxes_out = []
        for box in boxes_in:
            right_indent_splits = self.right_gap_splits(box)
            title_splits = self.title_splits_3_col(right_indent_splits)
            left_splits = self.left_indent_split(title_splits)
            merged_boxes = self.merge_consecutive_tables(left_splits)
            boxes_out.extend(merged_boxes)
        return boxes_out
    
    def readjust_boxes(self, boxes_in):
        boxes_out = []
        for box in boxes_in:
            box.update({'t':box['t']-2, 'b': box['b']+2, 'h': box['h']+4,
                        'l':box['l']-2, 'r': box['r']+2, 'w': box['w']+4})
            boxes_out.append(box)
        return boxes_out
                        
