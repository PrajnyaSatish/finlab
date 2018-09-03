#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:29:21 2018

@author: prajnya
"""
import numpy as np
import copy

from ZoningCommons import get_text_bounds
from ZoningUtils import get_lines_in_box, get_words_in_box

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
    ratio_threshold = 0.55 # Threshold for the ratio of number of words to expected number of words
    actual_num_chars = 0
    all_char_ws = []
    cas = []
    all_num = check_all_num(words_in_block, num_percent=0.8)
    if num_lines_in_block > 0:
        for word in words_in_block:
            if word['word']:
                actual_num_chars += len(word['word'])
                char_w = float(word['r']-word['l'])/len(word['word'])
                cas.append(round(char_w, 2))
        all_char_ws.extend(cas)
        average_char_width = np.mean(all_char_ws)
        expected_num_chars = (float(block_width)/average_char_width)*num_lines_in_block
        ratio = actual_num_chars/expected_num_chars
        if ratio < ratio_threshold and all_num:
            return True
        else: return False
    else: return False

class two_col_page():
    
    def __init__(self, page):
        self.page = page
        self.lines_in_page = page['line_info']
        self.words_in_page = page['word_info']
        self.left_indents_2_col = []
        
    def partition_2_col(self):
        final_boxes = []
        half_mark_variance = 200
        page_left, page_right, page_top, page_bottom = get_text_bounds(self.lines_in_page)
        text_half = page_left + (float(page_right - page_left)/2)
        vertical_splits = [text_half+el for el in list(range((-half_mark_variance), (half_mark_variance)))]
        intersect_count_list = []
        for split_line in vertical_splits:
            intersect_count = 0
            for line in self.lines_in_page:
                if line['l'] < split_line < line['r']:
                    intersect_count+=1
            intersect_count_list.append(intersect_count)
        masked_list = [el[1] for el in zip(intersect_count_list, vertical_splits) if 0<= el[0]<12]
        if masked_list:
            vertical_split_line = np.median(masked_list)
            box_1 = {'t': page_top, 'b': page_bottom,
                     'l': page_left, 'r': vertical_split_line,
                     'w': vertical_split_line-page_left, 'h': page_bottom-page_top,
                     'color':'orange', 'num_col': 'two'}
            box_2 = {'t': page_top, 'b': page_bottom,
                     'l': vertical_split_line, 'r': page_right,
                     'w': page_right-vertical_split_line, 'h': page_bottom-page_top,
                     'color':'orange', 'num_col': 'two'}
            final_boxes.extend([box_1, box_2])
        else:
            return [{'l': page_left, 't': page_top, 'r': page_right, 'b': page_bottom,
                            'w': page_right-page_left, 'h': page_bottom-page_top,
                            'color':'red', 'num_col': 'one'}]
        return final_boxes
    
    def unsplit_boxes(self, top_bounds, bot_bounds, box, actual_left, actual_right):
        """ Function converts adjacent two column boxes to a single column box"""
        
        new_boxes_m = []
        if top_bounds:
            new_boxes = [{'t': c_top, 'b': c_bot,
                          'l': actual_left, 'r': actual_right,
                          'w': actual_right-actual_left, 'h': c_bot-c_top,
                          'color': 'seagreen', 'num_col': 'one'} \
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
        return new_boxes_m
    
    def find_gap_intersections(self, boxes_in):
        actual_left = min(boxes_in, key=lambda x:x['l'])['l']
        actual_right = max(boxes_in, key=lambda x:x['r'])['r']
        final_splits = []
        paired_boxes = []
        for box in boxes_in:
            if 'num_col' in box:
                if box['num_col'] == 'two' and box['l'] == actual_left:
                    box_on_left = box
                    for box_in in boxes_in:
                        if box_in['r'] > box_on_left['r'] and box_in['r'] == actual_right and \
                        box_in['t']==box_on_left['t'] and box_in['b']==box_on_left['b'] and box['num_col']=='two':
                            box_on_right = box_in
                    paired_boxes.append({'left_box':box_on_left, 'right_box': box_on_right})
                elif box['num_col']=='one':
                    final_splits.append(box)
                        
        if paired_boxes:
            for pair in paired_boxes:
                tops = []; bots = []
                left_box = pair['left_box']
                right_box = pair['right_box']
                gutter = {'l': left_box['r'], 'r': right_box['l'], 
                          'w': right_box['l'] - left_box['l']}
                for line in self.lines_in_page:
                    gutter_line = False
                    if line['t'] >= left_box['t'] and line['b'] <= left_box['b']:
                        # Check if whole word is in gutter:
                        if gutter['l'] < line['l'] and gutter['r'] > line['r']:
                            gutter_line = True
                        # Word starts after gutter but ends in box:
                        elif gutter['l'] <= line['l'] < gutter['r'] and line['r'] > gutter['r']:
                            gutter_line = True
                        # Word starts in box but ends in gutter
                        elif line['l'] < gutter['l'] and gutter['l'] < line['r'] < gutter['r']:
                            gutter_line = True
                        # word starts before guttre and ends after guttr.
                        elif line['l'] < gutter['l'] and line['r'] > gutter['r']:
                            gutter_line = True
                    if gutter_line:
                        line_top = line['t']
                        line_bot = line['b']
                        tops.append(line_top)
                        bots.append(line_bot)
                non_gap_boxes = self.unsplit_boxes(tops,bots, box, actual_left, actual_right)
                if non_gap_boxes:
                    to_split_l = copy.deepcopy([left_box])
                    to_split_r = copy.deepcopy([right_box])
                    for num, nbm in enumerate(non_gap_boxes):
                        boxes_l_split = []
                        boxes_r_split = []
                        for bnum, outer_box in enumerate(to_split_l):
                            if nbm['t'] == outer_box['t'] and nbm['b'] < outer_box['b']:
                                box2_l = copy.deepcopy(outer_box); box2_r = copy.deepcopy(to_split_r[bnum])
                                box2_l.update({'t':nbm['b']}); box2_r.update({'t':nbm['b']})
                                boxes_l_split.extend([box2_l])
                                boxes_r_split.extend([box2_r])
                            elif outer_box['t'] < nbm['t'] < outer_box['b'] and outer_box['t'] < nbm['b'] < outer_box['b']:
                                box1_l = copy.deepcopy(outer_box); box3_l = copy.deepcopy(outer_box)
                                box1_l.update({'b':nbm['t']})
                                box3_l.update({'t':nbm['b']})
                                box1_r = copy.deepcopy(to_split_r[bnum]); box3_r = copy.deepcopy(to_split_r[bnum])
                                box1_r.update({'b':nbm['t']})
                                box3_r.update({'t':nbm['b']})
                                boxes_l_split.extend([box1_l, box3_l])
                                boxes_r_split.extend([box1_r, box3_r])
                            elif outer_box['t'] < nbm['t'] < outer_box['b'] and nbm['b'] == outer_box['b']:
                                box1_l = copy.deepcopy(outer_box); box1_r = copy.deepcopy(to_split_r[bnum])
                                box1_l.update({'b':nbm['t']}); box1_r.update({'b':nbm['t']})
                                boxes_l_split.extend([box1_l])
                                boxes_r_split.extend([box1_r])
                            else:
                                boxes_l_split.extend([outer_box])
                                boxes_r_split.extend([to_split_r[bnum]])
                        to_split_l = copy.deepcopy(boxes_l_split)
                        to_split_r = copy.deepcopy(boxes_r_split)
                    final_splits += to_split_l+to_split_r+non_gap_boxes
                else: 
                    final_splits.extend([left_box, right_box])
    
        final_splits = [bo for bo in final_splits if bo['t'] != bo['b']]
        boxes_out = []
        for fs in final_splits:
            if fs not in boxes_out:
                boxes_out.append(fs)
        final_splits_out = [box for box in boxes_out if get_words_in_box(box, self.words_in_page)]
        return final_splits_out

    def create_new_boxes(self, top_bounds, bot_bounds, box):
        """ Given a set of top and bounds, split the outer box into a set of smaller ones"""
        final_splits = []
        new_boxes_m = []
        if top_bounds:
            new_boxes = [{'t': c_top, 'b': c_bot,
                          'l': box['l'], 'r': box['r'],
                          'w': box['r']-box['l'], 'h': c_bot-c_top,
                          'color': 'orange', 'num_col': 'two'} \
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
    
    def title_splits_2_col(self, boxes_in):
        """ Split based on titles in the box"""
        zones_out = []
        for box in boxes_in:
            if box['num_col']=='one':
                continue
            tops = []; bots = []
            box_width = box['r'] - box['l']
            box_height = box['b'] - box['h']
            box_center = box['l']+(box_width/2)
            lines_in_box = get_lines_in_box(box, self.lines_in_page)
            for line_num, line_l in enumerate(lines_in_box):
                words_in_line = get_words_in_box(line_l, self.words_in_page)
                line_str = ' '.join(w['word'] for w in words_in_line)
                left_gap = line_l['l']-box['l']; right_gap = box['r']-line_l['r']
                gap_diff = right_gap - left_gap
                line_width = line_l['r'] - line_l['l']
                line_center = line_l['l']+((line_l['r']-line_l['l'])/2)
                center_diff = box_center - line_center
                all_num = check_all_num(words_in_line)
                ## Case 1: Check for entirely capital string.
                if line_str.isupper() and len(words_in_line) > 1 and abs(gap_diff) < 0.05*box_width:
                    if line_l['t'] not in tops:
                        tops.append(line_l['t'])
                        bots.append(line_l['b'])
                ## Case 2: Check if the line is centered
                elif left_gap > 0.05*box_width and right_gap > 0.05*box_width and not all_num and \
                abs(center_diff) < 0.015*box_width and abs(gap_diff) < 0.03*box_width and len(words_in_line)>1:
                    if line_l['t'] not in tops:
                        tops.append(line_l['t'])
                        bots.append(line_l['b'])
                # Case 3: Check for "Moody's" title 
                elif box_height > 200 and line_l['t'] < 1500 and \
                0.65*box_width < line_width < 0.8*box_width:
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
        if self.left_indents_2_col:
            mean = np.mean(self.left_indents_2_col, axis=0)
            sd = np.std(self.left_indents_2_col, axis=0)
            without_outliers = [x for x in self.left_indents_2_col if (x > mean - (2*sd))]
            without_outliers = [x for x in without_outliers if (x < mean + (2*sd))]
            two_col_tab = np.mean(without_outliers)
        else:
            two_col_tab = 320
        return two_col_tab

    def right_gap_splits(self, box):
        """ Perform horizontal splits based on space at the end of a line when the 
        next line is a new paragraph."""
        max_word_width = np.percentile([w['r']-w['l'] for w in self.words_in_page], 98)
        new_boxes = []
        words_in_block = get_words_in_box(box, self.words_in_page)
        lines_in_block = get_lines_in_box(box, self.lines_in_page)
        is_table = check_table(words_in_block, box['r']-box['l'], len(lines_in_block))
        if box['num_col']=='two' and not is_table:
            to_split = copy.deepcopy(box)
            sorted_ud = sorted(lines_in_block, key=lambda k: ("t" not in k, k.get('t', None)))
            for lnum, line in enumerate(sorted_ud[1:-1], 1):
                right_space = box['r']-line['r']
                if right_space > max_word_width:
                    new_b = {'t':to_split['t'], 'b': line['b'],
                                      'l':to_split['l'], 'r': to_split['r'],
                                      'w':to_split['r']-to_split['l'], 'h': line['b']-to_split['t'],
                                      'color':'orange', 'num_col': 'two'}
                    newb_lines = get_lines_in_box(new_b, self.lines_in_page)
                    selected_lines = newb_lines[1:]
                    newb_words = []
                    for s_line in selected_lines:
                        newb_words.extend(get_words_in_box(s_line, self.words_in_page))
                    all_num = check_all_num(newb_words, num_percent=0.5)
                    if newb_words:
                        if not all_num:
                            new_boxes.append(new_b)
                            to_split.update({'t':line['b']})
                            self.left_indents_2_col.append(lines_in_block[lnum+1]['l']-box['l'])
            new_boxes.append(to_split)
        else:
            new_boxes.append(box)
        return new_boxes
    
    def left_indent_split(self, boxes_in):
        """ Split the box if there is a left indent at the start of the box."""
        final_boxes = []
        for in_box in boxes_in:
            two_col_tab = self.get_tab_width()
            new_boxes=[]
            to_split = copy.deepcopy(in_box)
            lines_in_block = get_lines_in_box(in_box, self.lines_in_page)
            is_table = check_table(get_words_in_box(in_box, self.words_in_page), 
                                   in_box['r']-in_box['l'],
                                   len(get_lines_in_box(in_box, lines_in_block)))
            if not is_table and in_box['num_col']=='two':
                lines_in_block = get_lines_in_box(in_box, self.lines_in_page)
                sorted_ud = sorted(lines_in_block, key=lambda k: ("t" not in k, k.get('t', None)))
                for lnum, line in enumerate(sorted_ud[1:-1], 1):
                    is_indent = 0.65*two_col_tab<line['l']-in_box['l']<1.25*two_col_tab
                    next_line_indent = lines_in_block[lnum+1]['l']<line['l']
                    if is_indent and next_line_indent:
                        self.left_indents_2_col.append(line['l']-in_box['l'])
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

    
    def minor_horizontal_splits(self, boxes_in):
        """ Split one column horizontal boxes into minor splits."""
        boxes_out = []
        for box in boxes_in:            
            right_indent_splits = self.right_gap_splits(box)
            left_splits = self.left_indent_split(right_indent_splits)
            boxes_out.extend(left_splits)
        return boxes_out
    