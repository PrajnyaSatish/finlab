#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 14:46:41 2018

@author: prajnya
"""
import numpy as np
import copy
import string

from ZoningUtils import get_lines_in_box, get_words_in_box
from ZoningUtils import cosine_similarity

from ZoningCommons import get_text_bounds, mean_gap_between
from TwoColTemplate import two_col_page
from ThreeColTemplate import redefined_line_bounds, three_col_page


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
    ratio_threshold = 0.50
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
    
def formatlinelist(numberlist):
    """Sorts the list of line numbers that do not intersect at block half and 
    returns consecutive lines"""
    prev_number = min(numberlist) if numberlist else None
    pagelist = list()

    for number in sorted(numberlist):
        if number != prev_number+1:
            pagelist.append([number])
        elif len(pagelist[-1]) > 1:
            pagelist[-1][-1] = number
        else:
            pagelist[-1].append(number)
        prev_number = number

    return [(el[0],el[-1]) for el in pagelist]

def find_vertical_line(third_val, words_in_box, num_lines_in_block):
    third_variance = 100
    vertical_splits = [third_val+el for el in list(range((-third_variance), (third_variance)))]
    intersect_count_list = []
    for split_line in vertical_splits:
        intersect_count = 0
        for line in words_in_box:
            if line['l'] < split_line < line['r']:
                intersect_count+=1
        intersect_count_list.append(intersect_count)
    masked_list = [el[1] for el in zip(intersect_count_list, vertical_splits) if 0<= el[0]<(num_lines_in_block/2)]
    if masked_list:
        vertical_split_line = np.median(masked_list)
        return vertical_split_line
    else:
        masked_list = [el[1] for el in zip(intersect_count_list, vertical_splits) if 0<= el[0]<(num_lines_in_block/3)]
        if masked_list:
            vertical_split_line = np.median(masked_list)
            return vertical_split_line
    return None


class one_col_page:
    """ Class that holds templates for one column pages"""
    def __init__(self, page):
        self.page = page
        self.lines_in_page = page['line_info']
        self.words_in_page = page['word_info']
        self.all_line_heights = [w['b'] - w['t'] for w in self.lines_in_page]
        self.left_indents = []
    
    def create_new_boxes(self, top_bounds, bot_bounds, box):
        """ Given a set of top and bounds, split the outer box into a set of smaller ones"""
        final_splits = []
        new_boxes_m = []
        if top_bounds:
            new_boxes = [{'t': c_top, 'b': c_bot,
                          'l': box['l'], 'r': box['r'],
                          'w': box['r']-box['l'], 'h': c_bot-c_top,
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
            ### Case 1: Check the whole box
            if not moodys_box:
                moodys_box = self.find_moodys_box(box, expected_num_lines)
                if moodys_box:
                    final_boxes.append(box)
                    continue
            ### Check 2: First line in box
            if lines_in_block and not moodys_box:
                sorted_lines = sorted(lines_in_block, key=lambda k: ("t" not in k, k.get('t', None)))
                for top_line in sorted_lines[:2]:
                    l_indent = top_line['l']-box['l']
                    r_indent = box['r'] - top_line['r']
                    large_lr_indent = (l_indent > 1000) or (r_indent > 1000)
                    if large_lr_indent:
                        moodys_box = self.find_moodys_box(top_line, 1)
                        if moodys_box:
                            box.update({'t':top_line['b'], 'h':box['b']-top_line['b']})
                            final_boxes.append(box)
                            break
            ### Check 3: Try splitting the box again
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
                if sel_box['t'] > moodys_box['t'] and sel_box not in zones_out:
                    zones_out.append(sel_box)
        else:
            zones_out = final_boxes
        return zones_out
    
    def title_splits_1_col(self, boxes):
        """ Partition Step 3 : Horizontal splits in vertically split boxes"""
    #    min_tab_width = 20
        zones_out = []
        new_boxes_m = []
        self.all_line_heights = [w['b'] - w['t'] for w in self.lines_in_page]
        height_5_percentile = np.percentile(self.all_line_heights, 90)
        for box in boxes:
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
                ## Case 1: Check for entirely capital string.
                if line_str.isupper() and len(words_in_line) > 1:
                    if line_l['t'] not in tops:
                        tops.append(line_l['t'])
                        bots.append(line_l['b'])
                ## Case 2: Check for heights greater than some 99 percentile.
                elif (line_l['b']-line_l['t']) > height_5_percentile and \
                left_gap > 0.1*box_width and right_gap > 0.1*box_width:
                    if line_l['t'] not in tops:
                        tops.append(line_l['t'])
                        bots.append(line_l['b'])
                ## Case 3: Check if the line is centered
                elif left_gap > 0.1*box_width and right_gap > 0.1*box_width and \
                abs(gap_diff) < 0.01*box_width and \
                abs(center_diff) < 0.005*box_width:
                    if line_l['t'] not in tops:
                        tops.append(line_l['t'])
                        bots.append(line_l['b'])
                # Case 4: Check for "Moody's" title 
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
        return zones_out
    
    def vertical_splits_one_col(self, boxes_in):
        zones_out = []
        final_splits = []
        for working_block in boxes_in:
            to_split = [copy.deepcopy(working_block)]
            new_boxes = []
            lines_in_block = get_lines_in_box(working_block, self.lines_in_page)
            words_in_block = get_words_in_box(working_block, self.words_in_page)
            half_width = (working_block['r']-working_block['l'])/2
            box_half = working_block['l'] + half_width
            lines_on_left = {}
            lines_on_right = {}
            for line_num, line in enumerate(lines_in_block):
                if line['l'] > box_half+10:
                    lines_on_right[line_num] = line
                if box_half > line['r']:
                    lines_on_left[line_num] = line
            split_line_nums = sorted([*lines_on_right])
            consecutive_lines = formatlinelist(split_line_nums)
            binwidth_2 = 60
            tops = []; bots = []
            for line_range in consecutive_lines:
                if lines_in_block[line_range[0]:line_range[1]]:
                    average_line_width = np.mean([ll['r']-ll['l'] for ll in lines_in_block[line_range[0]:line_range[1]]])
                else: average_line_width=0
                block_lefts = [lines_in_block[l_num]['l'] for l_num in range(line_range[0], line_range[1]+1)]
                y_freq, y_ranges = np.histogram(block_lefts, \
                                                bins=np.arange(min(block_lefts), \
                                       max(block_lefts) + binwidth_2, binwidth_2))
                if y_freq.any():
                    if line_range[-1]-line_range[0]>2 and max(y_freq)>2 and average_line_width > 0.4*half_width:
                        lines_selected = lines_in_block[line_range[0]: line_range[-1]+1]
                        sorted_td = sorted(lines_selected, key=lambda k: ("t" not in k, k.get('t', None)))
                        tops.append(sorted_td[0]['t'])
                        bots.append(sorted_td[-1]['b'])
            
            if tops:
                for c_top, c_bot in zip(tops, bots):
                    left_box = {'t': c_top, 'b': c_bot,
                                'l': working_block['l'], 'r': box_half,
                                'w': box_half-working_block['l'], 'h': c_bot-c_top,
                                'color':'orange', 'num_col': 'two'}
                    right_box = {'t': c_top, 'b': c_bot,
                                 'l': box_half, 'r': working_block['r'],
                                 'w': working_block['r']-box_half, 'h': c_bot-c_top,
                                 'color':'orange', 'num_col': 'two'}
                    is_left_table = check_table(get_words_in_box(left_box, words_in_block), 
                                                left_box['r']-left_box['l'],
                                                len(get_lines_in_box(left_box, lines_in_block)))
                    is_right_table = check_table(get_words_in_box(right_box, words_in_block), 
                                                 right_box['r']-right_box['l'],
                                                 len(get_lines_in_box(right_box, lines_in_block)))
                    if is_right_table and is_left_table:
                        new_boxes = [{'t': c_top, 'b': c_bot,
                                      'l': working_block['l'], 'r': working_block['r'],
                                      'w': working_block['r']-working_block['l'], 'h': c_bot-c_top,
                                      'color':'seagreen', 'num_col': 'one'}]
                    else:
                        new_boxes= [left_box, right_box]
                    only_split_boxes = []
                    final_splits.extend(new_boxes)
                    for outer_box in to_split:
                        if left_box['t'] == outer_box['t'] and left_box['b'] < outer_box['b']:
                            box1 = copy.deepcopy(outer_box); box2 = copy.deepcopy(outer_box)
                            box1.update({'b':left_box['b']}); box2.update({'t':left_box['b']})
                            only_split_boxes.extend([box1, box2])
                        elif outer_box['t'] < left_box['t'] < outer_box['b'] and outer_box['t'] < left_box['b'] < outer_box['b']:
                            box1 = copy.deepcopy(outer_box); box3 = copy.deepcopy(outer_box)
                            box1.update({'b':left_box['t']});box3.update({'t':left_box['b']})
                            only_split_boxes.extend([box1, box3])
                        elif outer_box['t'] < left_box['t'] < outer_box['b'] and left_box['b'] == outer_box['b']:
                            box1 = copy.deepcopy(outer_box)
                            box1.update({'b':left_box['t']})
                            only_split_boxes.extend([box1])
                        else:
                            only_split_boxes.append(outer_box)
                        to_split = only_split_boxes
                final_splits.extend(only_split_boxes)
            else:final_splits.append(working_block)
        for fs in final_splits:
            if fs not in zones_out:
                zones_out.append(fs)
        return zones_out
    
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
                                box1_l = copy.deepcopy(outer_box); box2_l = copy.deepcopy(outer_box)
                                box1_l.update({'b':nbm['b']})
                                box2_l.update({'t':nbm['b']})
                                box1_r = copy.deepcopy(to_split_r[bnum]); box2_r = copy.deepcopy(to_split_r[bnum])
                                box1_r.update({'b':nbm['b']})
                                box2_r.update({'t':nbm['b']})
                                boxes_l_split.extend([box1_l, box2_l])
                                boxes_r_split.extend([box1_r, box2_r])
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
                                box1_l = copy.deepcopy(outer_box); box2_l = copy.deepcopy(outer_box)
                                box1_l.update({'b':nbm['t']})
                                box2_l.update({'t':nbm['t']})
                                box1_r = copy.deepcopy(to_split_r[bnum]); box2_r = copy.deepcopy(to_split_r[bnum])
                                box1_r.update({'b':nbm['t']})
                                box2_r.update({'t':nbm['t']})
                                boxes_l_split.extend([box1_l, box2_l])
                                boxes_r_split.extend([box1_r, box2_r])
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
    
    def right_gap_splits(self, box):
        """ Perform horizontal splits based on space at the end of a line when the 
        next line is a new paragraph."""
        max_word_width = np.percentile([w['r']-w['l'] for w in self.words_in_page], 60)
        new_boxes = []
        lines_in_block = redefined_line_bounds(self.get_split_lines(box))
        is_table = check_table(get_words_in_box(box, self.words_in_page), 
                               box['r']-box['l'], len(lines_in_block))
        if box['num_col']=='one':
            to_split = copy.deepcopy(box)
            sorted_ud = sorted(lines_in_block, key=lambda k: ("t" not in k, k.get('t', None)))
            for lnum, line in enumerate(sorted_ud[1:-1], 1):
                right_space = box['r']-line['r']
                if right_space > max_word_width:
                    new_boxes.append({'t':to_split['t'], 'b': line['b'],
                                      'l':to_split['l'], 'r': to_split['r'],
                                      'w':to_split['r']-to_split['l'], 'h': line['b']-to_split['t'],
                                      'color':'seagreen', 'num_col': 'one'})
                    to_split.update({'t':line['b']})
                    self.left_indents.append(lines_in_block[lnum+1]['l'])
            new_boxes.append(to_split)
        else:
            new_boxes.append(box)
        return new_boxes
    
    def left_indent_split(self, boxes_in):
        """ Split the box if there is a left indent at the start of the box."""
        one_col_tab = 350
        boxes_out = []
        for in_box in boxes_in:
            if self.left_indents:
                mean_left_indent = np.mean(self.left_indents)
            else:
                mean_left_indent = min(boxes_in, key=lambda x:x['l'])['l']+one_col_tab
            new_boxes=[]
            to_split = copy.deepcopy(in_box)
            lines_in_block = redefined_line_bounds(self.get_split_lines(in_box))
            is_table = check_table(get_words_in_box(in_box, self.words_in_page), 
                                   in_box['r']-in_box['l'], len(lines_in_block))
            if not is_table and in_box['num_col']=='one':
                sorted_ud = sorted(lines_in_block, key=lambda k: ("t" not in k, k.get('t', None)))
                for lnum, line in enumerate(sorted_ud[1:-1], 1):
                    is_indent = 0.85*mean_left_indent<line['l']<1.25*mean_left_indent
                    next_line_indent = lines_in_block[lnum+1]['l']<line['l']
                    if is_indent and next_line_indent:
                        self.left_indents.append(line['l'])
                        new_boxes.append({'t':to_split['t'], 'b':line['t'],
                                   'l':to_split['l'], 'r': to_split['r'],
                                   'w':to_split['w'], 'h': to_split['h'],
                                   'color':'seagreen', 'num_col':to_split['num_col']})
                        to_split.update({'t':line['t']})
                new_boxes.append(to_split)
            else:
                new_boxes.append(in_box)
            boxes_out.extend(new_boxes)
        return boxes_out
    
    def minor_horizontal_splits(self, boxes_in):
        """ Split one column horizontal boxes into minor splits."""
        boxes_out = []
        for box in boxes_in:
            if box['num_col']=='one':
                right_indent_splits = self.right_gap_splits(box)
                left_splits = self.left_indent_split(right_indent_splits)
                boxes_out.extend(left_splits)
            elif box['num_col']=='two':
                splitter = two_col_page(self.page)
                minor_splits = splitter.minor_horizontal_splits([box])
                boxes_out.extend(minor_splits)
            elif box['num_col']=='three':
                threeColSplitter = three_col_page(self.page)
                boxes_out.extend(threeColSplitter.minor_horizontal_splits([box]))
        final_zones =[]
        for box in boxes_out:
            words_in_box = get_words_in_box(box, self.words_in_page)
            if words_in_box:
                final_zones.append(box)
        return final_zones
    
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
                                  'color':'seagreen','num_col':'one'})
        final_sorted = sorted(sorted_ud, key=lambda k: ("t" not in k, k.get('t', None)))
        return final_sorted
    
    def get_split_lines(self, box):
        # Split words in box into lines.
        words_in_box = get_words_in_box(box, self.words_in_page)
        sorted_td = sorted(words_in_box, key = lambda k: ("t" not in k, k.get('t', None)))
        prev_line_bottom = 0
        line_split = []
        new_line = []
        for word in sorted_td:
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
    
    def title_splits_3_1_col(self, boxes):
        """ Partition Step 3 : Horizontal splits in vertically split boxes"""
    #    min_tab_width = 20
        zones_out = []
        new_boxes_m = []
        self.all_line_heights = [w['b'] - w['t'] for w in self.lines_in_page]
#        height_5_percentile = np.percentile(self.all_line_heights, 90)
        for box in boxes:
            tops = []; bots = []
#            box_height = box['b'] - box['h']
            lines_in_box = redefined_line_bounds(self.get_split_lines(box))
            for line_num, line_l in enumerate(lines_in_box[1:]):
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
        return zones_out
    
    def three_one_separation(self, boxes):
        zones_out = []
        lines_out = []
        for box in boxes:
            lines_in_box = redefined_line_bounds(self.get_split_lines(box))
            lines_out.extend(lines_in_box)
            sorted_td = sorted(lines_in_box, key = lambda k: ("t" not in k, k.get('t', None)))
            if sorted_td:
                prev_line_bottom = sorted_td[0]['b']
#                line_split = []
#                new_line = []
                for line in sorted_td[1:]:
                    space_from_prev_line = line['t'] - prev_line_bottom
                    if space_from_prev_line > 55:
                        split_upper = copy.deepcopy(box)
                        split_upper.update({'b': prev_line_bottom, 'h': prev_line_bottom-split_upper['t']})
                        zones_out.append(split_upper)
                        box.update({'t':prev_line_bottom, 'h': box['b']-prev_line_bottom})
                    prev_line_bottom = line['b']
            zones_out.append(box)
        return zones_out
    
    
    def find_3_col_intersects(self, main_box, gutter_l, gutter_r, box_1, box_2, box_3, words_in_block, lines_in_block):
        boxes_out = []
        all_of_them = []
        tops = []; bots = []
        for line in lines_in_block:
            only_words = [w for w in get_words_in_box(line, words_in_block) if w['word'] not in string.punctuation]
            for word in only_words:
                gutter_line = False
                if word['t'] >= box_1['t'] and word['b'] <= box_1['b']:
                    if (gutter_l['l'] < word['l'] and gutter_l['r'] > word['r']) or (gutter_r['l'] < word['l'] and gutter_r['r'] > word['r']):
                        gutter_line = True
                    elif (gutter_l['l'] <= word['l'] < gutter_l['r'] and word['r'] > gutter_l['r']) or (gutter_r['l'] <= word['l'] < gutter_r['r'] and word['r'] > gutter_r['r']):
                        gutter_line = True
                    elif (word['l'] < gutter_l['l'] and gutter_l['l'] < word['r'] < gutter_l['r']) or (word['l'] < gutter_r['l'] and gutter_r['l'] < word['r'] < gutter_r['r']):
                        gutter_line = True
                    elif (word['l'] < gutter_l['l'] and word['r'] > gutter_l['r']) or (word['l'] < gutter_r['l'] and word['r'] > gutter_r['r']):
                        gutter_line = True
                if gutter_line:
                    tops.append(line['t'])
                    bots.append(line['b'])
                    break
        non_gap_boxes = self.unsplit_boxes(tops, bots, main_box, main_box['l'], main_box['r'])
        if non_gap_boxes:
            to_split_l = copy.deepcopy([box_1])
            to_split_m = copy.deepcopy([box_2])
            to_split_r = copy.deepcopy([box_3])
            for num, ngb in enumerate(non_gap_boxes):
                boxes_l_split = []
                boxes_m_split = []
                boxes_r_split = []
                for bnum, outer_box in enumerate(to_split_l):
                    if ngb['t'] == outer_box['t'] and ngb['b'] < outer_box['b']:
                        box1_l = copy.deepcopy(outer_box); box2_l = copy.deepcopy(outer_box)
                        box1_l.update({'b':ngb['b']}); box2_l.update({'t':ngb['b']})
                        boxes_l_split.extend([box1_l, box2_l])
                        
                        box1_m = copy.deepcopy(to_split_m[bnum]); box2_m = copy.deepcopy(to_split_m[bnum])
                        box1_m.update({'b':ngb['b']}); box2_m.update({'t':ngb['b']})
                        boxes_m_split.extend([box1_m, box2_m])
                        
                        box1_r = copy.deepcopy(to_split_r[bnum]); box2_r = copy.deepcopy(to_split_r[bnum])
                        box1_r.update({'b':ngb['b']}); box2_r.update({'t':ngb['b']})
                        boxes_r_split.extend([box1_r, box2_r])
                    elif outer_box['t'] < ngb['t'] < outer_box['b'] and outer_box['t'] < ngb['b'] < outer_box['b']:
                        box1_l = copy.deepcopy(outer_box); box3_l = copy.deepcopy(outer_box)
                        box1_l.update({'b':ngb['t']}); box3_l.update({'t':ngb['b']})
                        boxes_l_split.extend([box1_l, box3_l])
                        
                        box1_m = copy.deepcopy(to_split_m[bnum]); box3_m = copy.deepcopy(to_split_m[bnum])
                        box1_m.update({'b':ngb['t']}); box3_m.update({'t':ngb['b']})
                        boxes_m_split.extend([box1_m, box3_m])
                        
                        box1_r = copy.deepcopy(to_split_r[bnum]); box3_r = copy.deepcopy(to_split_r[bnum])
                        box1_r.update({'b':ngb['t']}); box3_r.update({'t':ngb['b']})
                        boxes_r_split.extend([box1_r, box3_r])
                    elif outer_box['t'] < ngb['t'] < outer_box['b'] and ngb['b'] == outer_box['b']:
                        box1_l = copy.deepcopy(outer_box); box2_l = copy.deepcopy(outer_box)
                        box1_l.update({'b':ngb['t']}); box2_l.update({'t':ngb['t']})
                        boxes_l_split.extend([box1_l, box2_l])
                        
                        box1_m = copy.deepcopy(to_split_m[bnum]); box2_m = copy.deepcopy(to_split_m[bnum])
                        box1_m.update({'b':ngb['t']}); box2_m.update({'t':ngb['t']})
                        boxes_m_split.extend([box1_m, box2_m])
                        
                        box1_r = copy.deepcopy(to_split_r[bnum]); box2_r = copy.deepcopy(to_split_r[bnum])
                        box1_r.update({'b':ngb['t']}); box2_r.update({'t':ngb['t']})
                        boxes_r_split.extend([box1_r, box2_r])
                    else:
                        boxes_l_split.extend([outer_box])
                        boxes_m_split.extend([to_split_m[bnum]])
                        boxes_r_split.extend([to_split_r[bnum]])
                to_split_l = copy.deepcopy(boxes_l_split)
                to_split_m = copy.deepcopy(boxes_m_split)
                to_split_r = copy.deepcopy(boxes_r_split)
            all_of_them += to_split_l+to_split_m+to_split_r+non_gap_boxes
            ### Merge the single col boxes that were unsplit
            sorted_td = sorted(all_of_them, key=lambda k: ("t" not in k, k.get('t', None)))
            boxes_out = [sorted_td[0]]
            for bnum, box in enumerate(sorted_td,1):
                if box['num_col']=='one' and boxes_out[-1]['num_col']=='one':
                    boxes_out[-1]['b']=box['b']
                else:
                    boxes_out.append(box)
        else:
            boxes_out.extend([box_1, box_2, box_3])
        return boxes_out
    
    def vertical_splits_three_col(self, boxes_in):
        boxes_out = []
        for box in boxes_in:
            words_in_block = get_words_in_box(box, self.words_in_page)
            lines_in_block = redefined_line_bounds(self.get_split_lines(box))
            is_table = check_table(words_in_block, box['w'], len(lines_in_block))
            if not is_table and words_in_block:
                one_third = (box['r']-box['l'])/3
                page_left = box['l']
                page_right = box['r']
                words_in_single_box = get_words_in_box(box, self.words_in_page)
                text_one_third = page_left + one_third
                text_two_third = page_right- one_third
                
                vertical_split_1= find_vertical_line(text_one_third, words_in_single_box, len(lines_in_block))
                vertical_split_2 = find_vertical_line(text_two_third, words_in_single_box, len(lines_in_block))
                if vertical_split_1 and vertical_split_2:
                    box_1 = {'t': box['t'], 'b': box['b'],
                             'l': page_left, 'r': vertical_split_1,
                             'w': vertical_split_1-page_left, 'h': box['b']-box['t'],
                             'color':'mediumvioletred', 'num_col': 'three'}
                    box_2 = {'t': box['t'], 'b': box['b'],
                             'l': vertical_split_1, 'r': vertical_split_2,
                             'w': vertical_split_2-vertical_split_1, 'h': box['b']-box['t'],
                             'color':'mediumvioletred', 'num_col': 'three'}
                    box_3 = {'t': box['t'], 'b': box['b'],
                             'l': vertical_split_2, 'r': page_right,
                             'w': page_right-vertical_split_2, 'h': box['b']-box['t'],
                             'color':'mediumvioletred', 'num_col': 'three'}
                    gutter_l = {'l':box_1['r']-1, 'r':box_2['l']+1,'w':box_2['l']-box_1['r']+2}
                    gutter_r = {'l':box_2['r']-1, 'r':box_3['l']+1,'w':box_3['l']-box_2['r']+2}
                    split_again = self.find_3_col_intersects(box, gutter_l, gutter_r, box_1, box_2, box_3, words_in_block, lines_in_block)
                    boxes_out.extend(split_again)
                else:
                    boxes_out.append(box)
            else:
                boxes_out.append(box)
        return boxes_out

    def readjust_boxes(self, boxes_in):
        boxes_out = []
        for box in boxes_in:
            box.update({'t':box['t']-2, 'b': box['b']+2, 'h': box['h']+4,
                        'l':box['l']-2, 'r': box['r']+2, 'w': box['w']+4})
            boxes_out.append(box)
        return boxes_out