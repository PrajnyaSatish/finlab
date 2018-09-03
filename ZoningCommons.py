#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:15:20 2018

@author: prajnya
"""
import numpy as np

def left_right_bounds(left_lims, right_lims):
    max_width = max(right_lims)-min(left_lims)
    selected_lefts = [ls for ls in left_lims if ls < ((max_width/2)-500)]
    if not selected_lefts:
        fixed_left_lim = min(left_lims)
    else:
        left_freq_counts, bins = np.histogram(selected_lefts, bins=120, range=(min(selected_lefts), max(selected_lefts)))
        mask = [el if el>0.05*len(left_lims) else 0 for el in left_freq_counts.tolist()]
        try:
            left_start_step = bins[np.nonzero(mask)[0][0]]
        except:
            left_start_step = bins[(left_freq_counts.tolist()).index(max(left_freq_counts))]
        left_end_step = left_start_step+100
        fixed_left_lim = left_end_step
        for val in left_lims:
            if left_start_step <= val:
                if val < fixed_left_lim:
                    fixed_left_lim = val
    
    selected_rights = [rs for rs in right_lims if rs > (min(left_lims)+(max_width/2)+1000)]
    if not selected_rights:
        fixed_right_lim = min(right_lims)
    else:
        right_freq_counts, bins = np.histogram(selected_rights, bins=170, range=(min(right_lims), max(right_lims)+100))
        mask_r = [el if el>0.05*len(right_lims) else 0 for el in right_freq_counts.tolist()]
        try:
            right_start_step = bins[np.nonzero(mask_r)[0][-1]]
        except:
            right_start_step = bins[(right_freq_counts.tolist()).index(max(right_freq_counts))]
        right_end_step = right_start_step+100
        fixed_right_lim = right_start_step
        for val in right_lims:
            if val <= right_end_step:
                if fixed_right_lim < val:
                    fixed_right_lim = val
    return fixed_left_lim, fixed_right_lim

def get_text_bounds(input_bounds):
    left_lims = []
    right_lims = []
    top_most = []
    b_most = []
    left_lims = [line['l'] for line in input_bounds]
    right_lims = [line['r'] for line in input_bounds]
    page_left, page_right = left_right_bounds(left_lims, right_lims)
    top_most.append(min(input_bounds, key=lambda k: ('t' not in k, k.get('t', None)))['t'])
    b_most.append(max(input_bounds, key=lambda k: ('b' not in k, k.get('b', None)))['b'])
    page_top = min(top_most)
    page_bottom = max(b_most) 
    return page_left, page_right, page_top, page_bottom

def mean_gap_between(horizontal_line_dict, gap_threshold):
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
    empty_line_sizes_lim = [l for l in empty_line_sizes if l < 600] # By default remove all gaps larger than 500 so it does't skew averages.
    empty_line_sizes_new = [l for l in empty_line_sizes_lim if l > 20]
    if empty_line_sizes_new:
        mean = np.mean(empty_line_sizes_new,axis=0)
        sd = np.std(empty_line_sizes_new,axis=0)
        if mean-sd>0:
            high_vals = [x for x in empty_line_sizes_new if x> mean+sd]
            if high_vals:
                percentile_thresh = max(high_vals)
            else:
                percentile_thresh = np.percentile(empty_line_sizes_new, gap_threshold)
        else:
            remove_extremes = [x for x in empty_line_sizes_new if x < mean+sd]
            non_extremes = [x for x in remove_extremes if x > mean-sd]
            mean_nx = np.mean(non_extremes)
            sd_nx = np.std(non_extremes)
            above_min = mean_nx+sd_nx
            if above_min and mean_nx < 50:
                non_extremes = [el for el in empty_line_sizes_new if el > above_min]
                percentile_thresh = np.percentile(non_extremes, gap_threshold)
            else:
                percentile_thresh = np.percentile(empty_line_sizes_new, gap_threshold)
    elif empty_line_sizes_lim:
        percentile_thresh = np.percentile(empty_line_sizes_lim, gap_threshold)
    elif empty_line_sizes:
        percentile_thresh = np.percentile(empty_line_sizes, gap_threshold)
    else:
        # TODO: Choose a number that's not pulled out of thin air 
        percentile_thresh = 150 # Magic number
    return percentile_thresh, empty_lines_list

#    mean = np.mean(empty_line_sizes,axis=0)
#    sd = np.std(empty_line_sizes,axis=0)
#    if mean-sd>0:
#        high_vals = [x for x in empty_line_sizes if x> mean+sd]
#        if high_vals:
#            percentile_thresh = max(high_vals)
#        else:
#            percentile_thresh = np.percentile(empty_line_sizes, gap_threshold)
#    else:
#        remove_extremes = [x for x in empty_line_sizes if x < mean+sd]
#        non_extremes = [x for x in remove_extremes if x > mean-sd]
#        percentile_thresh = np.percentile(non_extremes, gap_threshold)