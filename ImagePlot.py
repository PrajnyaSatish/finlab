#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:20:14 2018

@author: prajnya
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_rectangles(color_coded_list, dest_file):
    fig_to_draw = plt.figure()
    fig_ax = fig_to_draw.add_subplot(111)
    for el in color_coded_list:
        in_color = el['color']
        rect_list = el['word_list']
        for word in rect_list:
            current_word_ax = (word['l'], word['b'])
            width = word['r']- word['l']
            height = word['t'] - word['b']
            if 'color' in word:
                in_color = word['color']
            else: in_color = el['color']
            word_rect = patches.Rectangle(current_word_ax, width, height, linewidth=1, \
                                          color=in_color, fill=False)
            fig_ax.add_patch(word_rect)
    fig_to_draw.gca().invert_yaxis()
    plt.axis('scaled')
    fig_to_draw.savefig(dest_file, dpi=90, bbox_inches='tight')
    plt.plot()
    plt.close()
    
##########################################################################################
#    plt.hist(word_left_in_block, range=(left_lims, right_lims), \
#             bins=np.arange(min(word_left_in_block), \
#                            max(word_left_in_block) + binwidth_3, binwidth_3))
#    try:
#        plt.savefig('Histogram_Plots_1928/Page_{1}/Histogram_for_block_{0}'.format(block_num+1, \
#                    page_num+1))
#    except:
#        os.mkdir('Histogram_Plots_1928/Page_{0}'.format(page_num+1))
#        plt.savefig('Histogram_Plots_1928/Page_{1}/Histogram_for_block_{0}'.format(block_num+1, \
#                    page_num+1))
#        
#    plt.cla()
#    
###########################################################################################        
