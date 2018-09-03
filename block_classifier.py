#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:07:44 2018

@author: prajnya
"""

import json
import pandas as pd

from ImagePlot import draw_rectangles
from zone_extraction import template
from zone_extraction import get_all_words, sort_two_d, get_major_blocks

page_num = 606
with open('page_{0}_1940.json'.format(page_num+1), 'r') as rf:
    working_page = json.load(rf)

boxes = template(working_page, page_num)
draw_rectangles([{'color':'black', 'word_list':get_all_words(working_page)},
             {'color': 'seagreen', 'word_list': boxes}], \
            'major_b_page_{0}'.format(page_num+1))