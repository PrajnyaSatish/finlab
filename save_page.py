#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 15:46:04 2018

@author: prajnya
"""

import json

page=748
year=1941
with open('Pages/page_{0}_{1}.json'.format(page,year), 'w') as wf:
    json.dump(all_pages[page-1], wf, indent=4)