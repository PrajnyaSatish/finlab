#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 23:02:30 2018

@author: hari
"""

# -*- coding: utf-8 -*-

f = open('Industrials_1928_raw_image.XML', "r", encoding = 'utf-16-le')

i = 0
name_management = {}
management_details = []
names = []
temp_list = []
temp_list_or = []
m_flag = 0
name_flag = 0
cl_name_flag = 0
of_flag = 0
of_count = 0
str = ""
n_str = ""
for line in f:
    if line[0:3] == "<wd":
        
        line = line.split(">")[1].split("<")[0]
        
        if  cl_name_flag:
            names.append(n_str)
            n_str = ""
            cl_name_flag = 0
        
        if of_flag:
            of_count += 1
            
        if not name_flag:
            
            if line == "Management:":
                 str = line
                 m_flag = 1
            elif line == "OFFICE:" and m_flag:
                of_flag = 1
            elif line == "Comparative" or line == "Consolidated":
               if m_flag:
                    temp_list_or.append(str)
                    management_details.append(str)
                    if names[len(names)-1] not in name_management:
                        temp = []
                        temp.append(str)
                        name_management[names[len(names)-1]] = temp
                    else:
                        temp = name_management[names[len(names)-1]]
                        temp.append(str)
                        name_management[names[len(names)-1]] = temp
                    str = ""
                    m_flag = 0
                    of_flag = 0
                    of_count = 0
            elif of_flag and of_count > 50:
                if m_flag:
                    temp_list.append(str)
                    management_details.append(str)
                    if names[len(names)-1] not in name_management:
                        temp = []
                        temp.append(str)
                        name_management[names[len(names)-1]] = temp
                    else:
                        temp = name_management[names[len(names)-1]]
                        temp.append(str)
                        name_management[names[len(names)-1]] = temp
                    str = ""
                    m_flag = 0
                    of_flag = 0
                    of_count = 0
            elif line != None:
                str += line+" "
            
        elif name_flag and ("MOOD" in line or "ALPHA" in line or "DEFI" in line or "MANUAL" in line):
            name_flag = 0
            cl_name_flag = 0
            n_str = ""
            continue
        
        elif name_flag:
            if line != None:
                n_str += line+" "
            
            
    elif line[0:3] == "<ln":
            
        line = line.split("fontSize")[1].split("\"")[1]
        if int(line) == 1300 or int(line) == 1200:
            name_flag = 1
            
    elif line[0:4] == "</ln" and name_flag:
        cl_name_flag = 1
        name_flag = 0
        
        
print (len(management_details))
print (len(names))
print (len(name_management.keys()))


for name in names:
    if name not in  name_management:
        print (name)
             