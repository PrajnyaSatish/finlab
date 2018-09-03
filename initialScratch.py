# In[0]:
from pprint import pprint as pp
import re
from timeit import default_timer as timer
import os
import pandas as pd




# In[]:
## Change the current working directory
os.chdir('/home/prajnya/Desktop/fin_lab_workstation')


list_info_file = 'Data/list_sections20180318.xlsx'
xl = pd.ExcelFile(list_info_file)

list_info = xl.parse(xl.sheet_names[0])

# In[]:
filename = 'xml_output_201802/Industrials_1940_raw_image.XML'

with open(filename, encoding = 'utf-8') as rd:
    file_content = rd.readlines()





# In[]:

def get_microfiche_name(str_in):
    split_string = str_in.split("\"")
    if ".tif" in split_string[1]:
        return split_string[1]
    return ''

    
# In[]:
## Parse the info and separate them by pages so it's easy to visualize the details
xml_each_line = []
new_line_content = ''
page_lines = []
pageCount = -1
line_dict = {}

start = timer()
for xml_line in file_content:
    if xml_line.startswith('<source '):
        page_lines.append(line_dict)
        pageCount+=1
        microfiche_name = get_microfiche_name(xml_line)
        line_dict = {'pageNum':pageCount, 'microfiche':microfiche_name, 'content': []}
    if xml_line.startswith('</ln>'):
        xml_line = xml_line.replace('\n','')
        new_line_content+=xml_line
        line_dict['content'].append(new_line_content)
        new_line_content = ''
    else:
        if xml_line.startswith('<ln '):
            new_line_content = xml_line
        elif xml_line.startswith('<wd ') or xml_line.startswith('<space '):
            new_line_content+=xml_line
        elif xml_line.startswith('</wd>') or xml_line.startswith('</space>'):
            new_line_content+=xml_line
# Remove the first unnecessary line
page_lines = page_lines[1:]
end = timer()
print("\nTime to read XML lines pagewise = {0} seconds".format(round(end-start,3)))

# In[01]:
def parse_features(input_string):
    split_string = input_string.split('=')
    try:
        int_feature = int(split_string[1].strip('"'))
        return split_string[0], int_feature
    except:
        return split_string[0], split_string[1].strip('"') 





# In[]:
## Convert the information in each line to word features
start = timer()

all_pages = []
for current_page in page_lines:
    page_info = {'pageNum': current_page['pageNum'], 'microfiche': current_page['microfiche'], 'page_lines': []}
    for line_num, line_block in enumerate(current_page['content']):
        line_information = {'words':[], 'string': ''}
        split_lines = line_block.split('\n')
        word_num = 0
        for inner_num, line in enumerate(split_lines):
            line = re.sub(r'[<>]', ' ', line)
            whitespace_split = line.split(' ')
            whitespace_split = whitespace_split[2:-2]
            if 'baseLine' in line:
                for feature in whitespace_split:
                    if '=' in feature:
                        feat_name, feat_val = parse_features(feature)
                        line_information[feat_name] = feat_val
            ## Only lines with words and word shape features have some values in the whitespace_split list/
            ## These lines are parsed to get word information.
            elif whitespace_split:
                word_features = {}
                word_in_question = whitespace_split[-1]
                word_in_question = word_in_question.replace('&apos;',"'")
                word_in_question = word_in_question.replace('&amp;',"&")
                line_information['words'].append(word_in_question)
                line_information['string'] = ''.join([line_information['string'], word_in_question+' '])
                
                for feature in whitespace_split:
                    if '=' in feature:
                        feat_name, feat_val = parse_features(feature)
                        word_features[feat_name] = feat_val
                        word_features['word'] = word_in_question
                        
                ## Get the width of spaces after each word.
                space_line = split_lines[inner_num+1]
                if '=' in space_line:
                    space_line = re.sub(r'[<>/]', ' ', space_line)
                    width_feature = space_line.split(' ')[2] #Assuming that the feature is always the third element after the split.
                    space_after = parse_features(width_feature)[1]
                    word_features['space_after_word'] = space_after
                    
                line_information[word_num] = word_features
                word_num += 1
        page_info['page_lines'].append(line_information)
    all_pages.append(page_info)
end = timer()
print("Time to parse XML = {0} seconds".format(round(end-start,3)))



# In[3]:


def is_number(word_list):
    for word in word_list:
        if word.isdigit():
            return 1
        elif word[1:].replace(',','').isdigit():
            return 1
    return 0

# In[4]:
## Code to plot and save the images for the entire xml file.
    ## NEed to run only once.
import matplotlib.pyplot as plt
import matplotlib.patches as patches

for page in all_pages[:100]:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    for d_line in page['page_lines']:
        for i in range(len(d_line['words'])):
            current_word_ax = (d_line[i]['l'],d_line[i]['b'])
            width = d_line[i]['r']- d_line[i]['l']
            height = d_line[i]['t'] - d_line[i]['b']
            
            word_rect = patches.Rectangle(current_word_ax, width, height, linewidth = 1, fill=False)
            ax1.add_patch(word_rect)
    ax1.plot()
    fig1.gca().invert_yaxis()
    fig1.savefig('Data/1940_Image_Plots/Industrials_1940_{0}'.format(page['pageNum']+1), dpi = 90, bbox_inches='tight')
    plt.close('all')
   

# In[3]:
## Ignore the Moody's manual title and advertiseements at the base if any.
## Check if MOODY|MANUAL|INVESTMENT occurs in the line.
def check_book_title(line):
    flag = 0
    check_for_heading = ["MOODY'S MANUAL OF INVESTMENTS",'MOODY', 'MANUAL OF INVESTMENTS']
    for check_word in check_for_heading:
        if check_word in line['string']:
            flag = 1
            ## Discard the line and break
            return flag

def check_all_caps(line_info):
    all_caps_flag = False
    if line_info['string'].isupper():
        all_caps_flag = True     
    return all_caps_flag

def has_ignore_words(string):
    ignore_list = ['comparative', 'consolidated']
    for word in ignore_list:
        if word in string.lower():
            return 1
    return 0

# In[7]:
def check_company_name(line_info_dict, line_number, line_features):
    """Check if the string is the name of a company, and if it is, append it to the list of 
    company names.
    5 Checks for single column documents - \n
    Check 1 : Space before text line > 100 dpi \n
    Check 2: Height of the word > 180 dpi \n
    Check 3: There are no numbers in the line \n
    Check 4: The number of words in the string is limited between 3 and 8 inclusive. \n
    Check 5: Ignore certain words like comparative and consolidated
    """
    word_height = line_info_dict['b'] - line_info_dict['t']
    space_before_line = max(line_info_dict['t'] - line_features[line_number-1]['b'],0)
    if line_number < len(line_features)-1:
        space_after_line = max(line_features[line_number+1]['t'] - line_info_dict['b'],0)
    else:
        space_after_line = 0
    if(space_before_line>100) and (word_height>180) and not is_number(line_info_dict['words']) and 3 <= len(line_info_dict['words']) <= 8:
        if not has_ignore_words(line_info_dict['string']):
            return line_info_dict['string']
    return ''


def if_tab(line_info_dict, columns = 1):
    """
    For single column documents, normal start location is assumed to be around 1750-1770 dpi.
    Should verify if this remains the same for rotated images.
    """
    if columns ==1:
        if line_info_dict['l'] > 1800:
            return 1
    return 0



# In[4]:

name_specific_strings = ['COMPANY', 'CORPORATION', 'LIMITED', 'LTD.', 'CO.', 'CORP', 'INC.']

            
# In[5]:
company_names = []
struct_info_company = {}
count = -1

for index, text_line in enumerate(line_features):
    moody_flag = check_book_title(text_line)
    if not moody_flag:
        comp_name = check_company_name(text_line, index, line_features)
        if comp_name:
            count+=1
            current_comp_name = comp_name
            if text_line['string'] not in company_names:
                company_names.append(text_line['string'])
                struct_info_company.update({comp_name:{}})
                continue
        space_before_line = max(text_line['t'] - line_features[index-1]['b'],0)
        if space_before_line > 10:
            if if_tab(text_line, 1) and ':' in text_line['words'][0]:
                sub_heading = text_line['words'][0].replace(':','')
                rest_of_para = ' '.join(text_line['words'][1:])
            else:
                sub_heading = 'FOR_LATER'
                rest_of_para = text_line['string']
            if current_comp_name not in struct_info_company:
                struct_info_company = {current_comp_name: {sub_heading: rest_of_para}}
            else:
                struct_info_company[current_comp_name].update({sub_heading: rest_of_para})
        else:
            if sub_heading in struct_info_company[current_comp_name]:
                struct_info_company[current_comp_name][sub_heading] += ' '+text_line['string']            
                 
# In[6]:
                

# =============================================================================
#         if first_word_height > 180:
#             company_names.append(text_line['string'])
#             print(text_line['string']+' '+str(first_word_height))
# =============================================================================
# =============================================================================
#         for word in name_specific_strings:
#             if word in text_line['string'] and is_all_caps:
#                 if word_height > 180:
#                     if text_line['string'] not in company_names:
#                         company_names.append(text_line['string'])
#                     #print(text_line['string']+str(word_height))
#                     continue
# =============================================================================
