# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:50:09 2020

@author: Brendan Non-Admin
"""

import os
from bs4 import BeautifulSoup
import re

def results_as_html(results, filename=None, model_name=None):
    html_doc = (results.summary().as_html()
        .replace('class="simpletable"', 'class="simpletable table table-sm table-hover"')
        .replace("Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.", "")
    )
    html_doc = '<div class="container border border-dark mb-2">{}</div>'.format(html_doc)
    
    soup = BeautifulSoup(html_doc, 'html.parser')
    
    if model_name:
        soup.find('caption').string = model_name
    
    pretty_soup = soup.prettify()
    
    def update_soup(soup):
        return re.sub(r'<tr>\n   <td></td>\n   <th>coef</th>', '<tr class="border-bottom">\n   <td></td>\n   <th>coef</th>',
            re.sub(r'<th>\s+', r'<th>',
                re.sub(r'\s+</th>', r'</th>',
                    re.sub(r'\s+<td>\s+', r'<td>', 
                        re.sub(r'\s+</td>', r'</td>', pretty_soup)
        ))))
    pretty_soup = update_soup(pretty_soup)

    
    if filename:
        filepath = os.path.join(
            os.path.expanduser('~'),
            'Documents',
            'GitHub',
            'metis-projects-flask-app',
            'metis_app',
            'blog_posts',
            'templates',
            'blog_posts',
            'include_docs',
            'nhl-goals-regression-analysis',
            filename,
        )
        with open(filepath, 'w') as f:
            f.write(pretty_soup)
    
    return pretty_soup

#results_as_html(results, 'model1.html', 'My Cool New Model')