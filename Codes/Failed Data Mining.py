# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:40:13 2020

@author: Ahmad
"""

# Example on iterating through obtaining pdf files in a glob

import glob

# Specify folder name with file extension
folder = r""<INSERT FILE PATH>""
file_ext = r"\*.pdf"

# Test with a folder
for filename in glob.glob(folder + file_ext ):
    print(filename)
    
# Opening a pdf and converting the context to txt
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text

filepath = "<INSERT FILE PATH>"

txt = convert_pdf_to_txt(filepath)

# Converting a pdf to html using the command line
python "<INSERT FILE PATH>" -t html "<INSERT FILE PATH>"


    
    

