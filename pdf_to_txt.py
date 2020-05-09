import os
import sys
import six
import json
import os
import sys, getopt

from pdfminer import utils
import pdfminer.layout as layout
from pdfminer.pdfpage import PDFPage
from pdfminer.pdffont import PDFUnicodeNotDefined

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

from pdfminer.converter import TextConverter

from pdfminer.pdfdevice import PDFDevice, TagExtractor



from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTImage


class pdf_to_text:
    
    def __init__(self, pdfDir, txtDir):
        
        self.pdfDir = pdfDir
        self.txtDir = txtDir
    """
    This class convert PDF Novels/StoryBooks to TXT.format
    """
        
    #converts pdf, returns its text content as a string
    
    def convertMultiple(self):
        
        def convert(fname, pages=None):
            if not pages:
                pagenums = set()
            else:
                pagenums = set(pages)
            output = StringIO()
            manager = PDFResourceManager()
            converter = TextConverter(manager, output, laparams=LAParams())
            interpreter = PDFPageInterpreter(manager, converter)
            
            infile = open(fname, 'rb')
            for page in PDFPage.get_pages(infile, check_extractable=False):
                interpreter.process_page(page)
            infile.close()
            converter.close()
            text = output.getvalue()
            output.close
            return text
    
        if self.pdfDir == "": self.pdfDir = os.getcwd() + "\\" #if no pdfDir passed in 
        for pdf in os.listdir(self.pdfDir): #iterate through pdfs in pdf directory
            fileExtension = pdf.split(".")[-1]
            if fileExtension == "pdf":
                pdfFilename = self.pdfDir + pdf
                text = convert(pdfFilename) #get string of text content of pdf
                textFilename = self.txtDir + pdf + ".txt"
                textFile = open(textFilename, "w", encoding='utf-8', errors='ignore') #make text file
                textFile.write(text) #write text to text file
    
