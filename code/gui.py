#!/usr/bin/python -tt
import re
import sys
from Tkinter import *
import tkFileDialog
sys.path.append('..')
import classifiers as cl
import meta_classifiers as mcl
import utils
import cPickle as pickle

class UserInput:
  
  def __init__(self):
    
    self.TRAINING_DATA = []
    self.CLASSIFIER = None
    self.OUTPUT_FILENAME = None
    

def askopenfile():
  options = {}
  options['defaultextension'] = '.txt'
  options['filetypes'] = [('all files', '.*'), ('text file', '.txt'), ('classifier file', '.clfr'), ('CSV file', '.csv')]
  options['parent'] = app
  options['title'] = 'Open file...'

  #store file name of the file user selects
  filename = tkFileDialog.askopenfilename(**options)

  #get the file ending
  file_ending = filename.strip().split('.')[-1]
  
  #if user specified a preconstructed classifier file
  if file_ending == 'clfr':
    ui.CLASSIFIER = pickle.load(filename)
  else:
    #else parse the file using method in utils.py
    ui.TRAINING_DATA = utils.parse_training_data(filename)

  #we now have our information, so destroy the app
  app.destroy()

def askopenfilename(self):

  """Returns an opened file in read mode.
  This time the dialog just returns a filename and the file is opened by your own code.
  """

  # get filename
  filename = tkFileDialog.askopenfilename(**self.file_opt)

  # open file on your own
  if filename:
    return open(filename, 'r')

def asksaveasfile(self):

  """Returns an opened file in write mode."""

  return tkFileDialog.asksaveasfile(mode='w', **self.file_opt)

def asksaveasfilename(self):

  """Returns an opened file in write mode.
  This time the dialog just returns a filename and the file is opened by your own code.
  """

  # get filename
  filename = tkFileDialog.asksaveasfilename(**self.file_opt)

  # open file on your own
  if filename:
    return open(filename, 'w')

def askdirectory(self):

  """Returns a selected directoryname."""

  return tkFileDialog.askdirectory(**self.dir_opt)

def beenClicked():
  radioValue = relStatus.get()
  tkinter.messagebox.showinfo('You clicked', radioValue)

def changeLabel():
  name = 'Thanks for the click ' + yourName.get()
  labelText.set(name)
  yourName.delete(0, END)
  yourName.insert(0, 'My name is Amrit')



app = Tk()
app.title('SWAG GUI')
app.geometry('450x600+200+200')
ui = UserInput()

labelText = StringVar()
labelText.set('Please specify a precomputed classifier or training data:')
label1 = Label(app, textvariable=labelText, height=4)
label1.pack()

'''
checkBoxVal = IntVar()
checkBox1 = Checkbutton(app, variable=checkBoxVal, text='Happy?')
checkBox1.pack()
'''

'''
custName = StringVar(None)
yourName = Entry(app, textvariable=custName)
yourName.pack()
'''

#button to get input file, whether it is classifier or training data
button1 = Button(app, text='Open file...', command=askopenfile)
button1.pack(padx=5, pady=5)

#run the first window on app
app.mainloop()


