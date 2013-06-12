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
    

def initial_window():
  options = {}
  options['defaultextension'] = '.txt'
  options['filetypes'] = [('all files', '.*'), ('text file', '.txt'), ('classifier file', '.clfr'), ('CSV file', '.csv')]
  options['parent'] = app
  options['title'] = 'Open file...'

  #store file name of the file user selects
  filename = tkFileDialog.askopenfilename(**options)

  #get the file ending
  file_ending = filename.strip().split('.')[-1]
  app.destroy()
  
  #if user specified a preconstructed classifier file
  if file_ending == 'clfr':
    ui.CLASSIFIER = utils.load_classifiers(filename)
  else:
    #else parse the file using method in utils.py
    ui.TRAINING_DATA = utils.parse_training_data(filename)
  
def test_window():
  options = {}
  options['defaultextension'] = '.txt'
  options['filetypes'] = [('all files', '.*'), ('text file', '.txt'), ('classifier file', '.clfr'), ('CSV file', '.csv')]
  options['parent'] = app
  options['title'] = 'Open file...'

  filename = tkFileDialog.askopenfilename(**options)
  outfile = open(filename.rsplit('.', 1)[0] + '.lbls', 'w')
  for vector in utils.parse_training_data(filename):
    outfile.write(str(ui.CLASSIFIER.classify_vector(vector)))
    outfile.write('\n')

  outfile.close()
  outfile = open(filename.rsplit('.', 1)[0] + '.info', 'w')

  outfile.write('Classifiers used:\n\n')
  if ui.CLASSIFIER.TYPE == 'AdaBoost':
    for c in ui.CLASSIFIER.C:
      outfile.write(c.get_info() + '\n')
  outfile.close()
  app.destroy()

def manual_window():
  for child in app.winfo_children():
    child.destroy()

  #app.columnconfigure(0, weight=1)
  #app.rowconfigure(1, weight=1)

  labelText = StringVar()
  labelText.set('Please specify at least one algorithm to use.\nParameters may be auto-selected by selecting auto under each\nalgorithm, or you make select manual and specify them yourself.')
  label1 = Label(app, textvariable=labelText, height=4)
  label1.grid(row=0, columnspan=6)

  CheckVar1 = IntVar()
  CheckVar2 = IntVar()
  CheckVar3 = IntVar()
  
  C1 = Checkbutton(app, text = 'k Nearest Neighbors', variable = CheckVar1, onvalue = 1, offvalue = 0)
  C1.grid(row = 1, columnspan=2, sticky=W)

  C1av = IntVar()
  C1ab = Radiobutton(text = 'auto', variable = C1av, value=1)
  C1ab.grid(row=2, column=1, sticky=W)

  C1mv = IntVar()
  C1mb = Radiobutton(text = 'manual', variable = C1mv, value=0)
  C1mb.grid(row=2, column=2, sticky=W)

  ls1 = StringVar()
  ls1.set('k = ')
  label2 = Label(app, textvariable=ls1)
  label2.grid(row=3, column=2, sticky=W)

  C1kv = IntVar()
  C1ke = Entry(textvariable = C1kv, width=5, state=DISABLED)
  C1ke.grid(row=3, column=3, sticky=W)

  C2 = Checkbutton(app, text = 'Support Vector Machine', variable = CheckVar2, onvalue = 1, offvalue = 0, height=1)
  C2av = IntVar()
  C2ab = Radiobutton(text = 'auto', variable = C2av, value=1)
  C2ab.grid(row=5, column=1, sticky=W)

  C2mv = IntVar()
  C2mb = Radiobutton(text = 'manual', variable = C2mv, value=0)
  C2mb.grid(row=5, column=2, sticky=W)

  ls2 = StringVar()
  ls2.set('kernel = ')
  label3 = Label(app, textvariable=ls2)
  label3.grid(row=6, column=2, sticky=W)

  kv = IntVar()
  ke = Entry(textvariable = kv, width=5, state=DISABLED)
  ke.grid(row=6, column=3, sticky=W)

  ls3 = StringVar()
  ls3.set('margin = ')
  label4 = Label(app, textvariable=ls3)
  label4.grid(row=7, column=2, sticky=W)

  mv = IntVar()
  me = Entry(textvariable = mv, width=5, state=DISABLED)
  me.grid(row=7, column=3, sticky=W)

  ls4 = StringVar()
  ls4.set('gamma = ')
  label5 = Label(app, textvariable=ls4)
  label5.grid(row=8, column=2, sticky=W)

  gv = IntVar()
  ge = Entry(textvariable = gv, width=5, state=DISABLED)
  ge.grid(row=8, column=3, sticky=W)

  C3 = Checkbutton(app, text = 'Naive Bayes', variable = CheckVar3, onvalue = 1, offvalue = 0, height=1)

  C3av = IntVar()
  C3ab = Radiobutton(text = 'auto', variable = C3av, value=1)
  C3ab.grid(row=10, column=1, sticky=W)

  C3mv = IntVar()
  C3mb = Radiobutton(text = 'manual', variable = C3mv, value=0)
  C3mb.grid(row=10, column=2, sticky=W)

  ls5 = StringVar()
  ls5.set('binning = ')
  label6 = Label(app, textvariable=ls5)
  label6.grid(row=11, column=2, sticky=W)

  bv = IntVar()
  be = Entry(textvariable = bv, width=5, state=DISABLED)
  be.grid(row=11, column=3, sticky=W)


  C2.grid(row=4, columnspan=2, sticky=W)
  C3.grid(row=9, columnspan=2, sticky=W)

  ls6 = StringVar()
  ls6.set(' ')
  label7 = Label(app, textvariable=ls6)
  label7.grid(row=12, column=2, sticky=W)
  button1 = Button(app, text='Next', command=sys.exit)
  button1.grid(row=13, column=4, padx=5, pady=5)



def auto_window():
  for child in app.winfo_children():
    child.destroy()
  labelText = StringVar()
  labelText.set('Please specify data to be classified: \n\nNote: if input is named data.txt, output file of\nlabels will be in same folder, named data.lbls')
  label1 = Label(app, textvariable=labelText, height=4)
  label1.pack()

  ui.CLASSIFIER = mcl.idk_ML(ui.TRAINING_DATA, val_data = ui.TRAINING_DATA)

  button1 = Button(app, text='Open file...', command=test_window)
  button1.pack(padx=5, pady=5)
  app.mainloop()


  
  

app = Tk()
app.title('SWAG GUI')
app.geometry('400x400+200+200')
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
button1 = Button(app, text='Open file...', command=initial_window)
button1.pack(padx=5, pady=5)

app.mainloop()

app = Tk()
app.title('SWAG GUI')
app.geometry('400x400+200+200')

#user specified classifier instead of training data, so skip straight to the classification of data
if ui.CLASSIFIER != None:
  lab = StringVar()
  lab.set('Please specify data to be classified: \n\nNote: if input is named data.txt, output file of\nlabels will be in same folder, named data.lbls')
  el2 = Label(app, textvariable=lab, height=4)
  el2.pack()

  button1 = Button(app, text='Open file...', command=test_window)
  button1.pack(padx=5, pady=5)
  app.mainloop()

#user specified training data, now we must ask if he wants auto selection
#or manual selection of algorithms (parameters can still be auto-tuned)
else:
  lT = StringVar()
  lT.set('Please select whether you want to engage in\nmanual or automatic selection of algorithms')
  el1 = Label(app, textvariable=lT, height=4
  el1.pack()

  button1 = Button(app, text='Automatic', command=auto_window)
  button1.pack(padx=5, pady=5)
  button2 = Button(app, text='Manual', command=manual_window)
  button2.pack(padx=5, pady=5)



#run the first window on app
app.mainloop()


