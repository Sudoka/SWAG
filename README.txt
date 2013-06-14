-------------------------
INSTALLATION INSTRUCTIONS
-------------------------
Once extracting all files:

1. Navigate to SWAG/lib/libsvm
2. Run the command 'make'
3. Navigate to SWAG/lib/easygui
4. Run the command 'sudo python setup.py install'

----------
USING SWAG
----------
There are two ways to use SWAG, through a simple GUI interface, or simply by
invoking code in classifiers.py, meta_classifiers.py, and utils.py directly.
It is also highly advised to reading the information present in swagguide.pdf
in order to acclimate yourself with machine learning and how SWAG uses it.

-------------
USING THE GUI
-------------
The GUI is an executable file and is used by simply navigating to SWAG/code
and executing the following command: './gui.py'

The GUI is fairly simple and intuitive to use, however its output can be
complicated for a new user. If we choose to create a new classifier, and we
classify some data with this newly constructed classifier, we will be presented
with a few new files, listed here:

	your_filename.lbls
		contains the predicted labels from your classified data

	your_filename.info
		contains information about the classifiers used

	your_filename.clfr (and your_filename-svm.clfr if the classifier you
	constructed contained a SVM)
		cached classifier you constructed.

As a side note, when loading a previously cached classifier, only specify
your_filename.clfr, and not your_filename-svm.clfr, as the program will
automatically look for your_filename-svm.clfr.

-----------------
USING SOURCE CODE
-----------------
All code needed is provided in three files:

	classifiers.py
		contains multiple learning algorithms, each algorithm is invoked
		by creating a new object instance using provided constructors

	meta_classifiers.py
		contains meta classification algorithms, at the present only has
		AdaBoost.

	utils.py
		contains several utility functions used in SWAG, such as storing
		classifiers to disk, and loading them, etc.

--------------
TESTING IT OUT
--------------
If you navigate to SWAG/code/test/ you will find two files:

	train-bc.csv
		half of the breast cancer dataset for training your classifier
	
	test-bc.csv
		the other half of breast cancer data, use to test classifier

It is advised to use the GUI for new users of SWAG, but people comfortable
with python and machine learning are welcome to use the source code directly.
