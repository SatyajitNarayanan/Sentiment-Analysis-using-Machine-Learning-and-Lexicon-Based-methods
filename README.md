# Sentiment-Analysis-using-Machine-Learning-and-Lexicon-Based-methods

This project is on twitter sentimental analysis by combining lexicon based and machine learning approaches. I was part of this project with a group of other students as a part of the Data Mining course at North Carolina State University. A supervised lexicon-based approach for extracting sentiments from tweets was implemented. Various supervised machine learning approaches were tested using scikit-learn libraries in python and implemented Decision Trees and Naive Bayes techniques.

The entire code for pre-processing, implementation and post-processing of the project was done in Python 2.7

## Requirements

The packages required for running the code are listed below.

* Sklearn
* Pandas
* Numpy
* Math
* io
* os
* NLTK

## Installations

Most of the packages can be installed using normal pip commands. Installing NLTK may require special instructions which can be found at https://www.nltk.org/install.html

The pre-processing files which are required to run the code are as follows:

tweetylabel.csv #contains the input tweets
dic.csv	#contains the dictionary created and merged
intense.csv. #contains the intensifiers
bucket.csv. #creates the bucket
positive-words.txt #contains the positive word list as text file
negabuse.txt #contains the negative and abusive word list as text file

## Instruction for running the code

Keep all the above mentioned pre-processing files in the same folder and change the directory to that folder. lexi_plus_ml.py file contains the entire code for the project. Open the code and specify the working directory on line 17 of the code.
