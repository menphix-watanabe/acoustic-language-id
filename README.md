# acoustic-language-id
Spoken Language Detection using the Kaggle Spoken Language Identification data set. 

This work is for the final project of the Machine Learning (CSCI E-89 (16392)) course at Harvard Extension School. 

Course link: https://canvas.harvard.edu/courses/79842/assignments/syllabus

### Steps to run:
#### Pre-processing the Data
1. Download the “train” and “test” data sets from  https://www.kaggle.com/toponowicz/spoken-language-identification. This will result in two files: “train.zip” and “test.zip”.
2. Unzip the two files into “train” and “test” folders
3. In the “test” folder, you should see three sub-folders: “de”, “en”, and “es”. Move all the files under the sub-folders to the “test” folder, and delete the empty sub-folders. This step ensures that the train and test sets have the same directory structure. 

#### Checking out the code
1. git clone git@github.com:menphix-watanabe/acoustic-language-id.git
2. cd acoustic-language-id

#### Train the model
```
python run-exp.py 
	--train-dir <train dir>
	--test-dir <test dir>
	--save-model-dir <direction to save model file>
```
  
By default it’s going to train the CNN model, but feel free to change the code to call build the RNN model. 

#### Testing the model
```
python run-exp.py 
	--train-dir <train dir>
	--test-dir <test dir>
	--load-model <model file>
```
This is going to output the final accuracy, which is 94.07%. 
