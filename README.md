# Face_Detection

This is a Regression model trained to recognize faces and predict ages.

# Project Goals
Accurately estimate a person’s age from their facial features using two approaches:
1. A regression model that predicts exact age, and...
2. A classification model that predicts age groups.

Analyze how demographic factors (race and gender) influence model accuracy to identify potential biases in facial age estimation.

Evaluate ethical concerns surrounding the use of age-prediction models , including risks in security, surveillance, recommendation systems, and demographic profiling.

Performance goal: achieve ~5-year MAE for regression and ≥70% classification accuracy for age-group prediction.

# Project Setup
- **Notebooks** is where the multiple iterations of our ipynb files are. This holds the code where we cleaned the data and ran the model.
- **pickle_files** is where the data is stored, being training data, test data, and validation data. The files are in pkl format and are imported into our code via the pickle Python library.
- **index.html** and its accompanying folders 'assets' and 'vendor' is the frontend of the project.

# How to Setup
1. Download UTKFaces_updated.ipynb in the Notebook folder.
2. Open the file in Google Colab.
3. Open up the UTKFace folder in Google Drive [here](https://drive.google.com/drive/folders/17smTilKxBXzxsQYiP-0gh-b1l-0pCFzj?usp=sharing) and download part1.tar.gz, part2.tar.gz, and part3.tar.gz.
4. Put the three tar.gz files into data/raw/.
5. Run the cells to train the model.

# Dependencies
- PIL                 11.3.0
- matplotlib          3.10.0
- numpy               2.0.2
- pandas              2.2.2
- seaborn             0.13.2
- sklearn             1.6.1
- torch               2.9.0+cu126
- torchvision         0.24.0+cu126
- tqdm                4.67.1
