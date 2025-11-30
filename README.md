# Face_Detection

This is a Regression model trained to recognize faces and predict ages.

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
PIL                 11.3.0  
matplotlib          3.10.0  
numpy               2.0.2  
pandas              2.2.2  
seaborn             0.13.2  
sklearn             1.6.1  
torch               2.9.0+cu126  
torchvision         0.24.0+cu126  
tqdm                4.67.1  
