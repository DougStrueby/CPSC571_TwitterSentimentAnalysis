# CPSC571 Twitter Sentiment Analysis

## Group Members
- Doug Strueby  
- Nolan Ruzicki  

## General Description
This project analyzes the sentiment behind tweets to identify potential criminal behavior.  

## Running Instructions
1. **Download the project**: Download the ZIP file containing all the project files.  
2. **Unzip the folder**: Extract the contents of the ZIP file.  
3. **Set working directory**: Set the working directory to the folder containing the CSV files using `setwd(FILEPATH)`.  
4. **Open R files**: Open the following R scripts in your R environment:  
   - `DataPreprocessing.R`  
   - `571-Models.R`  
5. **Preprocess data**:  
   - Run `DataPreprocessing.R` first.  
   - Note: This script will override the CSV files in the project folder.  
6. **Run models**: Run the R scripts line by line using `CTRL + ENTER` in the following order:  
   - `DataPreprocessing.R`  
   - `571-Models.R`  
7. **Install required packages**: Run `install.packages` commands as needed. This step is only required once.  
8. **LTSM model requirements**: For the LSTM model, ensure that an up-to-date version of Python is installed. If Python is not detected, the script will prompt you with a function to run.  
