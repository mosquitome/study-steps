# study-steps
Automates the analysis of 20-step recordings from Andrés/Albert labs.

### install
SonPy was very particular to install. On Windows, I did the following (where C:\Path\To\Anaconda3\envs\ is the path to my Anaconda3 environments):
```
conda create -p C:\Path\To\Anaconda3\envs\steps python=3.9 numpy pandas scipy scikit-learn quantities pyamg matplotlib seaborn spyder statsmodels
conda activate steps
C:\Path\To\Anaconda3\envs\steps\Scripts\pip3.exe install sonpy
```

### instructions
1. fill out the metadata.txt file with the location of the folders you would like to analyse. Each folder should contain the averaged SMRX files for all steps.
2. Activate the steps environment from the Anaconda terminal: `conda activate steps`.
3. open spyder from within the environment: `spyder`.
4. Run the script.
