# MOMA_Proteome_Lasso
Implementation of the proteome model of MOMA
There are three python code files and a dataset directory:
1. Example.py: A driver program which will create Lasso models given four different network datasets and draw a boxplot to show the prediction performance in terms of Perason correlation coefficient (PCC) between predicted values and ground truth.
2. DataProcessing.py: Modules which can merge transcriptome dataset and proteome dataset given a mapping file which can map the entries in protome dataset to transcriptome dataset.
3. Models.py: A class with methods which can load the transcriptome, proteome and network datasets, create LASSO models, and evaluate baselines of prediction performance.

Usage: Please refer Example.py file and run it:
1. Please download the dataset and python code files.
   (NOTE: For transcriptome dataset file "Ecomics.transcriptome.no_avg.v8.txt", please download it using this link:
    https://www.dropbox.com/sh/lqzyd6dzmg1a2c4/AADHqUbNXzKyya_tNQOHN__Wa?dl=0)
2. Put all dataset files and python code files in the same directory (or you can specify the dataset path in Example.py)
3. Run Example.py file
4. After LASSO models are created and proteome expression are predicted, a boxplot which shows the prediction performace (in terms of PCC) will be plotted.
(NOTE: It will take several hours to run the example program.)