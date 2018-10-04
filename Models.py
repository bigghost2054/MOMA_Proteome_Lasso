'''
Filename: Models.py

Description: A class contains methods which perform lasso model and baselines (average and random baseline)

Authors: Minseung Kim (msgkim@ucdavis.edu)
         ChengEn Tan (cetan@ucdavis.edu)

Changes:
    - 10/04/2018: initial commit
'''

import DataProcessing
import pandas as pd
import numpy  as np

from sklearn.linear_model import LassoCV

class General_Model:
    # A class contains methods which perform lasso model and baselines
    # Actually different lasso model are based on different network data only, therefore only one class has to be implemented.
    # You can create different model by loading different network dataset using load_network_data() function

    def __init__(self):
        self.network_data = None
        self.merged_data = None

    def load_network_data(self, network_data_path):
        # loading network data (in this case, network data could be cpn, ppi, KEGG pathway and TRN(TF))
        # The network dataset will contain two columns of genes, and the first column of genes will regulate the second column of genes(protein)
        # Input: The filepath of the network data
        # Output: None (the network data will be loaded)
        self.network_data = pd.read_csv(network_data_path, sep = DataProcessing.FILE_DELIMETER, header = None)

    def load_data(self, transcriptome_data_path, proteome_data_path, mapping_data_path):
        # loading transcriptome and proteome data and then merge together to prepare the dataset of the model
        # Input: 
        #       1. The filepath of the transcriptome data
        #       2. The filepath of the proteome data
        #       3. The filepath of the mapping data which can map the transcriptome data entries to proteome data
        # Output: None (the data will be loaded, normalized and merged)
        self.merged_data = DataProcessing.merge_transcriptome_and_proteome(transcriptome_data_path, proteome_data_path, mapping_data_path)

    def lasso(self):
        # Creating a LASSO model given a merged dataset (provided by load_data() function) and network dataset (provided by load_network_data() function)
        # A LASSO model will be trained for each protein for each profiles.
        # Input: None (You should prepare the dataset first by using load_data() and load_network_data())
        # Output:
        #       1. PCC: Pearson correlation coefficients between predicted proteome expression profiles and truth values
        #       2. proteome_data_predicted: Predicted proteome expression profiles (N*M matrix, which N is the #profiles and M is the #proteins)
        #       3. proteome_data: True proteome expression profiles (N*M matrix, which N is the #profiles and M is the #proteins)
        expression_column_names_proteome        = list(filter(lambda x: x.startswith(DataProcessing.OUTPUT_COLUMN_NAME_PREFIX_PROTEOME + DataProcessing.PREFIX_DELIMETER_PROTEOME), self.merged_data.columns.tolist()))
        expression_column_names_transcriptome   = list(filter(lambda x: x.startswith(DataProcessing.OUTPUT_COLUMN_NAME_PREFIX_TRANSCRIPTOME + DataProcessing.PREFIX_DELIMETER_TRANSCRIPTOME), self.merged_data.columns.tolist()))
        transcriptome_data = self.merged_data[expression_column_names_transcriptome].as_matrix()
        proteome_data = self.merged_data[expression_column_names_proteome].as_matrix()

        proteome_data_predicted = np.zeros(proteome_data.shape)

        related_genes_idx_list = [] #Find the related genes for each protein. This is a time-comsuming work but independent to the conditions therefore we should do outside
        for j in range(len(expression_column_names_proteome)): #for each of 1001 proteins
            related_genes_idx = self.query_related_genes_from_protein(expression_column_names_proteome[j], expression_column_names_transcriptome)
            related_genes_idx_list.append(related_genes_idx)
            if j % 20 == 0:
                print('Related genes searching: ' + str(j) + ' proteins\n')

        for i in range(transcriptome_data.shape[0]): #for each proteome profiles
            train_transcriptome_all = np.delete(transcriptome_data, i, 0)
            train_proteome_all = np.delete(proteome_data, i, 0)
            test_transcriptome_all = transcriptome_data[i,:]
            test_proteome_all = proteome_data[i,:]
            for j in range(len(expression_column_names_proteome)): 
                #for each proteins A, obtain the related genes that can regulate A
                train_y = train_proteome_all[:,j]
                test_y = test_proteome_all[j]
                related_genes_idx = related_genes_idx_list[j]
                if len(related_genes_idx) > 0:
                    #If genes that can regulate A exist, train a LASSO model to predict A expression given gene expressions of the found related genes
                    train_x = train_transcriptome_all[:,related_genes_idx]
                    test_x = test_transcriptome_all[related_genes_idx]

                    #Lasso regression
                    regr = LassoCV(n_jobs = -2, tol = 0.001, cv = train_x.shape[0]) #l1_ratio = 0 if ridge regression, default n_fold = 3
                    regr.fit(train_x, train_y)
                    #Predict the expression of protein A
                    predicted_result = regr.predict(test_x.reshape(1,-1))
                else:
                    #If genes that can regulate A do not exist, skip this protein and mark the predicted expression of A as NaN
                    predicted_result = np.nan

                if j % 20 == 0:
                    print(str(j) + ' proteins are predicted')
                proteome_data_predicted[i,j] = predicted_result

        proteome_data_predicted = pd.DataFrame(proteome_data_predicted)
        proteome_data = pd.DataFrame(proteome_data) #Convert it back to pd dataframe (for calculating PCC with nan values)
        PCC = np.zeros((proteome_data.shape[0],1))
        for idx in range(len(PCC)):
            Tmp = pd.concat([proteome_data.take([idx]),proteome_data_predicted.take([idx])],axis = 0).transpose()
            PCC[idx] = Tmp.corr().as_matrix()[0,1]

        return PCC, proteome_data_predicted, proteome_data

    def random_baseline(self, random_set_size = 10, sample_times = 1000):
        # Finding a random baseline
        # For each profile, n (= 10 in default) profiles in the training set will be chosen randomly and the average expressions will be evaluated
        # This evaluation will be performed m times (= 1000 in default). Therefore, for each profile, m prediction results will be generated
        # In example.py, #profiles * m PCC values will be reshaped as a vector for boxplot
        # Input: 
        #       1. random_set_size (n)
        #       2. samples_times (m)
        # Output:
        #       1. PCC (Pearson correlation coefficients between predicted proteome expression profiles and truth values) (a matrix with size #profiles * m)
        #       2. proteome_data_predicted: Predicted proteome expression profiles (a 3D matrix with size #n_profiles * #n_proteins * m)
        #       3. proteome_data: True proteome expression profiles (a 2D matrix with size #n_profiles * #n_proteins)
        expression_column_names_proteome        = list(filter(lambda x: x.startswith(DataProcessing.OUTPUT_COLUMN_NAME_PREFIX_PROTEOME + DataProcessing.PREFIX_DELIMETER_PROTEOME), self.merged_data.columns.tolist()))
        proteome_data = self.merged_data[expression_column_names_proteome].as_matrix()

        PCC = np.zeros((proteome_data.shape[0],sample_times))
        proteome_data_predicted = np.zeros((proteome_data.shape[0],proteome_data.shape[1],sample_times))

        for i in range(proteome_data.shape[0]): #for each A of 18 proteome profiles
            train_proteome = np.delete(proteome_data, i, 0)
            test_proteome = proteome_data[i,:]

            for j in range(sample_times):
                #sampling random set
                sampling_idx = np.random.choice(train_proteome.shape[0],random_set_size)
                sampling_matrix = train_proteome[sampling_idx,:]
                proteome_data_predicted[i,:,j] = np.mean(sampling_matrix,axis = 0)
                
                PCC[i,j] = np.corrcoef(proteome_data_predicted[i,:,j],test_proteome)[0,1]

        return PCC, proteome_data_predicted, proteome_data

    def mean_baseline(self):
        # Finding a average baseline
        # For each profile, the average expressions of all profiles in the training set will be evaluated
        # Input: None (You should prepare the dataset first by using load_data() and load_network_data())
        # Output:
        #       1. PCC (Pearson correlation coefficients between predicted proteome expression profiles and truth values) (a matrix with size #profiles * 1)
        #       2. proteome_data_predicted: Predicted proteome expression profiles (a 2D matrix with size #n_profiles * #n_proteins)
        #       3. proteome_data: True proteome expression profiles (a 2D matrix with size #n_profiles * #n_proteins)
        expression_column_names_proteome        = list(filter(lambda x: x.startswith(DataProcessing.OUTPUT_COLUMN_NAME_PREFIX_PROTEOME + DataProcessing.PREFIX_DELIMETER_PROTEOME), self.merged_data.columns.tolist()))
        proteome_data = self.merged_data[expression_column_names_proteome].as_matrix()

        PCC = np.zeros((proteome_data.shape[0],1))
        proteome_data_predicted = np.zeros((proteome_data.shape))

        for i in range(proteome_data.shape[0]): #for each A of 18 proteome profiles
            train_proteome = np.delete(proteome_data, i, 0)
            test_proteome = proteome_data[i,:]
            proteome_data_predicted[i,:] = np.mean(train_proteome,axis = 0)

        proteome_data_predicted = pd.DataFrame(proteome_data_predicted)
        proteome_data = pd.DataFrame(proteome_data) #Convert it back to pd dataframe (for calculating PCC with nan values)
        PCC = np.zeros((proteome_data.shape[0],1))
        for idx in range(len(PCC)):
            Tmp = pd.concat([proteome_data.take([idx]),proteome_data_predicted.take([idx])],axis = 0).transpose()
            PCC[idx] = Tmp.corr().as_matrix()[0,1]
        return PCC, proteome_data_predicted, proteome_data


    def query_related_genes_from_protein(self, protein, expression_column_names_transcriptome):
        #Given a protein A, find the related genes that can regulate A in the network dataset
        #After genes are found, report the gene indices in transcriptome dataset
        #Input:
        #       1. Protein A
        #       2. Gene names in transcriptome dataset
        #Output:
        #       1. Gene indices in transcriptome dataset
        genes_in_network_data = self.network_data[0]
        proteins_in_network_data = self.network_data[1]
        hit_genes = []
        hit_transcriptome_data_entry_idx = []
        occurrences = lambda s, lst: (i for i,e in enumerate(lst) if e == s)
        hit_gene_index = list(occurrences(protein, proteins_in_network_data))
        for i in range(len(hit_gene_index)):
            hit_genes.append(genes_in_network_data[hit_gene_index[i]])
        for idx in range(len(hit_genes)):
            #NOTE: some genes in the network is not included in transcriptome data
            if (DataProcessing.OUTPUT_COLUMN_NAME_PREFIX_TRANSCRIPTOME + DataProcessing.PREFIX_DELIMETER_TRANSCRIPTOME + hit_genes[idx]) in expression_column_names_transcriptome:
                hit_transcriptome_data_entry_idx.append(expression_column_names_transcriptome.index(DataProcessing.OUTPUT_COLUMN_NAME_PREFIX_TRANSCRIPTOME + DataProcessing.PREFIX_DELIMETER_TRANSCRIPTOME + hit_genes[idx]))

        return hit_transcriptome_data_entry_idx
