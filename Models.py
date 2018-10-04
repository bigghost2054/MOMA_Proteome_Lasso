'''
Filename: filename.py

Description:

Authors: name and contact info

Copyright: copyright info if any

Changes:
    - xx/xx/2018: initial commit
'''

import DataProcessing
import pandas as pd
import numpy  as np
import glmnet_python

from sklearn.linear_model import LassoCV

class General_Model:
    # need class comment section here
    # description of the class

    def __init__(self):
        self.network_data = None
        self.merged_data = None

    def load_network_data(self, network_data_path):
        # function comments
        self.network_data = pd.read_csv(network_data_path, sep = DataProcessing.FILE_DELIMETER, header = None)

    def load_data(self, transcriptome_data_path, proteome_data_path, mapping_data_path):
        # function comments
        self.merged_data = DataProcessing.merge_transcriptome_and_proteome(transcriptome_data_path, proteome_data_path, mapping_data_path)

    def lasso(self):
        # function comments
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

        for i in range(transcriptome_data.shape[0]): #for each A of 18 proteome profiles
            train_transcriptome_all = np.delete(transcriptome_data, i, 0)
            train_proteome_all = np.delete(proteome_data, i, 0)
            test_transcriptome_all = transcriptome_data[i,:]
            test_proteome_all = proteome_data[i,:]
            for j in range(len(expression_column_names_proteome)): #for each of 1001 proteins
                train_y = train_proteome_all[:,j]
                test_y = test_proteome_all[j]
                related_genes_idx = related_genes_idx_list[j]
                if len(related_genes_idx) > 0:
                    train_x = train_transcriptome_all[:,related_genes_idx]
                    test_x = test_transcriptome_all[related_genes_idx]

                    #Lasso regression
                    regr = LassoCV(n_jobs = -2, tol = 0.001, cv = train_x.shape[0]) #l1_ratio = 0 if ridge regression, default n_fold = 3
                    regr.fit(train_x, train_y)
                    #Predict
                    predicted_result = regr.predict(test_x.reshape(1,-1))
                else:
                    predicted_result = np.nan

                #print(predicted_result)
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
        # function comments
        expression_column_names_proteome        = list(filter(lambda x: x.startswith(DataProcessing.OUTPUT_COLUMN_NAME_PREFIX_PROTEOME + DataProcessing.PREFIX_DELIMETER_PROTEOME), self.merged_data.columns.tolist()))
        proteome_data = self.merged_data[expression_column_names_proteome].as_matrix()

        PCC = np.zeros((proteome_data.shape[0],1))
        proteome_data_predicted = np.zeros((proteome_data.shape))

        for i in range(proteome_data.shape[0]): #for each A of 18 proteome profiles
            train_proteome = np.delete(proteome_data, i, 0)
            test_proteome = proteome_data[i,:]

            predicted_result = np.zeros((sample_times,proteome_data.shape[1]))
            for j in range(sample_times):
                #sampling random set
                sampling_idx = np.random.choice(train_proteome.shape[0],random_set_size)
                sampling_matrix = train_proteome[sampling_idx,:]
                predicted_result[j,:] = np.mean(sampling_matrix,axis = 0)

            proteome_data_predicted[i,:] = np.mean(predicted_result,axis = 0)

        proteome_data_predicted = pd.DataFrame(proteome_data_predicted)
        proteome_data = pd.DataFrame(proteome_data) #Convert it back to pd dataframe (for calculating PCC with nan values)
        PCC = np.zeros((proteome_data.shape[0],1))
        for idx in range(len(PCC)):
            Tmp = pd.concat([proteome_data.take([idx]),proteome_data_predicted.take([idx])],axis = 0).transpose()
            PCC[idx] = Tmp.corr().as_matrix()[0,1]
        return PCC, proteome_data_predicted, proteome_data

    def mean_baseline(self):
        # function comments
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
        # function comments
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


# do we need this part for delivery?
# and if so, I think we should comment this little better

#For testing
ori_transcriptome_data_path = "Ecomics.transcriptome.no_avg.v8.txt"
ori_proteome_data_path = "proteome.txt"
mapping_data_path = "pair_list_of_transcriptome_proteome.txt"

ppi_data_path = 'network.ppi.txt'
cpn_data_path = 'network.cpn.txt'
pathway_data_path = 'network.pathway.txt'
tf_data_path = 'network.tf.txt'


ppi_model = General_Model()
ppi_model.load_network_data(ppi_data_path)
ppi_model.load_data(ori_transcriptome_data_path,ori_proteome_data_path,mapping_data_path)
pcc_ppi, ppi_predicted, proteome_data = ppi_model.lasso()

cpn_model = General_Model()
cpn_model.load_network_data(cpn_data_path)
cpn_model.load_data(ori_transcriptome_data_path,ori_proteome_data_path,mapping_data_path)
pcc_cpn, cpn_predicted, proteome_data = cpn_model.lasso()


pathway_model = General_Model()
pathway_model.load_network_data(pathway_data_path)
pathway_model.load_data(ori_transcriptome_data_path,ori_proteome_data_path,mapping_data_path)
pcc_pathway, pathway_predicted, proteome_data = pathway_model.lasso()


tf_model = General_Model()
tf_model.load_network_data(tf_data_path)
tf_model.load_data(ori_transcriptome_data_path,ori_proteome_data_path,mapping_data_path)
pcc_tf, tf_predicted, proteome_data = tf_model.lasso()

#consensus
average_predicted = np.nanmean(np.dstack((ppi_predicted,cpn_predicted,pathway_predicted,tf_predicted)),axis=2)
average_predicted = pd.DataFrame(average_predicted)
pcc_average = np.zeros((proteome_data.shape[0],1))
for idx in range(len(pcc_average)):
    Tmp = pd.concat([proteome_data.take([idx]),average_predicted.take([idx])],axis = 0).transpose()
    pcc_average[idx] = Tmp.corr().as_matrix()[0,1]

print(pcc_ppi[:,0])
print(pcc_cpn[:,0])
print(pcc_pathway[:,0])
print(pcc_tf[:,0])
print(pcc_average[:,0])

#random baseline
#Sampling 10 profiles (from 18 profiles in this case) (without replacement) 1000 times for training set
baseline_model = General_Model()
baseline_model.load_data(ori_transcriptome_data_path,ori_proteome_data_path,mapping_data_path)
pcc_random, random_predicted, proteome_data = baseline_model.random_baseline()
pcc_mean, mean_predicted, proteome_data = baseline_model.mean_baseline()


import matplotlib.pyplot as plt
import matplotlib.axes as axes
data = [pcc_random[:,0],pcc_mean[:,0],pcc_ppi[:,0],pcc_cpn[:,0],pcc_pathway[:,0],pcc_tf[:,0],pcc_average[:,0]]
labels = ['Random','Mean','PPI','CPN','Pathway','TRN','Integration']
fig, ax = plt.subplots()
ax.set_title('PCC of proteome expression prediction')
ax.boxplot(data,labels = labels)
ax.set_ylim((-0.5,1.0))
plt.show()

import pickle
with open('LassoResult.pkl','wb') as f:
    pickle.dump([pcc_random, pcc_mean, pcc_ppi, pcc_cpn, pcc_pathway, pcc_tf, pcc_average,
                 ppi_predicted, cpn_predicted, pathway_predicted, tf_predicted, average_predicted,
                 proteome_data], f)