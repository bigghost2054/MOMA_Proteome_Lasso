'''
Filename: Example.py

Description: A driver program which can run lasso models with different network and baseline and then draw a boxplot to show the prediction performance using Pearson correlation coefficients (PCC)

Authors: Minseung Kim (msgkim@ucdavis.edu)
         ChengEn Tan (cetan@ucdavis.edu)

Changes:
    - 10/04/2018: initial commit
'''

import DataProcessing
import Models
import pandas as pd
import numpy  as np

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

#random baseline
#Sampling 10 profiles (from 18 profiles in this case) (without replacement) 1000 times for training set
baseline_model = General_Model()
baseline_model.load_data(ori_transcriptome_data_path,ori_proteome_data_path,mapping_data_path)
pcc_random, random_predicted, proteome_data = baseline_model.random_baseline()
pcc_mean, mean_predicted, proteome_data = baseline_model.mean_baseline()


import matplotlib.pyplot as plt
import matplotlib.axes as axes
data = [pcc_random.flatten(),pcc_mean[:,0],pcc_ppi[:,0],pcc_cpn[:,0],pcc_pathway[:,0],pcc_tf[:,0],pcc_average[:,0]]
labels = ['Random','Mean','PPI','CPN','Pathway','TRN','Integration']
fig, ax = plt.subplots()
ax.set_title('PCC of proteome expression prediction')
ax.boxplot(data,labels = labels)
ax.set_ylim((-0.5,1.0))
plt.show()

                 
