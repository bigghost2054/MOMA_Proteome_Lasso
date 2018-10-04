'''
Filename: DataProcessing.py

Description: A class contains methods which perform lasso model and baselines (average and random baseline)

Authors: Minseung Kim (msgkim@ucdavis.edu)
         ChengEn Tan (cetan@ucdavis.edu)

Changes:
    - 10/04/2018: initial commit
'''

import pandas as pd
import numpy  as np

### CONSTANTS ###
FILE_DELIMETER = "\t"
ID_LABEL = "ID"
OUTPUT_COLUMN_NAME_PREFIX_TRANSCRIPTOME    = "m"       # e.g. m.b0001
PREFIX_DELIMETER_TRANSCRIPTOME             = "."
OUTPUT_COLUMN_NAME_PREFIX_PROTEOME         = "b"
PREFIX_DELIMETER_PROTEOME                  = ""

def normalizeOmics(ori_data_path):
    #input: original data path (in tab separated format) with annotation data
    #output: panda dataframe with normalized feature intensities
    ori_data = pd.read_csv(ori_data_path,sep = FILE_DELIMETER)
    column_names               = ori_data.columns.tolist()
    output_column_names        = list(filter(lambda x: x.startswith(OUTPUT_COLUMN_NAME_PREFIX_TRANSCRIPTOME + PREFIX_DELIMETER_TRANSCRIPTOME), column_names))
    metadata_column_names      = list(set(column_names) - set(output_column_names))
    n_features = len(output_column_names)

    ori_x = ori_data[metadata_column_names]
    ori_y = ori_data[output_column_names].as_matrix()

    #min-max normalize
    normalized_y = np.copy(ori_y) #initialize
    for idx in range(n_features):
        normalized_y[:,idx] = (normalized_y[:,idx] - np.min(normalized_y[:,idx]))/(np.max(normalized_y[:,idx]) - np.min(normalized_y[:,idx]))

    normalized_y = pd.DataFrame(normalized_y,columns = output_column_names)
    normalized_data = pd.concat([ori_x,normalized_y],axis = 1)
    return normalized_data

def merge_transcriptome_and_proteome(transcriptome_data_path, proteome_data_path, mapping_data_path):
    #input:
    #1. transcriptome data path (in tab separated format) with annotation data
    #2. proteome data path (in tab separated format) with annotation data
    #3. mapping data path (in tab separated format without header)
    #output: panda dataframe with normalized transcriptome feature intensities and proteome data path with annotation data (proteome part)
    transcriptome_data = normalizeOmics(transcriptome_data_path)
    proteome_data = pd.read_csv(proteome_data_path, sep = FILE_DELIMETER)
    mapping_data = pd.read_csv(mapping_data_path, sep = FILE_DELIMETER, header = None)

    proteome_id = list(proteome_data[ID_LABEL])
    transcriptome_id = list(transcriptome_data[ID_LABEL])
    mapping_data_proteome_id = list(mapping_data.as_matrix()[:,2])
    mapping_data_transcriptome_id = list(mapping_data.as_matrix()[:,1])

    queried_transcriptome_entry_idx = []
    for idx in range(len(proteome_id)):
        cur_proteome_id = proteome_id[idx]
        mapping_data_entry_idx = mapping_data_proteome_id.index(cur_proteome_id)
        cur_transcriptome_id = mapping_data_transcriptome_id[mapping_data_entry_idx]
        queried_transcriptome_entry_idx.append(transcriptome_id.index(cur_transcriptome_id))

    output_column_names_transcriptome        = list(filter(lambda x: x.startswith(OUTPUT_COLUMN_NAME_PREFIX_TRANSCRIPTOME + PREFIX_DELIMETER_TRANSCRIPTOME), transcriptome_data.columns.tolist()))

    tmp = transcriptome_data.take(queried_transcriptome_entry_idx)
    merged_data = pd.concat([proteome_data,tmp[output_column_names_transcriptome].reset_index(drop=True)],axis = 1)
    return merged_data


#For unit testing only
#ori_transcriptome_data_path = "Ecomics.transcriptome.no_avg.v8.txt"
#ori_proteome_data_path = "proteome.txt"
#mapping_data_path = "pair_list_of_transcriptome_proteome.txt"
#merged_data = merge_transcriptome_and_proteome(ori_transcriptome_data_path,ori_proteome_data_path,mapping_data_path)
#merged_data.to_csv("test.txt",sep = "\t")

