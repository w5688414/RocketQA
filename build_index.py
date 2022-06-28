import sys
import os 
import time
import faiss
import math
import numpy as np
from tqdm import tqdm 


def build_engine(para_emb_list, dim):
    index = faiss.IndexFlatIP(dim)
    # add paragraph embedding
    p_emb_matrix = np.asarray(para_emb_list)
    index.add(p_emb_matrix.astype('float32'))
    return index
    
def build_index():
    file_paths = ['output/part-00.npy','output/part-01.npy','output/part-02.npy','output/part-03.npy']
    hidden_size = 768
    for idx,item in tqdm(enumerate(file_paths)):
        para_embs = np.load(item)
        engine = build_engine(para_embs, hidden_size)
        output_file_name = os.path.join('output','para.index.part{}'.format(idx))
        faiss.write_index(engine, output_file_name)

def main():
    print('building index')
    build_index()

 

if __name__ == "__main__":
    main()

