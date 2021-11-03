import pandas as pd 
import numpy as np 
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='''
        Extract the SNP position information to get the fixed known SNP in a
        common real-life genotype imputation task. (A website to transfer SNP
        ID to SNP position:
        https://www.ncbi.nlm.nih.gov/snp/)
    '''
)

parser.add_argument(
    'input',
    help='path to the Manifest File (csv)'
)

parser.add_argument(
    'output',
    help='path to save the snp_pos file (csv)'
)

if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    df = pd.read_csv(input_path, header=7, usecols=[1,9,10,13]) 

    df.columns = ["ID","CHROM","POS","SRC"]
    
    df = df[["CHROM","POS","ID","SRC"]]
    
    df.index = range(1,len(df)+1)

    print(df.head())
    
    df.to_csv(output_path)

