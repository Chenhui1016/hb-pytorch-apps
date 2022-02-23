import pandas as pd 
import csv
import os
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
    help='''path to the Manifest File (csv) 
    https://webdata.illumina.com/downloads/productfiles/humanomni25/v1-5/infinium-omni2-5-8v1-5-a1-manifest-file-csv.zip
    '''
)

parser.add_argument(
    'output',
    help='path to save the snp_pos file (csv)'
)

if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output

    df = pd.read_csv(input_path, header=7, usecols=[1,9,10,13]) 

    df.columns = ["ID","CHROM","POS","SRC"]
    
    df = df[["CHROM","POS","ID","SRC"]]

    n = 0

    for chr in df["CHROM"]:
        with open(f"{output_dir}Chr{chr}_chip_snp.csv", mode='wt') as csvfile:
            csvwriter = csv.writer(csvfile)

            if os.stat(f"{output_dir}/Chr{chr}_chip_snp.csv").st_size == 0:
                csvwriter.writerow(["Chrom","Pos","ID","Src"])
            csvwriter.writerow(df.iloc[n])
        n += 1

    print(df.head())

