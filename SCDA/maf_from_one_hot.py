import argparse
import torch
from torch import nn
import pandas as pd


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='''
        Computes the minor allele frequency using the haplotypes. Assumes
        the input file has the same format as the yeast examples provided
        in the original SCDA repo:
        https://github.com/work-hard-play-harder/SCDA
        (i. e., a header, then one row per sample, the first column
        contains the id and the rest of the columns contain the alleles)
    '''
)
parser.add_argument(
    'input', help='path to the dataset (haplotypes)'
)


def compute_mafs(one_hot):
    samples_per_class = torch.sum(one_hot, dim=0) # for each feature (SNP), get the total number of samples in each class (allele)
    sorted, indices = torch.sort(samples_per_class, descending=True) # sort by frequency to identify the minor allele (second most common allele)
    minor_allele_indices = [allele_indices[1] for allele_indices in indices] # extract minor allele index in the one-hot representation for each SNP
    total_samples = one_hot.size(0)
    mafs = [str(float(samples_per_class[snp, minor_allele_index] / total_samples)) for snp, minor_allele_index in enumerate(minor_allele_indices)] # compute relative frequency of the minor allele for each snp
    return mafs


if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.input

    data_frame = pd.read_csv(input_path, sep='\t', index_col=0)
    tensor_data = torch.from_numpy(data_frame.values)
    one_hot = nn.functional.one_hot(tensor_data).float()

    mafs = compute_mafs(one_hot)

    with open(f'{input_path}.mafs', 'w') as out_file:
        snp_ids = data_frame.columns.to_list()
        out_file.write(','.join(snp_ids) + '\n')
        out_file.write(','.join(mafs))
