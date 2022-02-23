import argparse
import gzip
import pandas as pd
from tqdm import tqdm
from math import ceil

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='''
        Extracts Illumina OMNI SNPs
    '''
)
parser.add_argument(
    'vcf_input',
    help='path to the dataset (vcf.gz)'
)
parser.add_argument(
    'fix_pos',
    help='Chr_chip_snp.csv'
)

parser.add_argument(
    'mafs_path',
    help='path to the mafs extracted with the scripts in this dir (maf_from_xxx)'
)

SAMPLES_START_COLUMN = 9

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def split_haplotypes(genotype):
    # add 1 to the vcf variant in order to reserve 0 for the missing data
    return [str(int(haplotype) + 1) for haplotype in genotype.split('|')]

def first_haplotype_name(sample):
    return f'{sample}-1'

def second_haplotype_name(sample):
    return f'{sample}-2'

if __name__ == '__main__':
    args = parser.parse_args()

    vcf_path = args.vcf_input
    fix_pos = args.fix_pos

    pos_data = pd.read_csv(fix_pos)
    pos_data.sort_values(['Pos'], axis=0, inplace=True)
    mafs = pd.read_csv(args.mafs_path, header=None, sep='\t', index_col=0)


    # Todo: avoid append in python
    snps = []
    valid_snp_index = []

    with gzip.open(vcf_path, 'rt') as inputs:
        header_index = 0
        for line in inputs:
            if line[0:2] == '##':
                header_index += 1
                continue
            elif line[0] == '#':
                individuals = line.split()[SAMPLES_START_COLUMN:]
                continue
            fields = line.split()
            chromosome = fields[0]
            position = fields[1]
            ref = fields[3]
            alt = fields[4]
            # store the relevant SNP IDs for the header of the output file
            snps.append(f'{chromosome}:{position}_{ref}_{alt}')
    vcf_data = pd.read_csv(vcf_path, compression='gzip', sep='\t', usecols=range(SAMPLES_START_COLUMN), skiprows=range(header_index))

    print(vcf_data[0:5])
    current_snp_index = 0

    non_exist_snp = 0
    for index, line in pos_data.iterrows():
        while vcf_data.iloc[current_snp_index][1] < line['Pos']:
            current_snp_index += 1
        
        position = vcf_data.iloc[current_snp_index][1]
        
        if position == line['Pos']:
            valid_snp_index.append(current_snp_index)
            current_snp_index += 1
        else:
            non_exist_snp += 1
    
    print(f"{non_exist_snp} snps do not exist in the vcf file")
    
    output_path = f'{vcf_path}_fix_snps.haps'
    print(f'Writing fix snps to {output_path}')
    with open(output_path, 'wt') as out_file:
        out_file.write('SAMID\t' + '\t'.join(snps) + '\n')
        batch_size = 200
        # process in batches and write to the output file right away in order to use less RAM
        for batch_index, batch_samples in tqdm(enumerate(batch(individuals, batch_size)), total=ceil(len(individuals)/batch_size)):
            batch_haplotypes = {}
            for sample in batch_samples:
                batch_haplotypes[first_haplotype_name(sample)] = []
                batch_haplotypes[second_haplotype_name(sample)] = []
            
            with gzip.open(vcf_path, 'rt') as inputs:
                vcf_line = 0
                valid_index = 0
                num_valid_snp = 0

                batch_start = SAMPLES_START_COLUMN + (batch_index * batch_size)
                batch_end = batch_start + batch_size
                for line in inputs:
                    if line[0] == '#':
                        continue
                    if mafs[1][snps[vcf_line]] >= 0.01:
                        num_valid_snp += 1
                        if vcf_line == valid_snp_index[valid_index] and valid_index < len(valid_snp_index):
                            fields = line.split()
                            variant_genotypes = fields[batch_start:batch_end]

                            # separate the two haplotypes for each sample genotype
                            variant_haps = [split_haplotypes(g) for g in variant_genotypes]

                            for in_batch_index, sample in enumerate(batch_samples):
                                batch_haplotypes[first_haplotype_name(sample)].append(
                                    variant_haps[in_batch_index][0]
                                )
                                batch_haplotypes[second_haplotype_name(sample)].append(
                                    variant_haps[in_batch_index][1]
                                )

                            valid_index += 1
                        else:
                            for in_batch_index, sample in enumerate(batch_samples):
                                batch_haplotypes[first_haplotype_name(sample)].append('0')
                                batch_haplotypes[second_haplotype_name(sample)].append('0')

                    vcf_line += 1

            for sample_id in batch_haplotypes:
                out_file.write(sample_id + '\t' + '\t'.join(batch_haplotypes[sample_id]) + '\n')

    print(f'{num_valid_snp} snps have mafs > 1%.')
                

