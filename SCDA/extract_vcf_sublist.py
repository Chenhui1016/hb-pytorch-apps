import argparse
import re
import gzip


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='''
        Extracts a sublist of the SNPs in a VCF file and outputs the sublist
        in the haplotype format used as input by the SCDA model (see example
        datasets in the original SCDA repo:
        https://github.com/work-hard-play-harder/SCDA)
    '''
)
parser.add_argument(
    'input',
    help='path to the dataset (vcf.gz)'
)
parser.add_argument(
    'start_snp', type=int,
    help='starting position of the extracted sublist (>= 1)'
)
parser.add_argument(
    'end_snp', type=int,
    help='ending position of the extracted sublist (inclusive)'
)
parser.add_argument(
    '--snp_id_column', type=int, default=1, # 2 in version a, 1 in version b
    help='column within the vcf that contains the snp id'
)
parser.add_argument(
    '--ref_column', type=int, default=3,
    help='column within the snp ref'
)
parser.add_argument(
    '--samples_start_column', type=int, default=9,
    help='column within the vcf from which the sample data starts'
)


def split_haplotypes(genotype):
    # add 1 to the vcf variant in order to reserve 0 for the missing data
    return [str(int(haplotype) + 1) for haplotype in genotype.split('|')]

def first_haplotype_name(sample):
    return f'{sample}-1'

def second_haplotype_name(sample):
    return f'{sample}-2'

if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.input

    haplotypes = {}
    individuals = []
    snps = []

    print(f'Parsing SNPs {args.start_snp} to {args.end_snp}...')
    with gzip.open(input_path, 'rt') as inputs:
        current_snp_index = 0
        for line in inputs:
            if line[0:2] == '##':
                continue # skip metadata lines
            if line[0] == '#':
                # take the sample IDs
                individuals = line.split()[args.samples_start_column:]
                # create empty list for each haplotype of each sample
                for sample in individuals:
                    haplotypes[first_haplotype_name(sample)] = []
                    haplotypes[second_haplotype_name(sample)] = []
                continue # then skip to the next line
            current_snp_index += 1
            if current_snp_index < args.start_snp:
                continue # skip SNPs before the starting SNP
            fields = line.split()
            snps.append(f'{fields[args.ref_column]}{fields[args.snp_id_column]}')
            variant_genotypes = fields[args.samples_start_column:]
            # separate the two haplotypes for each sample genotype
            variant_haps = [split_haplotypes(g) for g in variant_genotypes]
            for index, sample in enumerate(individuals):
                haplotypes[first_haplotype_name(sample)].append(
                    variant_haps[index][0]
                )
                haplotypes[second_haplotype_name(sample)].append(
                    variant_haps[index][1]
                )
            if current_snp_index == args.end_snp:
                break # exit after the ending SNP is processed

    # Note: this is a plain text file with haps extension,
    # it doesn't conform to a particular haps format
    output_path = f'{input_path}-sub{args.start_snp}-{args.end_snp}.haps'
    print(f'Writing sublist to {output_path}')
    with open(output_path, 'wt') as out_file:
        out_file.write('SAMID\t' + '\t'.join(snps) + '\n')
        for sample_id in haplotypes:
            out_file.write(sample_id + '\t' + '\t'.join(haplotypes[sample_id]) + '\n')
