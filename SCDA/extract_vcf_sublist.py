import argparse
import gzip
from tqdm import tqdm
from math import ceil

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

    input_path = args.input

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
                individuals = line.split()[SAMPLES_START_COLUMN:]
                continue # then skip to the next line
            current_snp_index += 1
            if current_snp_index < args.start_snp:
                continue # skip SNPs before the starting SNP
            fields = line.split()
            chromosome = fields[0]
            position = fields[1]
            ref = fields[3]
            alt = fields[4]
            # store the relevant SNP IDs for the header of the output file
            snps.append(f'{chromosome}:{position}_{ref}_{alt}')
            if current_snp_index == args.end_snp:
                break # exit after the ending SNP is processed

    # Note: this is a plain text file with haps extension,
    # it doesn't conform to a particular haps format
    output_path = f'{input_path}-sub{args.start_snp}-{args.end_snp}.haps'
    print(f'Writing subset haplotypes to {output_path}')
    with open(output_path, 'wt') as out_file:
        out_file.write('SAMID\t' + '\t'.join(snps) + '\n')
        batch_size = 200
        # process in batches and write to the output file right away in order to use less RAM
        for batch_index, batch_samples in tqdm(enumerate(batch(individuals, batch_size)), total=ceil(len(individuals)/batch_size)):
            batch_haplotypes = {}
            for sample in batch_samples:
                batch_haplotypes[first_haplotype_name(sample)] = []
                batch_haplotypes[second_haplotype_name(sample)] = []

            with gzip.open(input_path, 'rt') as inputs:
                current_snp_index = 0
                for line in inputs:
                    if line[0] == '#':
                        continue
                    current_snp_index += 1
                    if current_snp_index < args.start_snp:
                        continue

                    fields = line.split()
                    batch_start = SAMPLES_START_COLUMN + (batch_index * batch_size)
                    batch_end = batch_start + batch_size
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

                    if current_snp_index == args.end_snp:
                        break # exit after the ending SNP is processed
            for sample_id in batch_haplotypes:
                out_file.write(sample_id + '\t' + '\t'.join(batch_haplotypes[sample_id]) + '\n')
