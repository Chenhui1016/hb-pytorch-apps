import argparse
import re
import gzip


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='''
        Extracts the minor allele frequency from the info column of the
        input VCF file and outputs the mafs to a file with the same name
        but with the .mafs suffix.
        The defaults assume 1k genomes phase 3 vcf format and the presence
        of an AF subfield within the info field
    '''
)
parser.add_argument(
    'input', help='path to the dataset (vcf.gz)'
)


if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.input

    results = {}
    print('Extracting mafs...')
    with gzip.open(input_path, 'rt') as inputs:
        for line in inputs:
            if line[0] != '#':
                fields = line.split()
                chromosome = fields[0]
                position = fields[1]
                ref = fields[3]
                alt = fields[4]
                info = fields[7]
                freq = re.search(r'AF=([\d.]+)[;,]', info).group(1)
                results[f'{chromosome}:{position}_{ref}_{alt}'] = freq

    output_path = f'{input_path}.mafs'
    print(f'Writing mafs to {output_path}')
    with open(output_path, 'wt') as out_file:
        for id in results:
            out_file.write(f'{id},{results[id]}\n')
