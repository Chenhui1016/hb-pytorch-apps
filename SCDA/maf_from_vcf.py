import argparse
import re


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
    'input', help='path to the dataset (vcf)'
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
    '--info_column', type=int, default=7,
    help='column within the vcf that contains the info field'
)


if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.input

    results = {}
    print('Extracting mafs...')
    with open(input_path, 'r') as inputs:
        for line in inputs:
            if line[0] != '#':
                fields = line.split()
                snp_id = fields[args.snp_id_column]
                snp_ref = fields[args.ref_column]
                info = fields[args.info_column]
                freq = re.search(r'AF=([\d.]+)[;,]', info).group(1)
                results[f'{snp_ref}{snp_id}'] = freq

    output_path = f'{input_path}.mafs'
    print(f'Writing mafs to {output_path}')
    with open(output_path, 'w') as out_file:
        # out_file.write(','.join(results.keys()) + '\n')
        # out_file.write(','.join(list(results.values())))
        for id in results:
            out_file.write(f'{id},{results[id]}\n')
