import argparse
import re
import gzip


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='''
        Given start and end positions within a chromosome,
        find the indexes of the variants that enclose that region
    '''
)
parser.add_argument(
    'input',
    help='path to the dataset (vcf.gz)'
)
parser.add_argument(
    'start_pos', type=int,
    help='starting position of the region (>= 1)'
)
parser.add_argument(
    'end_pos', type=int,
    help='ending position of the region (inclusive)'
)
parser.add_argument(
    '--snp_pos_column', type=int, default=1,
    help='column within the vcf that contains the snp position'
)

if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.input
    start = args.start_pos
    end = args.end_pos

    print(f'Finding indexes...')
    with gzip.open(input_path, 'rt') as inputs:
        current_snp_index = 0
        within_region = False
        last_pos_within_region = None
        for line in inputs:
            if line[0] == '#':
                continue # skip metadata and header lines
            current_snp_index += 1
            fields = line.split()
            pos = int(fields[args.snp_pos_column])
            if (pos < start):
                continue
            if (not within_region and pos >= start):
                within_region = True
                print(f'Start index: {current_snp_index} (pos {pos})')
            if (within_region):
                if (pos <= end):
                    last_pos_within_region = pos
                else:
                    current_snp_index -= 1
                    break
        if (within_region):
            print(f'End index: {current_snp_index} (pos {last_pos_within_region})')
        else:
            print('Region not found in input file')
