from random import shuffle
from tqdm import tqdm
import argparse

def main(args):
    mode = 'w'
    if args.append:
        mode = 'a'
    with open(args.src0_path, 'r') as src0:
        with open(args.src1_path, 'r') as src1:
            with open(args.save_path, mode) as tgt:
                data = []
                zero_count = 0
                print("Reading source 0")
                for line in tqdm(src0):
                    data.append('0\t'+line)
                    zero_count += 1
                one_count = 0
                print("Reading source 1")
                for line in tqdm(src1):
                    data.append('1\t'+line)
                    one_count += 1
                print("Shuffling")
                shuffle(data)
                print("Writing to data file")
                for sample in tqdm(data):
                    tgt.write(sample)
                print("Num w class 0: {}   Num w class 1: {} ".format(zero_count, one_count))


def get_args():
    parser = argparse.ArgumentParser(description='Combine two files of line separated sentences into one file with binary labels distinguishing source file')
    parser.add_argument('--src0_path', type=str, help='path to file with first style')
    parser.add_argument('--src1_path', type=str, help='path to file with second style')
    parser.add_argument('--save_path', type=str, help='path to save file')
    parser.add_argument('--append', action='store_true', default=True, help='if results should be appended (instead of overwriting)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
