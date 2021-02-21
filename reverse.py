# -*- coding: utf-8 -*-

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str)
    parser.add_argument("outfile", type=str)
    args = parser.parse_args()

    with open(args.infile, 'r') as inf:
        n = 0
        with open(args.outfile, 'w') as outf:
            for line in inf.readlines():
                line = line.strip().split()
                line.reverse()
                outf.write(' '.join(line) + '\n')
                n += 1
                if (n % 100000 == 0):
                    print(n)

if __name__ == "__main__":
    main()
