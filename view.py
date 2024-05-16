"""View msgpack data."""

import argparse, msgpack

parser = argparse.ArgumentParser()
parser.add_argument('path', help='path to result file')
parser.add_argument('-a', '--average', help='show average only', action='store_true')
args = parser.parse_args()
with open(args.path, 'rb') as file:
    result = msgpack.unpackb(file.read())
for key, value  in result.items():
    print(key)
    for algorithm, xss in value.items():
        if args.average:
            print(algorithm, [sum(xs) / len(xs) for xs in xss])
        else:
            print(algorithm, xss)
