"""
Find facial landmark on face tracks

Usage:
  extract_flandmark.py <video> <face_tracking> <bin_flandmark> <output_file>
  extract_flandmark.py -h | --help
"""

import os
from docopt import docopt

if __name__ == '__main__':
    # read args
    args = docopt(__doc__)
    # Run extract facial landmark
    os.popen(args['<bin_flandmark>']+" "+args['<video>']+" "+args['<face_tracking>']+" "+args['<output_file>'])
