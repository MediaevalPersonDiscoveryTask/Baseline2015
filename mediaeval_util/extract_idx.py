"""
extract idx file: frame to timestamp mapping

Usage:
  extract_idx.py <pathToffmpeg> <video_file> <output_idx>
  extract_idx.py -h | --help
"""

from docopt import docopt

import subprocess, os, glob

if __name__ == '__main__':
    # read args
    args = docopt(__doc__)

    cmd = args['<pathToffmpeg>']+" -i "+args['<video_file>']+" -y -an -vf showinfo /tmp/tmp.mp4"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    errcode = process.returncode
    os.remove('/tmp/tmp.mp4')

    start = 0.0
    l = []
    for line in err.splitlines():
        if "Duration:" in line and "start:" in line:
            start = float(line.split(' start: ')[1].split(',')[0])
        if "n:" in line and "pts_time:" in line:
            frameID = int(line.split('n:')[1].split(' ')[0])
            timestamp = float(line.split('pts_time:')[1].split(' ')[0])
            typeImage = line.split('type:')[1].split(' ')[0]
            pos = line.split('pos:')[1].split(' ')[0]
            l.append([frameID, typeImage, pos, timestamp+start])

    fout = open(args['<output_idx>'], 'w')
    for frameID, typeImage, pos, timestamp in sorted(l):
        fout.write(' '+str(frameID)+' '+typeImage+' '+pos+' '+str(timestamp)+'\n')
    fout.close()


    