"""
select shot with a duration higher than 2 seconds and in the uem segment

Usage:
  cut_thr_selection.py <video_name> <list_cut> <output_file> <idx_file> [--min_duration=<md>] [--uem=<uem>]
  cut_thr_selection.py -h | --help
Options:
  --min_duration=<md>  minimum duration (in frame) of a shot (>0) [default: 50]  
  --uem=<uem>          uem file  
"""

from docopt import docopt

# read alignement file between frame and timestamp
def sync_idx(f_idx):
    dic_idx = {}    
    for line in open(f_idx):
        frame, type, p, time = line.split()
        dic_idx[int(frame)] = float(time)
        max=int(frame)
    # file gap
    for i in range(max+10):
        time = i    
        if time not in dic_idx:
            while time not in dic_idx:
                time-=1     
                if time<1:
                    time = i
                    break
        if time not in dic_idx:                                    
            while time not in dic_idx:
                time+=1  
        
        if i not in dic_idx:
            dic_idx[i] = dic_idx[time]
    return dic_idx

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    
    # read cut segmentation
    l_cut = []
    for cut in open(args['<list_cut>']).read().splitlines():
        l_cut.append(int(cut))

    min_dur = int(args['--min_duration'])
    dic_idx = sync_idx(args['<idx_file>'])

    # read UEM file
    l_uem = []
    if args['--uem']:
        for line in open(args['--uem']).read().splitlines():
            v, p, start, end = line.split(' ')
            if v == args['<video_name>']:
                l_uem.append([float(start), float(end)])

    # save segmentation with shot higher than min the duration
    fout = open(args['<output_file>'], 'w')
    for i, j in zip(l_cut, l_cut[1:]):
        if j-i > min_dur:
            start_shot = dic_idx[i]
            end_shot = dic_idx[j]

            in_uem = False
            if l_uem == []:
                in_uem = True
            for s_uem, e_uem in l_uem:
                if start_shot>=s_uem and end_shot <=e_uem:
                    in_uem = True
                    break
            if in_uem:
                fout.write(args['<video_name>']+' 1 '+str(start_shot)+' '+str(end_shot-start_shot)+' shot NA NA shot_'+str(i)+'_to_'+str(j)+'\n')
    fout.close()
