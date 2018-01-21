import sys

f1 = sys.argv[1]
f2 = sys.argv[2]

# Exclude line pairs where data are empty on one side
with open(f1) as f1_in, open(f1+'.out', mode='w') as f1_out, open(f2) as f2_in, open(f2+'.out', mode='w') as f2_out:
    for l1, l2 in zip(f1_in.readlines(), f2_in.readlines()):
        if l1.strip() == "" or l2.strip() == "":
            continue
        else:
            print(l1, file=f1_out, end='')
            print(l2, file=f2_out, end='')
