#!/usr/bin/env python3

for filename in [
    "english.dev.conll",
    "english.train.conll",
    "english.test.conll",
]:

    depfilename = filename + ".dep"

    with open(filename + ".tab") as f:
        srldata = f.read().strip().split("\n")
    with open(depfilename) as f:
        depdata = f.read().strip().split("\n")

    outfile = filename + ".fused"

    line_no = 0
    with open(outfile, "w") as fout:
        for l1, l2 in zip(srldata, depdata):
            line_no += 1
            if (len(l1) == 0) != (len(l2) == 0):
                print(line_no, "err")
            if len(l1) == 0:
                fout.write("\n")
            else:
                s = l1.split("\t")
                t = l2.split("\t")
                fout.write(("\t".join(s[:11] + t[6:8] + s[11:])))
                fout.write("\n")
