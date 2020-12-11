#!/usr/bin/env python3

for filename in ["chinese.dev.conll", "chinese.test.conll", "chinese.train.conll"]:

    depfilename = filename + ".dep"

    with open(filename + ".tab") as f:
        srldata = f.read().strip().split("\n")
    with open(depfilename) as f:
        depdata = f.read().strip().split("\n")

    outfile = filename + ".fused"

    with open(outfile, "w") as fout:
        srl_i = 0
        dep_i = 0
        while srl_i < len(srldata) and dep_i < len(depdata):
            l1 = srldata[srl_i]
            l2 = depdata[dep_i]

            if len(l2) == 0:
                fout.write("\n")
                dep_i += 1
            elif len(l1) == 0:
                srl_i += 1
            else:
                s = l1.split("\t")
                t = l2.split("\t")
                if s[3] != t[1]:
                    srl_i += 1
                else:
                    fout.write(("\t".join(s[:11] + t[6:8] + s[11:])))
                    fout.write("\n")
                    srl_i += 1
                    dep_i += 1
