
def top_down_traversal(heads, pos=0):
    ret = []
    ret.append(pos)
    for i, h in enumerate(heads):
        if h == pos:
            ret.extend(top_down_traversal(heads, i))
    return ret


def prev_pred(heads, pset, p):
    cur = p
    while cur > 0:
        cur = heads[cur]
        if cur in pset:
            return cur
    return -1


def prev_pred_path(heads, pset, p):
    ret = []
    cur = p
    while cur > 0:
        cur = heads[cur]
        ret.append(cur)
        if cur in pset:
            return ret
    return ret


def subtree_rels(depspans, rels, lm, rm):
    ret = []
    while lm <= rm:
        rs = [r for l, r in depspans if l == lm and r <= rm]
        if len(rs) == 0:
            return "crossing"
        r = max(rs)
        ret.append(rels[depspans[lm, r]])
        lm = r + 1
    return " ".join(ret)


def prefix(lab):
    if len(lab) > 1 and (lab[0] == "R" or lab[0] == "C"):
        return lab[:2]
    else:
        return ""


def suffix(lab):
    if len(lab) > 1 and (lab[0] == "R" or lab[0] == "C"):
        return lab[2:]
    else:
        return lab


def get_deptree_spans(heads):
    length = len(heads)

    lmost = [length for i in range(length)]
    rmost = [0 for i in range(length)]

    for i in range(1, length):
        cur = i
        while cur > 0:
            if i < lmost[cur]:
                lmost[cur] = i
            if i > rmost[cur]:
                rmost[cur] = i
            cur = heads[cur]

    spans = {}
    for i, (l, r) in enumerate(zip(lmost[1:], rmost[1:])):
        if (l, r) in spans:
            continue
        spans[(l, r)] = i + 1
    return spans, lmost, rmost


def least_containing(spanset, l, r):
    min_len = 100000
    ret = None
    for _l, _r in spanset:
        if _l <= l and _r >= r:
            if _r - _l < min_len:
                min_len = _r - _l
                ret = (_l, _r)
    return ret


def srl_forward_v2(sentence):
    heads = sentence.heads
    rels = sentence.rels
    srlrels = []
    depspans, lmost, rmost = get_deptree_spans(heads)
    pset = set()
    for graph in sentence.srl:
        verbid = graph.pred_id
        pset.add(verbid)
        for l, r, lab in graph.spans:
            if (l, r) in depspans:
                srlrels.append((verbid, depspans[(l, r)], lab))
            elif heads[verbid] >= l and heads[verbid] <= r:
                s = least_containing(depspans, l, r)
                srlrels.append((verbid, depspans[s], lab))
            elif (
                heads[verbid] > 0
                and heads[heads[verbid]] >= l
                and heads[heads[verbid]] <= r
            ):
                s = least_containing(depspans, l, r)
                srlrels.append((verbid, depspans[s], lab))
            elif subtree_rels(depspans, rels, l, r) == "dobj xcomp":
                for i in range(len(heads)):
                    if (
                        heads[i] == verbid
                        and rels[i] == "xcomp"
                        and lmost[i] >= l
                        and rmost[i] <= r
                    ):
                        break
                srlrels.append((verbid, i, lab + "-XC"))
            elif subtree_rels(depspans, rels, l, r) == "dobj dep":
                for i in range(len(heads)):
                    if (
                        heads[i] == verbid
                        and rels[i] == "dep"
                        and lmost[i] >= l
                        and rmost[i] <= r
                    ):
                        break
                srlrels.append((verbid, i, lab + "-XC"))

    srlrels_set = {(h, a): r for h, a, r in srlrels}

    for h, a, r in srlrels:
        if h != a and heads[h] == heads[a] and heads[h] not in pset:
            srlrels_set[(heads[h], a)] = srlrels_set[(h, a)]
        if (
            h != a
            and heads[h] == heads[a]
            and heads[h] in pset
            and (heads[h], a) not in srlrels_set
        ):
            srlrels_set[(heads[h], a)] = ""
        if (
            h != a
            and heads[h] != a
            and heads[heads[h]] == heads[a]
            and heads[heads[h]] not in pset
            and heads[h] not in pset
        ):
            srlrels_set[(heads[heads[h]], a)] = srlrels_set[(h, a)]
        if (
            h != a
            and heads[h] != a
            and heads[heads[h]] == heads[a]
            and heads[heads[h]] in pset
            and heads[h] not in pset
            and (heads[heads[h]], a) not in srlrels_set
        ):
            srlrels_set[(heads[heads[h]], a)] = ""
        if h != a and h != heads[a] and prev_pred(heads, pset, a) == h:
            srlrels_set[(heads[a], a)] = r + "-P"

    ret = [None for i in range(len(heads))]
    ret_r = [None for i in range(len(heads))]
    ret_argm = [None for i in range(len(heads))]
    ret_core = [None for i in range(len(heads))]

    for (verb, arg), lab in srlrels_set.items():
        if heads[arg] == verb:
            ret[arg] = lab
        elif heads[verb] == arg:
            ret_r[verb] = lab

    for p in pset:
        rel = rels[p]
        parent_argm = set()
        flag = False
        for h_p in prev_pred_path(heads, pset, p):
            for (h, a), r in srlrels_set.items():
                if (
                    h == h_p
                    and "ARGM" in r
                    and not (rel == "conj" and a > h and a < p)
                    and not (lmost[a] <= p and rmost[a] >= p)
                ):
                    parent_argm.add((a, r))
                    flag = True
            rel = rels[h_p]
            if flag:
                break
        child_argm = {
            (a, r)
            for (h, a), r in srlrels_set.items()
            if h == p and "ARGM" in r and r != "V"
        }

        if len(parent_argm - child_argm) < len(parent_argm.intersection(child_argm)):
            ret_argm[p] = True

    for p in pset:
        for h_p in prev_pred_path(heads, pset, p):
            parent_core = {
                a: r
                for (h, a), r in srlrels_set.items()
                if h == h_p and "ARGM" not in r and r != "V"
            }
            child_core = {
                a: r
                for (h, a), r in srlrels_set.items()
                if h == p and a != heads[h] and "ARGM" not in r and r != "V"
            }

            intersect = list(set(parent_core.keys()).intersection(child_core.keys()))
            if len(intersect) == 0:
                continue

            parent_suffix = [suffix(parent_core[x]) for x in intersect]
            child_suffix = [suffix(child_core[x]) for x in intersect]
            if (
                " ".join([prefix(parent_core[x]) for x in intersect])
                == " ".join([prefix(child_core[x]) for x in intersect])
                and len(set(parent_suffix)) == 1
                and len(set(child_suffix)) == 1
                and len(
                    [
                        a
                        for (h, a), r in srlrels_set.items()
                        if h == h_p and suffix(r) == parent_suffix[0]
                    ]
                )
                == len(parent_suffix)
            ):
                ret_core[p] = (False, parent_suffix[0], child_suffix[0])
            else:
                parent_rel = parent_core[intersect[0]]
                child_rel = child_core[intersect[0]]
                ret_core[p] = (True, parent_rel, child_rel)

            break

    sentence.srels = ret
    sentence.srels_r = ret_r
    sentence.srels_m = ret_argm
    sentence.srels_c = ret_core
    sentence.pset = pset


def conflict_update(covered, h, i, j):
    for k in range(i, j + 1):
        if (h, k) in covered:
            return False
    for k in range(i, j + 1):
        covered.add((h, k))
    return True


def srl_backward_v2(heads, rels, srlrels, srlrels_r, srlrels_argm, srlrels_core, pset):
    depspans, lmost, rmost = get_deptree_spans(heads)
    ret = {}
    covered = set()
    for p in pset:
        covered.add((p, p))

    for i, r in enumerate(srlrels):
        if r is not None:
            if r[-3:] == "-XC":
                if rels[i] in {"xcomp", "dep"}:
                    h = heads[i]
                    lm = lmost[i]
                    rm = rmost[i]
                    jj = i - 1
                    for j in range(i - 1, 0, -1):
                        if heads[j] == h:
                            jj = j
                            break
                    if rels[jj] == "dobj":
                        lm = lmost[jj]
                    if conflict_update(covered, heads[i], lm, rm):
                        ret[(heads[i], lm, rm)] = r[:-3]
                else:
                    if conflict_update(covered, heads[i], lmost[i], rmost[i]):
                        ret[(heads[i], lmost[i], rmost[i])] = r[:-3]
            elif r[-2:] == "-P":
                h = prev_pred(heads, pset, i)
                if h > 0 and conflict_update(covered, h, lmost[i], rmost[i]):
                    ret[(h, lmost[i], rmost[i])] = r[:-2]
            else:
                if conflict_update(covered, heads[i], lmost[i], rmost[i]):
                    ret[(heads[i], lmost[i], rmost[i])] = r

    for i, r in enumerate(srlrels_r):
        if i <= 0:
            continue
        if heads[i] <= 0:
            continue
        if r is not None:
            if r in {"ARGM-LVB", "ARGM-MOD"}:
                if conflict_update(covered, i, heads[i], heads[i]):
                    ret[(i, heads[i], heads[i])] = r
            elif heads[i] < i:
                # | | | * | | x - - -
                lm, rm = None, None
                for j in range(1, i):
                    if j == heads[i]:
                        if lm is None:
                            lm = j
                        rm = j
                    elif (
                        heads[j] == heads[i]
                        and rels[j] not in {"punct", "discourse", "cc"}
                        and srlrels[j] != "ARGM-ADV"
                    ):
                        if lm is None:
                            lm = lmost[j]
                        rm = rmost[j]
                if rm is None:
                    print(i, heads)
                if conflict_update(covered, i, lm, rm):
                    ret[(i, lm, rm)] = r
            else:
                if rels[i] in {"parataxis", "aux", "auxpass"}:
                    rightend = rmost[heads[i]] + 1
                else:
                    rightend = heads[i]
                    for j in range(heads[i] + 1, rmost[heads[i]] + 1):
                        if heads[j] == heads[i]:
                            if rels[j] == "num":
                                rightend = j + 1
                            else:
                                break

                # x | | | * - - -
                lm, rm = heads[i], heads[i]
                for j in range(i + 1, rightend):
                    if (
                        heads[j] == heads[i]
                        and rels[j] not in {"punct", "discourse", "cc", "conj"}
                        and srlrels[j] != "ARGM-ADV"
                    ):
                        lm = min(lm, lmost[j])
                        rm = max(rm, rmost[j])
                if conflict_update(covered, i, lm, rm):
                    ret[(i, lm, rm)] = r

    for i in top_down_traversal(heads):
        t = srlrels_argm[i]
        if i <= 0:
            continue
        if t:
            rel = rels[i]
            for h_p in prev_pred_path(heads, pset, i):
                argms = [
                    (lm, rm, ret[(h, lm, rm)])
                    for (h, lm, rm) in ret
                    if h == h_p
                    and "ARGM" in ret[(h, lm, rm)]
                    and not (rel == "conj" and lm > h and rm < i)
                    and not (lm <= i and rm >= i)
                ]
                for lm, rm, r in argms:
                    if conflict_update(covered, i, lm, rm):
                        ret[(i, lm, rm)] = r
                rel = rels[h_p]
                if len(argms) > 0:
                    break

    for i in top_down_traversal(heads):
        t = srlrels_core[i]
        if i <= 0:
            continue

        if t and t[0]:
            # single substitute
            for h_p in prev_pred_path(heads, pset, i):
                rel_dict = {
                    ret[(h, lm, rm)]: (h, lm, rm) for (h, lm, rm) in ret if h == h_p
                }
                if t[1] in rel_dict:
                    h_, l, r = rel_dict[t[1]]
                    if conflict_update(covered, i, l, r):
                        ret[(i, l, r)] = t[2]
                        flag = True
                    break
        elif t and not t[0]:
            # multiple substitute
            for h_p in prev_pred_path(heads, pset, i):
                flag = False
                for (h, lm, rm) in ret:
                    if h == h_p and suffix(ret[(h, lm, rm)]) == t[1]:
                        flag = True
                        break
                if flag:
                    for (h, lm, rm), l in list(ret.items()):
                        if h == h_p and suffix(l) == t[1]:
                            if conflict_update(covered, i, lm, rm):
                                ret[(i, lm, rm)] = prefix(l) + t[2]
                    break

    ret = [tuple(list(x) + [ret[x]]) for x in ret if x[0] in pset and ret[x] != ""]

    return ret
