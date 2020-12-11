#!/usr/bin/env python

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from cpython cimport bool

cdef np.float64_t NEGINF = -np.inf
cdef np.float64_t INF = np.inf
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
def span_decoding(np.ndarray[np.npy_intp, ndim=1] word_lengths,
                  np.ndarray[np.float64_t, ndim=3] label_scores,
                  np.ndarray[np.float64_t, ndim=4] left_scores,
                  np.ndarray[np.float64_t, ndim=4] right_scores,
                  int o_label):
    cdef int batch_size, seq_len, label_size
    cdef int batch_i, length, i, j, l, k, cur_pos, back_i, back_l
    cdef np.float64_t cand, max_cand
    cdef np.ndarray[np.npy_intp, ndim=2] ret_label
    cdef np.ndarray[np.npy_intp, ndim=2] ret_bio
    cdef np.ndarray[np.npy_intp, ndim=2] dp_back
    cdef np.ndarray[np.float64_t, ndim=3] span_scores
    cdef np.ndarray[np.float64_t, ndim=1] dp_scores

    batch_size, seq_len, label_size = np.shape(label_scores)
    ret_label = np.zeros((batch_size, seq_len), dtype=int)
    ret_bio = np.zeros((batch_size, seq_len), dtype=int)
    back_i = 0
    back_l = 0

    for batch_i in range(batch_size):
        length = word_lengths[batch_i]

        # finding all the span scores
        span_scores = np.zeros((seq_len, seq_len, label_size))

        for i in range(length):
            for j in range(i, length):
                for l in range(1, label_size):
                    if l == o_label:
                        for k in range(i, j + 1):
                            span_scores[i, j, l] += label_scores[batch_i, k, l]
                    else:
                        for k in range(i, j + 1):
                            span_scores[i, j, l] += left_scores[batch_i, l, k, i] + right_scores[batch_i, l, k, j] + label_scores[batch_i, k, l]

        # dynamic programming for finding the max
        dp_scores = np.zeros((length + 1, )) - INF
        dp_scores[0] = 0.
        dp_back = np.zeros((length, 2), dtype=int)

        #  print(dp_scores[-1])
        for j in range(length):
            max_cand = NEGINF

            for i in range(j + 1):
                for l in range(1, label_size):
                    cand = dp_scores[i] + span_scores[i, j, l]
                    # print(i, j, l, cand)
                    if cand > max_cand:
                        max_cand = cand
                        back_i = i
                        back_l = l

            dp_scores[j + 1] = max_cand
            dp_back[j, 0] = back_i
            dp_back[j, 1] = back_l

        # backtracking
        cur_pos = length - 1
        while cur_pos >= 0:
            back_i = dp_back[cur_pos, 0]
            back_l = dp_back[cur_pos, 1]
            for i in range(back_i, cur_pos + 1):
                ret_label[batch_i, i] = back_l
                ret_bio[batch_i, i] = 2
            ret_bio[batch_i, back_i] = 1
            cur_pos = back_i - 1

        for i in range(length):
            if ret_label[batch_i, i] == o_label:
                ret_bio[batch_i, i] = 3

    return ret_bio, ret_label


@cython.boundscheck(False)
@cython.wraparound(False)
def constrained_bio(np.ndarray[np.npy_intp, ndim=1] word_lengths,
                  np.ndarray[np.float64_t, ndim=3] scores):
    cdef int batch_size, seq_len, label_size
    cdef int batch_i, length, i, j, l, k, cur_pos, back_i, back_l
    cdef np.float64_t cand, max_cand

    cdef np.ndarray[np.npy_intp, ndim=2] ret

    cdef np.ndarray[np.npy_intp, ndim=2] dp_back
    cdef np.ndarray[np.float64_t, ndim=2] dp_scores

    batch_size, seq_len, label_size = np.shape(scores)

    label_size -= 2

    ret = np.zeros((batch_size, seq_len), dtype=int)

    for batch_i in range(batch_size):
        length = word_lengths[batch_i]

        # dynamic programming for finding the max
        dp_scores = np.zeros((length, label_size)) - INF
        dp_back = np.zeros((length, label_size), dtype=int)

        # the first in the sequence
        dp_scores[0, 1] = scores[batch_i, 0, 1]

        for i in range(2, label_size, 2):
            dp_scores[0, i] = scores[batch_i, 0, i]

        for i in range(1, length):
            for k in range(1, label_size):
                max_cand = NEGINF
                for j in range(1, label_size):
                    # k = 1: O
                    # k % 2 == 0: B-
                    # k = j + 1: I- following B-
                    # k = j: I- following same I-
                    if (k == 1) or (k % 2 == 0) or (k == j + 1) or (k == j):
                        cand = dp_scores[i - 1, j] + scores[batch_i, i, k]
                        if cand > max_cand:
                            max_cand = cand
                            back_i = j
                dp_scores[i, k] = max_cand
                dp_back[i, k] = back_i

        # backtracking
        cur_pos = length - 1
        back_i = 0
        max_cand = NEGINF
        for i in range(1, label_size):
            if dp_scores[cur_pos, i] > max_cand:
                max_cand = dp_scores[cur_pos, i]
                back_i = i

        while cur_pos >= 0:
            ret[batch_i, cur_pos] = back_i
            back_i = dp_back[cur_pos, back_i]
            cur_pos -= 1

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.npy_intp, ndim=1] parse_proj(np.ndarray[np.float64_t, ndim=2] scores):
    cdef int nr, nc, N, i, k, s, t, r, maxidx
    cdef np.float64_t tmp, cand
    cdef np.ndarray[np.float64_t, ndim=2] complete_0
    cdef np.ndarray[np.float64_t, ndim=2] complete_1
    cdef np.ndarray[np.float64_t, ndim=2] incomplete_0
    cdef np.ndarray[np.float64_t, ndim=2] incomplete_1
    cdef np.ndarray[np.npy_intp, ndim=3] complete_backtrack
    cdef np.ndarray[np.npy_intp, ndim=3] incomplete_backtrack
    cdef np.ndarray[np.npy_intp, ndim=1] heads

    nr, nc = np.shape(scores)

    N = nr - 1 # Number of words (excluding root).

    complete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    complete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).

    complete_backtrack = -np.ones((nr, nr, 2), dtype=int) # s, t, direction (right=1).
    incomplete_backtrack = -np.ones((nr, nr, 2), dtype=int) # s, t, direction (right=1).

    for i in range(nr):
        incomplete_0[i, 0] = NEGINF

    for k in range(1, nr):
        for s in range(nr - k):
            t = s + k
            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = complete_1[s, r] + complete_0[r+1, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
                if s == 0 and r == 0:
                    break
            incomplete_0[t, s] = tmp + scores[t, s]
            incomplete_1[s, t] = tmp + scores[s, t]
            incomplete_backtrack[s, t, 0] = maxidx
            incomplete_backtrack[s, t, 1] = maxidx

            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = complete_0[s, r] + incomplete_0[t, r]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            complete_0[s, t] = tmp
            complete_backtrack[s, t, 0] = maxidx

            tmp = NEGINF
            maxidx = s + 1
            for r in range(s+1, t+1):
                cand = incomplete_1[s, r] + complete_1[r, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            complete_1[s, t] = tmp
            complete_backtrack[s, t, 1] = maxidx

    heads = -np.ones(N + 1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    return heads


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_eisner(np.ndarray[np.npy_intp, ndim=3] incomplete_backtrack,
        np.ndarray[np.npy_intp, ndim=3]complete_backtrack,
        int s, int t, int direction, int complete, np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int r
    if s == t:
        return
    if complete:
        r = complete_backtrack[s, t, direction]
        if direction:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
    else:
        r = incomplete_backtrack[s, t, direction]
        if direction:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return
        else:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return
