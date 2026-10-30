"""Decide upper vs lowercase using how big each letter is on the line.

The classifier sees every character resized to 28x28, so it cannot tell "c"
from "C", "o" from "O", "s" from "S" and so on -- those pairs have the same
shape and differ only in size. Size is still available from the segmentation
boxes, so after recognition we compare each ambiguous letter with the rest of
its line:

* baseline  = typical bottom of letters that do not hang below the line
* cap height = typical height above the baseline of letters that are clearly tall
               (unambiguous capitals, digits, and b d h l)
* x-height   = typical height above the baseline of letters that are clearly
               short (a e g n r)

An ambiguous letter taller than halfway between the x-height and the cap height
is a capital, otherwise lowercase. Measuring the top above the baseline (rather
than total height) keeps "y"/"Y" and "p"/"P" apart, since the lowercase ones
hang below the line. If a line gives no reference letters, the classifier's own
answer is kept.
"""
import numpy as np

AMBIGUOUS = set("ckmopsuvwxyz")                 # same shape in both cases
TALL_REFERENCE = set("ABDEFGHIJLNRT0123456789bdhl")
SHORT_REFERENCE = set("aegnr")
HANGS_BELOW = set("gjpqyQ")


def resolve_case(chars, boxes, line_ids=None):
    """Return ``chars`` with ambiguous letters set to the right case.

    chars    -- predicted characters, one per glyph
    boxes    -- (x, y, w, h) for each glyph, in the same image coordinates
    line_ids -- which text line each glyph is on (defaults to all one line)
    """
    chars = list(chars)
    line_ids = list(line_ids) if line_ids is not None else [0] * len(chars)
    for line in sorted(set(line_ids)):
        members = [i for i, l in enumerate(line_ids) if l == line]
        bottoms = [boxes[i][1] + boxes[i][3] for i in members if chars[i] not in HANGS_BELOW]
        if not bottoms:
            continue
        baseline = float(np.median(bottoms))
        rise = {i: baseline - boxes[i][1] for i in members}          # top above baseline
        tall = [rise[i] for i in members if chars[i] in TALL_REFERENCE]
        short = [rise[i] for i in members if chars[i] in SHORT_REFERENCE]
        if tall and short:
            threshold = (np.median(tall) + np.median(short)) / 2
        elif tall:
            threshold = 0.775 * np.median(tall)      # x-height is about 55% of cap height
        elif short:
            threshold = 1.4 * np.median(short)
        else:
            continue
        for i in members:
            if chars[i].lower() in AMBIGUOUS:
                chars[i] = chars[i].upper() if rise[i] > threshold else chars[i].lower()
    return chars
