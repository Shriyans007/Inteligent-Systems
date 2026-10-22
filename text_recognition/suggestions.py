"""Offer a cautious reading of case and round characters in separated text."""

from statistics import median


AMBIGUOUS = {"0": "o", "O": "o", "C": "c", "S": "s"}


def suggest_text(predictions, glyphs):
    """Use relative letter height within a token, leaving raw CNN labels intact.

    A small round mark beside tall letters can look identical after each crop
    is scaled to 28x28. Only suggest a lowercase reading when the token has
    other lowercase letters and an observable short-letter height. This is a
    display suggestion, not a replacement for the classifier's prediction.
    """
    proposed = [item["character"] for item in predictions]
    start = 0
    for end in range(1, len(predictions) + 1):
        if end < len(predictions) and predictions[end]["before"] == "":
            continue
        token = predictions[start:end]
        chars = [item["character"] for item in token]
        # An actual digit in an identifier is a reason to leave the token alone.
        if len(token) >= 3 and sum(c.islower() for c in chars) >= 2 and not any(
            c.isdigit() and c not in AMBIGUOUS for c in chars
        ):
            heights = [glyphs[i].bbox[3] for i in range(start, end)]
            short = [heights[i] for i, c in enumerate(chars)
                     if c.islower() and c not in {"i", "j", "l", "t", "f", "b", "d", "h", "k"}]
            tall = max(heights)
            if short and tall >= 1.35 * median(short):
                for i, char in enumerate(chars):
                    if char in AMBIGUOUS and heights[i] <= 1.25 * median(short) and heights[i] < 0.8 * tall:
                        proposed[start + i] = AMBIGUOUS[char]
        start = end
    return "".join(item["before"] + character for item, character in zip(predictions, proposed))
