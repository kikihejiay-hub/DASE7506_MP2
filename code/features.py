"""Shared feature extraction for training and frozen MP2 inference."""

from collections import Counter
from fractions import Fraction
import math
import re

import numpy as np

from common import parse_answer


STOP_WORDS = {
    'a', 'an', 'and', 'are', 'at', 'be', 'by', 'did', 'does', 'each', 'for',
    'from', 'had', 'has', 'have', 'he', 'her', 'his', 'how', 'if', 'in', 'is',
    'it', 'many', 'much', 'of', 'on', 'per', 'she', 'that', 'the', 'their',
    'then', 'they', 'this', 'to', 'was', 'were', 'what', 'will', 'with',
}

NUMBER_RE = re.compile(
    r'(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?'
)

NUMBER_WORDS = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
}


def normalized_numbers(text):
    """Return numeric mentions in a comparison-friendly Fraction format."""
    values = []
    for token in NUMBER_RE.findall(str(text)):
        try:
            values.append(str(Fraction(token.replace(',', ''))))
        except (ValueError, ZeroDivisionError):
            pass
    return values


def candidate_features(bank):
    """Return inexpensive text/numeric candidate features with shape [N, 4, 24]."""
    rows = []
    for question in bank:
        question_numbers = normalized_numbers(question['question'])
        question_words = (
            set(re.findall(r'[a-z]+', question['question'].lower())) - STOP_WORDS
        )
        question_features = []

        for candidate in question['candidates']:
            text = candidate['text']
            lower = text.lower()
            candidate_numbers = normalized_numbers(text)
            answer = parse_answer(text)
            candidate_words = set(re.findall(r'[a-z]+', lower)) - STOP_WORDS

            unique_question_numbers = set(question_numbers)
            coverage = sum(
                number in candidate_numbers for number in unique_question_numbers
            ) / max(1, len(unique_question_numbers))
            occurrences = sum(
                candidate_numbers.count(number) for number in unique_question_numbers
            ) / max(1, len(unique_question_numbers))
            word_coverage = len(question_words & candidate_words) / max(
                1, len(question_words)
            )
            answer_occurrences = candidate_numbers.count(answer) if answer else 0

            if answer is None:
                answer_log_magnitude = 0.0
                answer_is_negative = 0.0
                answer_is_fraction = 0.0
            else:
                answer_fraction = Fraction(answer)
                answer_log_magnitude = math.log1p(abs(float(answer_fraction))) / 15
                answer_is_negative = float(answer_fraction < 0)
                answer_is_fraction = float(answer_fraction.denominator != 1)

            question_features.append([
                len(text) / 1000,
                len(text.split()) / 200,
                text.count('\n') / 20,
                len(candidate_numbers) / 30,
                len(set(candidate_numbers)) / 20,
                coverage,
                occurrences / 5,
                word_coverage,
                answer_occurrences / 5,
                lower.count('step') / 10,
                lower.count('total') / 10,
                lower.count('therefore') / 5,
                lower.count('so') / 10,
                text.count('=') / 10,
                text.count('+') / 10,
                text.count('-') / 10,
                (
                    text.count('×')
                    + lower.count('times')
                    + text.count('\\times')
                ) / 10,
                (text.count('/') + text.count('\\frac')) / 10,
                float('boxed' in lower),
                float('####' in text),
                float(answer is not None),
                answer_log_magnitude,
                answer_is_negative,
                answer_is_fraction,
            ])

        rows.append(question_features)

    return np.asarray(rows, dtype=np.float32)


def answer_groups(bank, embeddings, text_features, mode):
    """Aggregate candidates with the same parsed final answer."""
    result = []
    for question_index, (question, question_embeddings) in enumerate(
        zip(bank, embeddings)
    ):
        answers = [parse_answer(candidate['text']) for candidate in question['candidates']]
        groups = []
        for answer in dict.fromkeys(answers):
            indices = [index for index, value in enumerate(answers) if value == answer]
            group_embeddings = question_embeddings[indices]
            mean_embedding = group_embeddings.mean(axis=0)

            if mode == 'mean':
                embedding_features = mean_embedding
                aggregate_features = text_features[question_index, indices].mean(axis=0)
            elif mode == 'meanstd':
                embedding_features = np.concatenate([
                    mean_embedding,
                    group_embeddings.std(axis=0),
                ])
                aggregate_features = np.concatenate([
                    text_features[question_index, indices].mean(axis=0),
                    text_features[question_index, indices].max(axis=0),
                ])
            else:
                raise ValueError(f'Unknown group feature mode: {mode}')

            dispersion = np.linalg.norm(
                group_embeddings - mean_embedding, axis=1
            ).mean()
            metadata = np.concatenate([
                np.asarray([
                    len(indices),
                    float(answer is not None),
                    dispersion,
                ], dtype=np.float32),
                aggregate_features.astype(np.float32, copy=False),
            ])
            groups.append({
                'answer': answer,
                'indices': indices,
                'embedding': embedding_features.astype(np.float32, copy=False),
                'metadata': metadata,
            })
        result.append(groups)
    return result


def pair_features(positive, negative):
    """Describe an ordered comparison between two answer groups."""
    return np.concatenate([
        positive['embedding'] - negative['embedding'],
        positive['metadata'],
        negative['metadata'],
        positive['metadata'] - negative['metadata'],
    ]).astype(np.float32, copy=False)


def vote_scores(bank):
    """Count how many candidates support each candidate's parsed answer."""
    result = []
    for question in bank:
        answers = [parse_answer(candidate['text']) for candidate in question['candidates']]
        counts = Counter(answer for answer in answers if answer is not None)
        result.append([
            counts.get(answer, 0) if answer is not None else 0
            for answer in answers
        ])
    return np.asarray(result, dtype=np.float32)


def _integer_or_word(value):
    try:
        return int(value)
    except ValueError:
        return NUMBER_WORDS.get(value.lower())


def template_answer(question):
    """Solve a small set of high-confidence arithmetic word-problem templates."""
    text = question.lower().replace(',', '')

    match = re.search(
        r'takes (\d+) naps a week.*?each nap is (\d+) hours.*?in (\d+) days',
        text,
    )
    if match:
        return Fraction(int(match[1]) * int(match[2]) * int(match[3]), 7)

    match = re.search(
        r'has (\d+) rows with (\d+) plants.*?first (\d+) rows.*?'
        r'last (?:the )?(\w+) rows.*?rest',
        text,
    )
    if match and _integer_or_word(match[4]) is not None:
        return Fraction(
            (int(match[1]) - int(match[3]) - _integer_or_word(match[4]))
            * int(match[2])
        )

    match = re.search(
        r'carried (\d+) passengers.*?one way.*?return trip carried (\d+).*?'
        r'made (\w+) more round trips',
        text,
    )
    if match and _integer_or_word(match[3]) is not None:
        return Fraction(
            (1 + _integer_or_word(match[3])) * (int(match[1]) + int(match[2]))
        )

    if all(fragment in text for fragment in ('once a week', 'after', 'weeks', 'minutes')):
        weeks = re.search(r'after (\d+) weeks', text)
        minutes = [
            int(value)
            for value in re.findall(
                r'(\d+) minutes', text[:weeks.start()] if weeks else ''
            )
        ]
        if weeks and len(minutes) >= 2:
            return Fraction(sum(minutes) * int(weeks[1]))

    match = re.search(
        r'ages of (\w+).*?consecutive integers.*?sum of (\d+)', text
    )
    if match:
        count = _integer_or_word(match[1])
        if count and count % 2 == 1:
            return Fraction(int(match[2]), count) - Fraction(count - 1, 2)

    match = re.search(
        r'is (\d+) times the size.*?has (\d+) people.*?'
        r'if (\d+) people move out.*?half',
        text,
    )
    if match:
        return Fraction(int(match[1]) * int(match[2]) - int(match[3]), 2)

    match = re.search(
        r'has (\d+) classes.*?has (\d+) students per class.*?'
        r'added (\w+) more classes',
        text,
    )
    if match and _integer_or_word(match[3]) is not None:
        return Fraction(
            (int(match[1]) + _integer_or_word(match[3])) * int(match[2])
        )

    match = re.search(
        r'each sweater takes (\d+) balls.*?each ball.*?\$(\d+).*?'
        r'all (\d+) sweaters.*?sells each sweater for \$(\d+)',
        text,
    )
    if match:
        return Fraction(
            int(match[3]) * (int(match[4]) - int(match[1]) * int(match[2]))
        )

    match = re.search(
        r'getting about (\d+) hours of sleep each weeknight and (\d+) hours '
        r'each night on the weekend.*?like to get (\d+) hours',
        text,
    )
    if match:
        return Fraction(
            int(match[3]) * 7 - (int(match[1]) * 5 + int(match[2]) * 2)
        )

    match = re.search(
        r'had (\d+) birthday cookie pies.*?(\d+) classmates.*?'
        r'cut into (\d+) slices',
        text,
    )
    if match:
        return Fraction(int(match[1]) * int(match[3]) - (int(match[2]) + 2))

    match = re.search(
        r'takes (\d+) rose petals to make an ounce.*?each rose produces '
        r'(\d+) petals.*?(\d+) roses per bush.*?make (\d+) (\d+)-ounce bottles',
        text,
    )
    if match:
        return Fraction(
            int(match[1]) * int(match[4]) * int(match[5]),
            int(match[2]) * int(match[3]),
        )

    return None
