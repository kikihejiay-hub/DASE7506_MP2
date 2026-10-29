"""Compact answer-group ensemble for MP2 inference."""

from pathlib import Path

import joblib
import numpy as np

from common import parse_answer
from features import (
    answer_groups,
    candidate_features,
    pair_features,
    template_answer,
    vote_scores,
)


def _group_scores(model, groups):
    """Run all pairwise group comparisons in one model batch."""
    rows = []
    locations = []
    for question_index, question_groups in enumerate(groups):
        for left in range(len(question_groups)):
            for right in range(left + 1, len(question_groups)):
                rows.append(pair_features(question_groups[left], question_groups[right]))
                locations.append((question_index, left, right))

    candidate_scores = np.zeros((len(groups), 4), dtype=np.float32)
    if rows:
        probabilities = model.predict_proba(np.asarray(rows, dtype=np.float32))[:, 1]
        group_scores = [np.zeros(len(question_groups)) for question_groups in groups]
        for probability, (question_index, left, right) in zip(probabilities, locations):
            group_scores[question_index][left] += probability
            group_scores[question_index][right] += 1 - probability

        for question_index, question_groups in enumerate(groups):
            for group_index, group in enumerate(question_groups):
                candidate_scores[question_index, group['indices']] = group_scores[
                    question_index
                ][group_index]
    return candidate_scores


def _group_standardize(scores):
    return (
        scores - scores.mean(axis=1, keepdims=True)
    ) / (scores.std(axis=1, keepdims=True) + 1e-6)


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    if checkpoint is None:
        raise ValueError('This method requires --checkpoint runs/model_bundle')
    checkpoint = Path(checkpoint)
    model_path = checkpoint / 'model.joblib' if checkpoint.is_dir() else checkpoint
    bundle = joblib.load(model_path)
    if bundle.get('format') != 'dase7506-mp2-answer-group-v1':
        raise ValueError('Unsupported checkpoint format')

    text_features = candidate_features(bank)
    meanstd_groups = answer_groups(
        bank, arrays['embeddings'], text_features, mode='meanstd'
    )
    mean_groups = answer_groups(
        bank, arrays['embeddings'], text_features, mode='mean'
    )

    component_scores = [
        _group_scores(bundle['meanstd_model'], meanstd_groups),
        _group_scores(bundle['mean_model'], mean_groups),
        _group_scores(bundle['polynomial_model'], mean_groups),
    ]
    scores = vote_scores(bank)
    for weight, component in zip(bundle['weights'], component_scores):
        scores += float(weight) * _group_standardize(component)

    choices = scores.argmax(axis=1).astype(int).tolist()

    # These rules only override the learned ranker when a question exactly
    # matches a high-confidence arithmetic template and the computed answer is
    # present among the supplied candidates.
    for question_index, question in enumerate(bank):
        answer = template_answer(question['question'])
        if answer is None:
            continue
        matching = [
            index
            for index, candidate in enumerate(question['candidates'])
            if parse_answer(candidate['text']) == str(answer)
        ]
        if matching:
            choices[question_index] = int(
                max(matching, key=lambda index: scores[question_index, index])
            )

    return choices
