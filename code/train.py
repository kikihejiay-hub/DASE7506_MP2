"""Train the compact answer-group ensemble used by submission.py."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, load_inputs, parse_answer, read_jsonl
from features import answer_groups, candidate_features, pair_features


def correctness_labels(bank, references):
    labels = []
    for question, reference in zip(bank, references):
        gold = parse_answer(reference['answer'])
        labels.append([
            parse_answer(candidate['text']) is not None
            and parse_answer(candidate['text']) == gold
            for candidate in question['candidates']
        ])
    return np.asarray(labels, dtype=bool)


def group_training_rows(groups, labels):
    positive_rows = []
    for question_index, question_groups in enumerate(groups):
        positive_groups = [
            group
            for group in question_groups
            if labels[question_index, group['indices']].any()
        ]
        negative_groups = [
            group
            for group in question_groups
            if not labels[question_index, group['indices']].any()
        ]
        for positive in positive_groups:
            for negative in negative_groups:
                positive_rows.append(pair_features(positive, negative))

    positive_rows = np.asarray(positive_rows, dtype=np.float32)
    metadata_size = groups[0][0]['metadata'].size
    embedding_size = groups[0][0]['embedding'].size
    reverse_rows = np.concatenate([
        -positive_rows[:, :embedding_size],
        positive_rows[:, embedding_size + metadata_size:embedding_size + 2 * metadata_size],
        positive_rows[:, embedding_size:embedding_size + metadata_size],
        -positive_rows[:, embedding_size + 2 * metadata_size:],
    ], axis=1)
    rows = np.concatenate([positive_rows, reverse_rows], axis=0)
    targets = np.concatenate([
        np.ones(len(positive_rows), dtype=np.int8),
        np.zeros(len(positive_rows), dtype=np.int8),
    ])
    return rows, targets


def train_mlp(rows, targets, hidden_units, alpha, seed):
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            alpha=alpha,
            batch_size=128,
            learning_rate_init=5e-4,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=8,
            random_state=seed,
        ),
    )
    model.fit(rows, targets)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    bank, arrays = load_inputs('train', args.data_dir)
    references = read_jsonl(args.data_dir / 'gsm8k_train.jsonl')
    labels = correctness_labels(bank, references)
    text_features = candidate_features(bank)

    meanstd_groups = answer_groups(
        bank, arrays['embeddings'], text_features, mode='meanstd'
    )
    meanstd_rows, meanstd_targets = group_training_rows(meanstd_groups, labels)
    meanstd_model = train_mlp(
        meanstd_rows, meanstd_targets, hidden_units=32, alpha=0.3, seed=432
    )

    mean_groups = answer_groups(
        bank, arrays['embeddings'], text_features, mode='mean'
    )
    mean_rows, mean_targets = group_training_rows(mean_groups, labels)
    mean_model = train_mlp(
        mean_rows, mean_targets, hidden_units=128, alpha=1.0, seed=428
    )

    polynomial_model = make_pipeline(
        StandardScaler(),
        PolynomialCountSketch(
            gamma=1 / mean_rows.shape[1],
            degree=3,
            coef0=1,
            n_components=4096,
            random_state=10,
        ),
        LogisticRegression(C=0.3, max_iter=1000),
    )
    polynomial_model.fit(mean_rows, mean_targets)

    bundle = {
        'format': 'dase7506-mp2-answer-group-v1',
        'meanstd_model': meanstd_model,
        'mean_model': mean_model,
        'polynomial_model': polynomial_model,
        'weights': np.asarray([0.6, 0.9, 1.2], dtype=np.float32),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / 'model.joblib'
    joblib.dump(bundle, output_file, compress=0)
    print({
        'output': str(output_file),
        'bytes': output_file.stat().st_size,
        'training_pairs': int(len(mean_targets)),
    })


if __name__ == '__main__':
    main()
