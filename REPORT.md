# Selecting Reliable LLM Reasoning by Answer-Group Ranking

**DASE7506 Advanced Machine Learning Methods - MP2**  
**Name:** He Jiayin  
**Student ID:** 3036705993    
**Protocol:** dase7506-mp2-open-hard-verifier-v5

## Abstract

This project selects one of four language-model solutions for each GSM8K-style
math word problem. I investigated whether comparing candidates as groups of
identical final answers is more reliable than judging each trace independently.
The initial answer-consensus baseline achieved 86.22% validation accuracy. The
final method groups candidates by the numeric answer extracted by the official
parser and combines three compact pairwise answer-group rankers with consensus
counts and a small set of high-confidence arithmetic templates. It achieved
95.49% on validation, 89.86% on a difficult validation subset, and 81.58% on
the complete test bank. The large validation-test gap is an important negative
result: much of the aggregate validation gain came from easier groups, while
every public test group contains competing parsed answers. The submitted model
uses 2.62 MiB of inference assets and completes prediction plus scoring in
approximately 2.4 seconds of measured Python work on the local machine.

## 1. Task and research question

Each example contains a question, four generated reasoning traces, and a fixed
1024-dimensional representation for each trace. The selector must return one
candidate index from 0 to 3. A choice is correct when the selected candidate's
parsed final numeric answer equals the parsed reference answer. Multiple
candidates may therefore be correct, and intermediate reasoning errors do not
directly affect the metric.

The main research question was:

> Does ranking groups of candidates that share a final answer improve over
> candidate-level scoring and plain self-consistency?

My hypothesis was that agreement is useful but insufficient. A repeated wrong
interpretation can win a majority vote, while a correct minority answer can be
supported by a more reliable trace representation. Grouping lets the model use
both the size of an answer's support and the quality and diversity of the
supporting traces.

## 2. Data and protocol

The supplied data contain 4,755 training, 798 validation, and 467 test
questions, each with four candidates. The fixed embeddings have shape
[N, 4, 1024]. Training labels were created only from the supplied training
references by applying the official parse_answer function to both candidate
and reference texts. Validation was used for model and fusion selection. The
test method was frozen before the single reported test evaluation.

The public test distribution is deliberately harder than the aggregate
validation distribution. In my diagnostic split, only 355 of the 798
validation questions contained at least two different parseable candidate
answers; 443 were single-answer, unparseable, or otherwise easier groups. All
467 public test questions contain at least two distinct valid parsed answers.

No external datasets, pretrained weights, answer lookup tables, APIs, or new
LLM generations were used. Question IDs were used only for alignment.

## 3. Methods

### 3.1 Consensus baseline

For each question I parsed the four final answers, counted identical non-null
answers, and selected a candidate from the largest group. Ties were resolved by
candidate order. This training-free baseline is fast and measures the value of
answer agreement alone.

### 3.2 Candidate features

The learned method uses the supplied 1024-dimensional trace embedding and 24
small deterministic features extracted from candidate text. These include
length, number and equation counts, overlap with numbers and content words in
the question, answer parseability, answer magnitude, and formatting indicators.
The text features provide weak signals about completeness, numeric coverage,
and output structure.

### 3.3 Answer groups

Candidates are grouped by the official parsed answer, including a separate null
group for unparseable outputs. For each group I compute:

- the number of supporting candidates;
- whether the answer is parseable;
- mean embedding and within-group dispersion;
- mean text features;
- for one ensemble member, embedding standard deviation and maximum text
  features.

All candidates in a group receive the same group score because the official
metric treats their final answer identically.

### 3.4 Pairwise group rankers

Training examples are ordered pairs consisting of one correct answer group and
one incorrect answer group from the same question. A pair is represented by
the embedding difference, both groups' metadata, and the metadata difference.
Reverse pairs are added with the opposite label.

The ensemble contains:

1. an MLP with 32 hidden units using mean and standard-deviation embedding
   features;
2. an MLP with 128 hidden units using mean embeddings;
3. a degree-3 polynomial count-sketch representation with 4,096 components,
   followed by logistic regression.

The MLPs use standardized inputs, Adam optimization, early stopping, and fixed
random seeds. The final candidate score is:

vote_count + 0.6*z(model_1) + 0.9*z(model_2) + 1.2*z(model_3),

where each model score is standardized within its four-candidate question.

### 3.5 Exact arithmetic templates

A small deterministic layer handles eleven narrow, high-confidence word-problem
templates, such as weekly-rate conversion, consecutive integer ages, remaining
rows, and repeated round trips. A rule can override the learned ranker only if
the question matches the complete template and the computed numeric answer is
present among the supplied candidates. This layer corrected eleven validation
examples without changing any previously correct validation choice. Its limited
coverage is a deliberate trade-off against brittle general-purpose parsing.

## 4. Experimental results

| Method | Evaluation set | Correct | Accuracy |
|---|---|---:|---:|
| Consensus baseline | Validation | 688 / 798 | 86.22% |
| Linear pairwise embedding verifier | Validation | - | 90.35% |
| Answer-group neural ranker | Validation | - | 92.98% |
| Three-model group fusion, no templates | Validation | - | 94.11% |
| Final ensemble | Validation | 762 / 798 | 95.49% |
| Final ensemble | Hard validation subset | 319 / 355 | 89.86% |
| Final frozen ensemble | Test | 381 / 467 | 81.58% |

The answer-group design improved substantially over consensus on aggregate
validation. The hard-subset result is more informative: removing easy
single-answer groups reduced accuracy by 5.63 percentage points. The test score
was another 8.27 points lower, showing that the validation set was not a fully
representative proxy for the test bank.

## 5. Ablation and interpretation

Consensus alone was strong because candidate agreement is correlated with
correctness, but it fails when the generator repeats the same misconception.
A linear embedding verifier improved candidate discrimination but could not
capture nonlinear group relationships. Aggregating traces by answer and using
pairwise group comparisons produced the largest learned improvement. The
polynomial member and the two MLPs made different boundary errors, so their
fusion improved over any single group ranker.

Several explored alternatives were weaker:

- character and word n-gram classifiers reached only about 82% validation
  accuracy, suggesting that surface style was not a sufficient verifier;
- candidate-level MLPs and pairwise MLPs generally remained around 89-91%;
- nearest-neighbour retrieval and shallow tree ensembles did not improve the
  group ranker;
- generic equation-consistency checks were unreliable because units and
  algebraic variables are difficult to parse, and because the benchmark grades
  only the final answer. A trace can contain a wrong intermediate equation and
  still receive credit if its final number is correct.

The arithmetic templates raised aggregate validation accuracy above 95%, but
the test result demonstrates that narrow validation corrections should not be
confused with broad reasoning ability.

## 6. Error analysis

The remaining errors fall into four recurring categories:

1. **Repeated semantic mistake.** Three candidates may independently omit the
   same condition, causing both consensus and group size to favour a wrong
   answer.
2. **Correct arithmetic for the wrong interpretation.** Equation checking
   cannot detect a logically wrong setup when every calculation is internally
   consistent.
3. **Minority correct answer.** A single correct trace must overcome a large
   consensus prior and may have an embedding similar to fluent wrong traces.
4. **Distribution shift.** Aggregate validation includes many groups where all
   parseable candidates agree. These examples inflate accuracy but do not
   represent the public test condition.

The most important lesson is methodological: model selection should emphasize
train-only cross-validation and difficult multi-answer groups rather than only
aggregate validation accuracy.

## 7. Resource use and reproducibility

Experiments used Python 3.12.10 on Windows 11 with an Intel64 Family 6 Model 154
processor and 16 logical processors. Evaluation set OMP_NUM_THREADS,
OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS to 4.

| Resource | Measured value |
|---|---:|
| Checkpoint size | 2,742,942 bytes (2.62 MiB) |
| Validation prediction time | 2.439 s |
| Validation scoring time | 0.051 s |
| Test prediction time | 2.333 s |
| Test scoring time | 0.036 s |

The checkpoint SHA-256 is
15620F02F0882B872F6FBD83374818DAABAE58BFFA57CC8BD64B7CD3E82FE516.
The repository includes exact dependency versions, training and evaluation
commands, official prediction and score files, and the frozen checkpoint.
Measured Python timing fields do not include every part of process startup;
external wall-clock checks remained comfortably below the 30-second limit.

## 8. Limitations and future work

The principal limitation is validation overfitting. A better experimental
design would construct multiple hard train-only folds that require choosing
between distinct parsed answers, tune every fusion weight only on out-of-fold
predictions, and reserve validation as a single final confirmation. A
permutation-invariant set ranker could compare all four traces jointly while
avoiding candidate-position artefacts. A structured arithmetic verifier that
tracks units and equation dependencies may also complement embeddings, but it
must be evaluated against the benchmark's final-answer-only metric.

## 9. AI assistance and reuse disclosure

OpenAI Codex was used for code drafting. I reviewed the implementation and remain
responsible for the method, reported results, and conclusions. The solution
uses NumPy, SciPy, scikit-learn, joblib, threadpoolctl, and cloudpickle. No
external code, dataset, model weights, or generated answers were added beyond
the supplied coursework package and these ordinary Python libraries.

## References

1. DASE7506 MP2 project guide, *Selecting Reliable LLM Reasoning*, Fall 2026.
2. Karl Cobbe et al., *Training Verifiers to Solve Math Word Problems*,
   arXiv:2110.14168, 2021.
3. Supplied data manifest and attribution notice for GSM8K and Qwen3-0.6B.

