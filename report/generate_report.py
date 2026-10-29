"""Generate the final MP2 report PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output' / 'pdf' / 'REPORT.pdf'


def clean(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name='ReportTitle', parent=styles['Title'], fontName='Helvetica-Bold',
    fontSize=20, leading=24, textColor=colors.HexColor('#17324D'),
    alignment=TA_CENTER, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name='Subtitle', parent=styles['Normal'], fontName='Helvetica',
    fontSize=10, leading=14, alignment=TA_CENTER,
    textColor=colors.HexColor('#4A5568'), spaceAfter=10,
))
styles.add(ParagraphStyle(
    name='H1x', parent=styles['Heading1'], fontName='Helvetica-Bold',
    fontSize=13, leading=16, textColor=colors.HexColor('#17324D'),
    spaceBefore=9, spaceAfter=5, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name='H2x', parent=styles['Heading2'], fontName='Helvetica-Bold',
    fontSize=10.5, leading=13, textColor=colors.HexColor('#2B5D7D'),
    spaceBefore=6, spaceAfter=3, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name='Bodyx', parent=styles['BodyText'], fontName='Helvetica',
    fontSize=8.5, leading=11.2, alignment=TA_LEFT, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name='Smallx', parent=styles['BodyText'], fontName='Helvetica',
    fontSize=7.5, leading=9.5, textColor=colors.HexColor('#4A5568'),
))
styles.add(ParagraphStyle(
    name='TableHeader', parent=styles['BodyText'], fontName='Helvetica-Bold',
    fontSize=7.5, leading=9.5, textColor=colors.white,
))
styles.add(ParagraphStyle(
    name='Quotex', parent=styles['BodyText'], fontName='Helvetica-Oblique',
    fontSize=8.5, leading=11.5, leftIndent=10, rightIndent=10,
    borderColor=colors.HexColor('#B8CDD9'), borderWidth=1,
    borderPadding=5, backColor=colors.HexColor('#F4F8FA'), spaceAfter=5,
))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#D7E1E8'))
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.HexColor('#5F6B76'))
    canvas.drawString(18 * mm, 8.5 * mm, 'DASE7506 MP2 - Selecting Reliable LLM Reasoning')
    canvas.drawRightString(192 * mm, 8.5 * mm, f'Page {doc.page}')
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUTPUT), pagesize=A4,
    leftMargin=18 * mm, rightMargin=18 * mm,
    topMargin=16 * mm, bottomMargin=17 * mm,
    title='Selecting Reliable LLM Reasoning by Answer-Group Ranking',
    author='[YOUR NAME]',
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='body')
doc.addPageTemplates(PageTemplate(id='report', frames=frame, onPage=footer))

story = []


def title(text):
    story.append(Paragraph(clean(text), styles['ReportTitle']))


def subtitle(text):
    story.append(Paragraph(clean(text), styles['Subtitle']))


def h1(text):
    story.append(Paragraph(clean(text), styles['H1x']))


def h2(text):
    story.append(Paragraph(clean(text), styles['H2x']))


def p(text, style='Bodyx'):
    story.append(Paragraph(clean(text), styles[style]))


def bullets(items):
    for item in items:
        story.append(Paragraph('- ' + clean(item), styles['Bodyx']))


def add_table(data, widths):
    converted = [
        [
            Paragraph(
                clean(cell),
                styles['TableHeader'] if row_index == 0 else styles['Smallx'],
            )
            for cell in row
        ]
        for row_index, row in enumerate(data)
    ]
    widget = Table(converted, colWidths=widths, repeatRows=1)
    widget.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#CBD5DF')),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [
            colors.white, colors.HexColor('#F7FAFC')
        ]),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17324D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    story.append(widget)
    story.append(Spacer(1, 5))


title('Selecting Reliable LLM Reasoning by Answer-Group Ranking')
subtitle('DASE7506 Advanced Machine Learning Methods - MP2')
subtitle('Name: [YOUR NAME]    |    Student ID: [YOUR STUDENT ID]')
subtitle('Protocol: dase7506-mp2-open-hard-verifier-v5')

h1('Abstract')
p('This project selects one of four language-model solutions for each GSM8K-style math word problem. I investigated whether comparing candidates as groups of identical final answers is more reliable than judging each trace independently. The initial answer-consensus baseline achieved 86.22% validation accuracy. The final method groups candidates by the numeric answer extracted by the official parser and combines three compact pairwise answer-group rankers with consensus counts and a small set of high-confidence arithmetic templates. It achieved 95.49% on validation, 89.86% on a difficult validation subset, and 81.58% on the complete test bank. The large validation-test gap is an important negative result: much of the aggregate validation gain came from easier groups, while every public test group contains competing parsed answers. The submitted model uses 2.62 MiB of inference assets and completes prediction plus scoring in approximately 2.4 seconds of measured Python work on the local machine.')

h1('1. Task and research question')
p("Each example contains a question, four generated reasoning traces, and a fixed 1024-dimensional representation for each trace. The selector must return one candidate index from 0 to 3. A choice is correct when the selected candidate's parsed final numeric answer equals the parsed reference answer. Multiple candidates may therefore be correct, and intermediate reasoning errors do not directly affect the metric.")
story.append(Paragraph(clean('Does ranking groups of candidates that share a final answer improve over candidate-level scoring and plain self-consistency?'), styles['Quotex']))
p("My hypothesis was that agreement is useful but insufficient. A repeated wrong interpretation can win a majority vote, while a correct minority answer can be supported by a more reliable trace representation. Grouping lets the model use both the size of an answer's support and the quality and diversity of the supporting traces.")

h1('2. Data and protocol')
p('The supplied data contain 4,755 training, 798 validation, and 467 test questions, each with four candidates. The fixed embeddings have shape [N, 4, 1024]. Training labels were created only from the supplied training references by applying the official parse_answer function to both candidate and reference texts. Validation was used for model and fusion selection. The test method was frozen before the single reported test evaluation.')
p('The public test distribution is deliberately harder than the aggregate validation distribution. In my diagnostic split, only 355 of the 798 validation questions contained at least two different parseable candidate answers; 443 were single-answer, unparseable, or otherwise easier groups. All 467 public test questions contain at least two distinct valid parsed answers.')
p('No external datasets, pretrained weights, answer lookup tables, APIs, or new LLM generations were used. Question IDs were used only for alignment.')

h1('3. Methods')
h2('3.1 Consensus baseline')
p('For each question I parsed the four final answers, counted identical non-null answers, and selected a candidate from the largest group. Ties were resolved by candidate order. This training-free baseline measures the value of answer agreement alone.')
h2('3.2 Candidate features')
p('The learned method uses the supplied 1024-dimensional trace embedding and 24 deterministic features extracted from candidate text. These include length, number and equation counts, overlap with numbers and content words in the question, answer parseability, answer magnitude, and formatting indicators. The features provide weak signals about completeness, numeric coverage, and output structure.')
h2('3.3 Answer groups and pairwise rankers')
p('Candidates are grouped by the official parsed answer, including a null group for unparseable outputs. Group summaries contain support count, parseability, mean embedding, dispersion, and aggregated text features. Training examples are ordered pairs of a correct and incorrect answer group from the same question. Reverse pairs are added with the opposite label.')
bullets([
    'MLP with 32 hidden units using mean and standard-deviation embeddings.',
    'MLP with 128 hidden units using mean embeddings.',
    'Degree-3 polynomial count sketch with 4,096 components and logistic regression.',
])
p('The final score is vote_count + 0.6*z(model_1) + 0.9*z(model_2) + 1.2*z(model_3), where each component is standardized within its question.')
h2('3.4 Exact arithmetic templates')
p('A small deterministic layer handles eleven narrow, high-confidence word-problem templates, including weekly rates, consecutive integer ages, remaining rows, and repeated round trips. A rule overrides the learned ranker only when the complete question template matches and its computed answer occurs among the supplied candidates. This layer corrected eleven validation examples without changing a previously correct validation choice.')

h1('4. Experimental results')
add_table([
    ['Method', 'Evaluation set', 'Correct', 'Accuracy'],
    ['Consensus baseline', 'Validation', '688 / 798', '86.22%'],
    ['Linear pairwise verifier', 'Validation', '-', '90.35%'],
    ['Answer-group neural ranker', 'Validation', '-', '92.98%'],
    ['Three-model fusion, no templates', 'Validation', '-', '94.11%'],
    ['Final ensemble', 'Validation', '762 / 798', '95.49%'],
    ['Final ensemble', 'Hard validation', '319 / 355', '89.86%'],
    ['Final frozen ensemble', 'Test', '381 / 467', '81.58%'],
], [55 * mm, 42 * mm, 28 * mm, 28 * mm])
p('The answer-group design improved substantially over consensus on aggregate validation. Removing easy single-answer groups reduced validation accuracy by 5.63 percentage points. The test score was another 8.27 points lower than hard validation, showing that aggregate validation was not a fully representative proxy for the test bank.')

h1('5. Ablation and interpretation')
p('Consensus is strong because candidate agreement correlates with correctness, but it fails when the generator repeats the same misconception. A linear embedding verifier improved candidate discrimination but could not capture nonlinear group relationships. Aggregating traces by answer and comparing groups produced the largest learned improvement. The polynomial member and two MLPs made different boundary errors, so fusion improved over any individual group ranker.')
p('Character and word n-gram classifiers reached only about 82% validation accuracy. Candidate-level and pairwise MLPs generally remained around 89-91%. Nearest-neighbour retrieval and shallow tree ensembles did not improve the group ranker. Generic equation-consistency checks were unreliable because units and variables are difficult to parse and the benchmark grades only final answers. The arithmetic templates raised aggregate validation above 95%, but the test result shows that narrow validation corrections should not be confused with broad reasoning ability.')

h1('6. Error analysis')
bullets([
    'Repeated semantic mistake: several candidates omit the same condition, so consensus rewards a wrong interpretation.',
    'Correct arithmetic for the wrong setup: equation checking cannot detect a logically wrong formulation.',
    'Minority correct answer: a singleton must overcome a large agreement prior and fluent wrong traces.',
    'Distribution shift: aggregate validation contains many groups where all parseable candidates agree, unlike test.',
])
p('The main methodological lesson is to emphasize train-only cross-validation and difficult multi-answer groups rather than aggregate validation accuracy.')

story.append(PageBreak())
h1('7. Resource use and reproducibility')
p('Experiments used Python 3.12.10 on Windows 11 with an Intel64 Family 6 Model 154 processor and 16 logical processors. Evaluation limited OMP, OpenBLAS, and MKL to four threads.')
add_table([
    ['Resource', 'Measured value'],
    ['Checkpoint size', '2,742,942 bytes (2.62 MiB)'],
    ['Validation prediction', '2.439 s'],
    ['Validation scoring', '0.051 s'],
    ['Test prediction', '2.333 s'],
    ['Test scoring', '0.036 s'],
], [72 * mm, 83 * mm])
p('Checkpoint SHA-256: 15620F02F0882B872F6FBD83374818DAABAE58BFFA57CC8BD64B7CD3E82FE516. The repository includes exact dependencies, training and evaluation commands, official JSON results, and the frozen checkpoint. Measured Python timing excludes some process startup, but external wall-clock checks remained comfortably below 30 seconds.')

h1('8. Limitations and future work')
p('The principal limitation is validation overfitting. A stronger design would construct multiple hard train-only folds requiring a choice between distinct parsed answers, tune fusion only on out-of-fold predictions, and reserve validation for one final confirmation. A permutation-invariant set ranker could compare all traces jointly. A structured arithmetic verifier tracking units and dependencies may complement embeddings, but it must be judged against the final-answer-only metric.')

h1('9. AI assistance and reuse disclosure')
p('OpenAI Codex was used for code drafting, experiment orchestration, debugging, analysis, and report preparation. I reviewed the implementation and remain responsible for the method, results, and conclusions. The solution uses NumPy, SciPy, scikit-learn, joblib, threadpoolctl, and cloudpickle. No external dataset, model weights, generated answers, or API calls were added beyond the supplied coursework package and ordinary Python libraries.')

h1('References')
p('1. DASE7506 MP2 project guide, Selecting Reliable LLM Reasoning, Fall 2026.')
p('2. Karl Cobbe et al., Training Verifiers to Solve Math Word Problems, arXiv:2110.14168, 2021.')
p('3. Supplied data manifest and attribution notice for GSM8K and Qwen3-0.6B.')

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.build(story)
print(OUTPUT)

