from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from sty import fg, bg, ef, rs, RgbFg

def count_confusion_matrix(pred_entities, gold_entities):
    true_positive, false_positive, false_negative = 0, 0, 0
    for span_item in pred_entities:
        if span_item in gold_entities:
            true_positive += 1
            gold_entities.remove(span_item)
        else:
            false_positive += 1
    # these entities are not predicted.
    for span_item in gold_entities:
        false_negative += 1
    return true_positive, false_positive, false_negative
# import spacy
# spacy.prefer_gpu()
# nlp = spacy.load("en_core_web_sm")
# from spacy.gold import align

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
y_true = [['I-NP', 'I-NP', 'I-NP', 'O', 'I-C', 'O', 'I-C', 'O', 'I-C', 'O', 'O']]
y_pred = [['I-NP', 'I-NP', 'I-NP', 'O', 'I-C', 'O', 'I-C', 'O', 'I-NP', 'I-NP', 'O']]
# YP=[1,2,0,3,0,0]
# YT=[2,2,0,2,2,0]
print(classification_report(y_true, y_pred,digits=4))
print(f1_score(y_true, y_pred, average = 'macro'))
print(f1_score(y_true, y_pred))

y_true = [['I-NP', 'I-NP', 'I-NP', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'I-C', 'O', 'I-C', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'I-C', 'O', 'O']]
y_pred = [['B-NP', 'I-NP', 'I-NP','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O','I-C', 'O', 'B-C', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O','I-NP', 'I-NP', 'O']]
# YP=[1,2,0,3,0,0]
# YT=[2,2,0,2,2,0]
print(classification_report(y_true, y_pred,digits=4))
print(f1_score(y_true, y_pred, average = 'macro'))
print(f1_score(y_true, y_pred))


# set1=set()
# set1.add((1,2,3))
# set1.add((1,2,5))
# set1.add((1,2,4))
# set1.add((1,2,3))
# print(set1)
# p, r, f1s, s = metrics.precision_recall_fscore_support(y_pred=YP, y_true=YT, average='macro', warn_for=tuple())
# print(f1s)
# cm = confusion_matrix(y_pred=YP, y_true=YT)
# print(cm)
# aaa = f1_score(y_true, y_pred, average = 'macro')
# classification_report(y_true, y_pred)
# p, r, f1s, s = metrics.precision_recall_fscore_support(y_pred=YP, y_true=YT, average='macro', warn_for=tuple())
# cm = confusion_matrix(y_pred=YP, y_true=YT)

# sum_true_positive, sum_false_positive, sum_false_negative = 0, 0, 0
#
# for seq_pred_item, seq_gold_item in zip(y_pred, y_true):
#     gold_entity_lst = get_entity_from_bmes_lst(seq_gold_item)
#     pred_entity_lst = get_entity_from_bmes_lst(seq_pred_item)
#
#     true_positive_item, false_positive_item, false_negative_item = count_confusion_matrix(pred_entity_lst, gold_entity_lst)
#     sum_true_positive += true_positive_item
#     sum_false_negative += false_negative_item
#     sum_false_positive += false_positive_item
#
# batch_confusion_matrix = [sum_true_positive, sum_false_positive, sum_false_negative]
# print(batch_confusion_matrix)