
def test_confusion_matrix_combine():
    from cognimaker.evaluation.metric import confusion_matrix, ConfusionMatrix
    cm = [
        ConfusionMatrix(101, 4, 9, 20),
        ConfusionMatrix(102, 4, 9, 20),
        ConfusionMatrix(101, 4, 9, 20),
    ]

    assert(confusion_matrix.combine([cm]) == ConfusionMatrix(304, 12, 27, 60))
    assert(confusion_matrix.combine([cm, cm, cm]) == ConfusionMatrix(304, 12, 27, 60))
