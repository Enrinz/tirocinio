def f1_score(precision,recall):
    return 2*precision*recall/(precision+recall)

print(f1_score(0.5255, 0.0556))