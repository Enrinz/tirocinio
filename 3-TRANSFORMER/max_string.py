import csv

moves_string=[]
label=[]
moves=[]
with open("input_1.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        moves_string.append(row[0])
        label.append(row[1])
for i in range (len(moves_string)):
    moves.append(moves_string[i].replace("[","").replace("]","").replace("\'","").split(","))



def max_len_string(list_of_list):
    max=0
    string=""
    for l in list_of_list:
        for s in l:
            if max<len(s):
                max=len(s)
    return max,s
print(max_len_string(moves))


def get_max_str(lst):
    return max(lst, key=len)

flat_list = [item for sublist in moves for item in sublist]
print(get_max_str(flat_list))
str="ProbabilisticWorstRemovalCustomer"
print(len(str))