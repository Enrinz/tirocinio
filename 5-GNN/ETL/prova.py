def dictarget():
    target_std = {}
    input = open("DemandBasedDestroyCustomer_labels.txt", "r")
    labels=[]
    for lines in input.readlines():                                                
      line = lines.split()
      labels.append(int(line[0]))
    input.close()
    print(labels)
    for key in labels:
      # add each key to the dictionary, with the value being a default value
      target_std[key] = None
    print(target_std)
    return target_std

dictarget()