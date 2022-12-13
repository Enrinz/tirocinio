strings=['ciao','sono','bello','bello']



for i in range(len(strings) - 1):
    string1 = strings[i]
    string2 = strings[i + 1]
    if string1 == string2:
        continue
    else:
        print(string1)
