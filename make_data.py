import random

animals = [0, 1, 2]
train = ["path,label"]
test = ["path,label"]
for animal in animals:
    for i in range(1, 1001):
        path = f"Animals/{animal}_{i:04d}.jpg"
        s = f'{path},{animal}\n'
        if random.random() >= 0.2:
            train.append(s)
        else:
            test.append(s)
with open('train.csv', 'w') as f:
    f.writelines(train)
with open('test.csv', 'w') as f:
    f.writelines(train)