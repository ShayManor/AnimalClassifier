animals = [0, 1, 2]
with open('data.csv', 'w') as f:
    for animal in animals:
        for i in range(1, 1001):
            path = f"Animals/{animal}_{i:04d}.jpg"
            f.write(f'{path},{animal}\n')
