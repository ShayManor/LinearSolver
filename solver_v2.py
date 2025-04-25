import random

import numpy as np

parser = {'Iris-setosa': '0', 'Iris-versicolor': '1', 'Iris-virginica': '2'}


def solve(A, b):
    return np.linalg.lstsq(A, b)[0]


yesno = {'Yes': 1, 'No': -1}
gender = {'Male': 1, 'Female': -1}
physical = {'Low': -1, 'Medium': 0, 'High': 1}
smoking = {'Current': 1, 'Never': -1, 'Former': 0}
alcohol = {'Occasionally': 0, 'Never': -1, 'Regularly': 1}
cholesterol = {'High': 1, 'Normal': 0}
sleep = {'Poor': -1, 'Average': 0, 'Good': 1}
health = {'Unhealthy': -1, 'Average': 0, 'Healthy': 1}
employment = {'Unemployed': -1, 'Retired': 0, 'Employed': 1}
marriage = {'Single': -1, 'Widowed': 0, 'Married': 1}
rural_urban = {'Rural': -1, 'Urban': 1}
diagnosis = {'Yes': 1, 'No': 1}

def parsedata(data):
    data[0] = 0
    data[1] = (float(data[1]) - 50) / 50
    data[2] = gender[data[2]]
    data[3] = float(data[3]) / 19
    data[4] = (float(data[3]) - 18.5) / 18
    data[5] = physical[data[5]]
    data[6] = smoking[data[6]]
    data[7] = alcohol[data[7]]
    data[8] = yesno[data[8]]
    data[9] = yesno[data[9]]
    data[10] = cholesterol[data[10]]
    data[11] = yesno[data[11]]
    data[12] = float(data[12]) / 60.0
    data[13] = physical[data[13]]
    data[14] = sleep[data[14]]
    data[15] = health[data[15]]
    data[16] = physical[data[16]]
    data[17] = employment[data[17]]
    data[18] = marriage[data[18]]
    data[19] = yesno[data[19]]
    data[20] = physical[data[20]]
    data[21] = physical[data[21]]
    data[22] = physical[data[22]]
    data[23] = rural_urban[data[23]]
    data[24] = yesno[data[24]]
    return data


def split():
    with open('alzheimers_prediction_dataset.csv') as f:
        lines = f.readlines()
    test = []
    train = []
    for line in lines:
        if line == lines[0]:
            continue
        line = line.replace('\n', '')
        line = parsedata(line.split(','))
        for val in parser.keys():
            if line.count(val) > 0:
                line = line.replace(val, parser[val])
        if random.random() < 0.2:
            test.append()
        else:
            train.append(line)
    with open('test.csv', 'w') as f:
        f.writelines(test)
    with open('train.csv', 'w') as f:
        f.writelines(train)


def train(idx):
    with open('train.csv') as f:
        lines = f.readlines()
    A = []
    b = []
    for line in lines:
        data = [float(x) for x in line.split(',')]
        if data[-1] == idx:
            data[-1] = 1
        else:
            data[-1] = 0
        A.append(data[:-1])
        b.append(data[-1])
    return solve(A, b)


def test(weights, idx):
    with open('test.csv') as f:
        lines = f.readlines()

    def check(val1, val2):
        if val1 == idx:
            val1 = 1
        else:
            val1 = 0
        return val1 == round(val2)

    correct = 0
    for line in lines:
        data = np.array([float(x) for x in line.split(',')[:-1]])
        pred = np.matmul(weights, data)
        exp = float(line.split(',')[-1])
        if check(exp, pred):
            correct += 1
        # else:
        #     print(f'Incorrect, expected: {exp} Predicted {pred}')
    print(f'Accuracy: {correct}/{len(lines) - 1} for {idx}')
    return correct, len(lines) - 1


if __name__ == '__main__':
    tot = 0
    l = 0
    split()
    exit()
    for idx in range(1):
        weights = train(idx)
        correct, length = test(weights, idx)
        tot += correct
        l += length
    print(f"Total Accuracy: {tot}/{l} = {100.0 * float(tot)/l}%")

