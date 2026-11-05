import csv

def open_data():
    try:
        with open("datasets/dataset_train.csv", "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except (OSError, csv.Error):
        print('Warning: Failed to load file.')
        exit(1)
    return data

def is_num(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def print_keys(feats):
    print("             ", end="")
    for key in feats.keys():
        print(f"{key}    ", end="")
    print()


def print_count(feats):
    total = 0.0
    print("Count    ", end="")
    for value in feats.values():
        total = sum(value)
        print(f"{total}    ", end="")

def print_mean(feats):
    total = 0.0
    if (len(feats.values()) == 0):
        return 0.0
    print("Mean     ", end="")
    for value in feats.values():
        total = sum(value) / len(value)
        print(f"{total}    ", end="")

# L'écart-type mesure la dispersion (ou la variabilité) des données autour de la moyenne.
# Un std faible signifie que les valeurs sont très regroupées, très proches de la moyenne.
# Un std élevé signifie que les valeurs sont très étalées, dispersées sur une large plage de valeurs.
def print_std(feats):
    total = 0.0
    if (len(feats.values()) == 0):
        return 0.0
    print("Std      ", end="")
    for value in feats.values():
        n = len(value)
        mean = sum(value) / n
        for v in value:
            variance_sum = (v - mean) **2
            variance = variance_sum / (n - 1)
            res = variance **0.5
        print(f"{res}    ", end="")

def parse_feats(data):

    feats = {}

    first_row = data[0]
    for key, value in first_row.items():
        if (is_num(value)):
            feats[key] = []
    
    for row in data:
        for key in feats:
            value = row.get(key)
            if (is_num(value)):
                feats[key].append(float(value))
    return feats
    
def sort_values(values):
    sorted_values = []
    tmp_values = list(values)
    len_values = len(values)

    for i in range(len_values):
        min = tmp_values[0]
        for v in tmp_values:
            if (v < min):
                min = v
        sorted_values.append(min)
        tmp_values.remove(min)
    return sorted_values


def print_min(feats):
    print("Min      ", end="")
    for value in feats.values():
        print(f"{value[0]}    ", end="")

def print_max(feats):
    print("Max      ", end="")
    for value in feats.values():
        print(f"{value[len(value) - 1]}    ", end="")

def first_quartile(feats):
    print("25%      ", end="")
    for value in feats.values():
        res = percentile(value, 0.25)
        print(f"{res}    ", end="")
    
def second_quartile(feats):
    print("50%      ", end="")
    for value in feats.values():
        res = percentile(value, 0.50)
        print(f"{res}    ", end="")

def third_quartile(feats):
    print("75%      ", end="")
    for value in feats.values():
        res = percentile(value, 0.75)
        print(f"{res}    ", end="")

def percentile(values, p):
    n = len(values)

    if (not values):
        return 0.0
    if (n == 1):
        return values[0]
    pos = p * (n - 1)
    i = int(pos)
    frac = pos - i

    if (i + 1 >= n):
        return (values[i])

    lower_value = values[i]
    upper_value = values[i + 1]

    return float(lower_value + frac * (upper_value - lower_value))


def presentation(feats, sorted_feats, keys):
    sample = {}
    sample_sorted = {}

    for key in keys():
        sample[key] = feats[key]
    for key in keys():
        sample[key] = feats[key]
    for key in keys:
        print_count(sample_feats)
        print_mean(sample_feats)
        print()
        print_std(sample_feats)
        print()
        print_min(sample_sorted_feats)
        print()
        first_quartile(sample_sorted_feats)
        print()
        second_quartile(sample_sorted_feats)
        print()
        third_quartile(sample_sorted_feats)
        print()
        print_max(sample_sorted_feats)
        print()


if __name__ == "__main__":
    data = open_data()
    feats = parse_feats(data)
    sorted_feats = {}
    for key, value in feats.items():
        sorted_feats[key] = sort_values(value)

    keys = list(feats.keys())
    keys_cuted = [keys[i:i+4] for i in range(0, len(keys), 4)]
    print(keys_cuted)