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


#------------------------------PRINT FUNCTIONS-------------------------------------#

COL_LABEL_WIDTH = 12
COL_DATA_WIDTH = 22

def print_keys(keys):
    print(f"{'':<{COL_LABEL_WIDTH}}", end="")
    for key in keys:
        max_len = COL_DATA_WIDTH - 2
        
        if len(key) > max_len:
            display_key = key[:max_len - 10] + "..."
        else:
            display_key = key
            
        print(f"{display_key:>{COL_DATA_WIDTH}}", end="")
    print()

def print_count(feats):
    print(f"{'Count':<{COL_LABEL_WIDTH}}", end="")
    for value in feats.values():
        total = float(len(value))
        print(f"{total:>{COL_DATA_WIDTH}.6f}", end="")
    print()

def print_mean(feats):
    print(f"{'Mean':<{COL_LABEL_WIDTH}}", end="")
    for value in feats.values():
        mean = sum(value) / len(value) if value else 0.0
        print(f"{mean:>{COL_DATA_WIDTH}.6f}", end="")
    print()

# L'écart-type mesure la dispersion (ou la variabilité) des données autour de la moyenne.
# Un std faible signifie que les valeurs sont très regroupées, très proches de la moyenne.
# Un std élevé signifie que les valeurs sont très étalées, dispersées sur une large plage de valeurs.
def print_std(feats):
    print(f"{'Std':<{COL_LABEL_WIDTH}}", end="")
    for value in feats.values():
        n = len(value)
        if n < 2:
            res = 0.0
        else:
            mean = sum(value) / n
            variance_sum = sum((v - mean) ** 2 for v in value)
            variance = variance_sum / (n - 1)
            res = variance ** 0.5
        print(f"{res:>{COL_DATA_WIDTH}.6f}", end="")
    print()

def print_min(feats):
    print(f"{'Min':<{COL_LABEL_WIDTH}}", end="")
    for value in feats.values():
        print(f"{value[0]:>{COL_DATA_WIDTH}.6f}", end="")
    print()

def print_max(feats):
    print(f"{'Max':<{COL_LABEL_WIDTH}}", end="")
    for value in feats.values():
        print(f"{value[-1]:>{COL_DATA_WIDTH}.6f}", end="")
    print()

def first_quartile(feats):
    print(f"{'25%':<{COL_LABEL_WIDTH}}", end="")
    for value in feats.values():
        res = percentile(value, 0.25)
        print(f"{res:>{COL_DATA_WIDTH}.6f}", end="")
    print()
    
def second_quartile(feats):
    print(f"{'50%':<{COL_LABEL_WIDTH}}", end="")
    for value in feats.values():
        res = percentile(value, 0.50)
        print(f"{res:>{COL_DATA_WIDTH}.6f}", end="")
    print()

def third_quartile(feats):
    print(f"{'75%':<{COL_LABEL_WIDTH}}", end="")
    for value in feats.values():
        res = percentile(value, 0.75)
        print(f"{res:>{COL_DATA_WIDTH}.6f}", end="")
    print()

def presentation(feats, sorted_feats, keys):
    sample_feats = {k: feats[k] for k in keys}
    sample_sorted_feats = {k: sorted_feats[k] for k in keys}


    print_keys(sample_feats)
    print_count(sample_feats)
    print_mean(sample_feats)
    print_std(sample_feats)
    print_min(sample_sorted_feats)
    first_quartile(sample_sorted_feats)
    second_quartile(sample_sorted_feats)
    third_quartile(sample_sorted_feats)
    print_max(sample_sorted_feats)


if __name__ == "__main__":
    data = open_data()
    feats = parse_feats(data)
    sorted_feats = {}
    for key, value in feats.items():
        sorted_feats[key] = sort_values(value)

    keys = list(feats.keys())
    keys_cuted = [keys[i:i+4] for i in range(0, len(keys), 4)]
    for i in range(0, len(keys_cuted)):
        presentation(feats, sorted_feats, keys_cuted[i])
        print()
