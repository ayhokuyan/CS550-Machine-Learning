

def read_file(path):
    with open(path) as f:
        lines = f.readlines()
        f.close()
    entries = []
    for line in lines:
        idx_flag = False
        if line.endswith("\n"):
            idx_flag = True
        content = line.split("\n")[0]  # Get rid of \n character
        if idx_flag:
            content = content.split(" ")[:-1]
        else:
            content = content.split(" ")
        entries.append(content)
    return entries


def process_data_line(line):
    if len(line) % 2 != 0:
        raise ValueError("Invalid number of arguments in line")
    idx = 0
    line_params = {}
    while idx < len(line):
        key = line[idx]
        if key.startswith("("):
            key = key[1:]
        if key.endswith(":"):
            key = key[:-1]
        value = line[idx + 1]
        if value.endswith(")") or value.endswith(","):
            value = value[:-1]
        line_params[key] = float(value)
        idx += 2

    return line_params


def process_train_data(path):
    lines = read_file(path)
    entries = []
    for line in lines:
        if len(line) != 0:
            if line[0].startswith("("):
                entries.append(process_data_line(line))
    return entries
