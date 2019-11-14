

def value_from_result(result):
    if result == "1-0":
        return 1

    if result == "1/2-1/2":
        return 0

    if result == "0-1":
        return -1

    print("result string no recognized: ", result)
