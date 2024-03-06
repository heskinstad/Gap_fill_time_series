def Normalize(data_array, max, min):

    normalized_data_array = data_array.copy()

    for i in range(len(normalized_data_array)):
        normalized_data_array[i] = (normalized_data_array[i] + min) / (max+min)

    return normalized_data_array

def Normalize_single(value, max, min):
    return (value + min) / (max+min)

def Denormalize(data_array, max, min):

    denormalized_data_array = data_array.copy()

    for i in range(len(denormalized_data_array)):
        denormalized_data_array[i] = ((max+min) * denormalized_data_array[i]) - min

    return denormalized_data_array

def Denormalize_single(value, max, min):
    return ((max+min) * value) - min