import numpy as np

from Process_csv import process_csv_row, process_csv_column


def create_sample_target(path, gap_start, gap_end):
    num_of_elements = 1

    get_rows_length = len(process_csv_row(path, 1))

    samples = np.empty((num_of_elements, 97))
    targets = np.empty((num_of_elements, gap_end-gap_start))

    for i in range(1, num_of_elements):
        sample = np.array(process_csv_row(path, i)[:97], dtype=float)
        target = sample.copy()[gap_start:gap_end]

        for j in range(gap_start, gap_end):
            sample[j] = 0.0

        samples[i-1] = sample
        targets[i-1] = target


    return samples, targets