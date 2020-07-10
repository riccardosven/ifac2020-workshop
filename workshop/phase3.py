import numpy as np
from sklearn.datasets import fetch_openml


def load_data():
    data = fetch_openml(data_id=41187)

    ppmv_sums = []
    counts = []

    y = data.data[:, 0]
    m = data.data[:, 1]

    month_float = y + (m - 1) / 12
    ppmvs = data.target
    months = []

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            ppmv_sums[-1] += ppmv
            counts[-1] += 1
    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts

    return months, avg_ppmvs
