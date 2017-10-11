# generate a random sample of patients within hospitals
import pandas as pd
import numpy as np
from tmply import tmply


def make_sim(nHosps, nPatients):
    df = pd.DataFrame()
    df['patientid'] = range(nPatients)
    df['hospid'] = np.random.randint(0, nHosps, nPatients)
    df['rdm30d'] = np.random.uniform(0, 1, nPatients) < 0.1
    df['sex'] = np.random.randint(0, 2, nPatients)
    df['age'] = np.random.normal(65, 18, nPatients)
    df['race'] = np.random.randint(0, 4, nPatients)
    df['los'] = np.random.normal(8, 2, nPatients)
    race_dummy = pd.get_dummies(df['race'], prefix = 'race_')
    del df['race']
    df2 = pd.concat([df, race_dummy], axis=1)
    return df2


discharges = makeSim(100, 500000)
tp = tmply()
samples = tp.make_samples(discharges, 300)
closest_sample = tp.closest_sample(discharges, samples)

print(closest_sample.head())
print(closest_sample.shape)
