import pandas as pd
import scipy as sp
import numpy as np
from scipy.spatial.distance import mahalanobis

class tmply(object):
    """
    This is a python implementation of template matching
    for comparing hospital outcomes
    """
    def __init__(self):
        print("Template initiated")

    def make_sim(self, nHosps, nPatients):
        df = pd.DataFrame()
        df['patientid'] = range(nPatients)
        df['hospid'] = np.random.randint(0, nHosps, nPatients)
        df['rdm30d'] = np.random.uniform(0, 1, nPatients) < 0.1
        df['sex'] = np.random.randint(0, 2, nPatients)
        df['age'] = np.random.normal(65, 18, nPatients)
        df['race'] = np.random.randint(0, 4, nPatients)
        df['los'] = np.random.normal(8, 2, nPatients)
        race_dummy = pd.get_dummies(df['race'], prefix='race_')
        del df['race']
        df2 = pd.concat([df, race_dummy], axis=1)
        return df2

    def make_samples(self, orig, n_samples, n_cases):
        # generate k random samples of n patients
        samples = pd.DataFrame()
        for k in range(0, n_samples):
            new_sample = orig.sample(n=n_cases, random_state=k)
            new_sample['sample_id'] = k
            samples = pd.concat([samples, new_sample], axis=0)
        return samples

    def closest_sample(self, orig, samples):
        # pick closest sample based on Mahalanobis distance
        # first calculate means of each sample
        orig_means = orig.iloc[:, 3:].mean().to_frame().transpose()
        sample_means = samples.groupby('sample_id').mean()
        sample_means = sample_means.iloc[:, 3:]

        # calculate inverse of covariance matrix of original data
        covmx = orig.iloc[:, 3:].cov()
        invcovmx = sp.linalg.inv(covmx)

        # vectorize the matrices and join into same data frame
        sample_means['v1'] = sample_means.values.tolist()
        orig_means['v2'] = orig_means.values.tolist()
        sample_means['key'] = 1
        orig_means['key'] = 1
        final = pd.merge(sample_means, orig_means, on='key')
        final = final[['v1', 'v2']]

        # calculate the Mahalanobis distance between original means and each smpl mean
        final['mahal'] = final.apply(lambda x: (mahalanobis(x['v1'], x['v2'], invcovmx)),
                                     axis=1)

        # return the sample closest to population
        closest_index = final['mahal'].idxmin(axis=1)
        return samples[samples['sample_id'] == closest_index]
