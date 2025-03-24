import numpy as np
from pdb import set_trace

# source: A Sober Look at the Unsupervised Learning of Disentangled Representations and their Evaluation
# (a companion github)
def irs_dic(gen_factors, latents, diff_quantile=0.99):
    """Computes IRS scores of a dataset.

    Assumes no noise in X and crossed generative factors (i.e. one sample per
    combination of gen_factors). Assumes each g_i is an equally probable
    realization of g_i and all g_i are independent.

    Args:
    gen_factors: Numpy array of shape (num samples, num generative factors),
      matrix of ground truth generative factors.
    latents: Numpy array of shape (num samples, num latent dimensions), matrix
      of latent variables.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).

    Returns:
    Dictionary with IRS scores.
    """
    num_gen = gen_factors.shape[1]
    num_lat = latents.shape[1]

    # Compute normalizer.
    max_deviations = np.max(np.abs(latents - latents.mean(axis=0)), axis=0)
    cum_deviations = np.zeros([num_lat, num_gen])
    for i in range(num_gen):
        unique_factors = np.unique(gen_factors[:, i], axis=0)
        assert unique_factors.ndim == 1
        num_distinct_factors = unique_factors.shape[0]
        for k in range(num_distinct_factors):
            # Compute E[Z | g_i].
            match = gen_factors[:, i] == unique_factors[k]
            e_loc = np.mean(latents[match, :], axis=0)

            # Difference of each value within that group of constant g_i to its mean.
            diffs = np.abs(latents[match, :] - e_loc)
    #        print(f'(i,k) = {(i,k)}')
            #if (i,k) == (1,21):
            #    set_trace()
            max_diffs = np.percentile(diffs, q=diff_quantile*100, axis=0)
            cum_deviations[:, i] += max_diffs
        cum_deviations[:, i] /= num_distinct_factors
    # Normalize value of each latent dimension with its maximal deviation.
    normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
    irs_matrix = 1.0 - normalized_deviations
    disentanglement_scores = irs_matrix.max(axis=1)
    if np.sum(max_deviations) > 0.0:
        avg_score = np.average(disentanglement_scores, weights=max_deviations)
    else:
        avg_score = np.mean(disentanglement_scores)

    parents = irs_matrix.argmax(axis=1)
    score_dict = {}
    score_dict["disentanglement_scores"] = disentanglement_scores
    score_dict["avg_score"] = avg_score
    score_dict["parents"] = parents
    score_dict["IRS_matrix"] = irs_matrix
    score_dict["max_deviations"] = max_deviations

    return score_dict
