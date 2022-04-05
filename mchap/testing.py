import numpy as np

from mchap.io.util import PFEIFFER_ERROR


def simulate_reads(
    haplotypes,
    n_alleles=None,
    n_reads=20,
    uniform_sample=False,
    errors=True,
    error_rate=PFEIFFER_ERROR,
):
    """Simulate reads from haplotypes for tests.

    Parameters
    ----------
    haplotypes : array_like, int, shape (ploidy, n_base)
        Haplotypes encoded as integer alleles.
    n_alleles : array_like, int, shape (n_base, )
        Number of possible alleles at each base position.
    n_reads : int
        Number of reads to simulate.
    uniform_sample: bool
        If True then an even number of reads is generated
        from each haplotype.
    errors : bool
        If True then reads are resampled based on probabilities
        to introduce errors into the underlying alleles.
    error_rate : float
        Error rate of read calls to use in adition to qual scores.
    qual : tuple, int
        Lower and upper qual scores to randomly assign to base calls.

    Returns
    -------
    reads : ndarray, int, (n_reads, n_base)
        Simulated reads encoded as integers.

    Notes
    -----
    This function is intended only for use in unit tests
    and simulated reads are not intended to be an accurate
    simulation of real molecular data.

    """
    ploidy, n_pos = haplotypes.shape
    if n_alleles is None:
        n_alleles = np.max(haplotypes) + 1
    if isinstance(n_alleles, int):
        n_alleles = np.repeat(n_alleles, n_pos)

    # reads are a sample of haplotypes
    if uniform_sample:
        reads = np.tile(haplotypes, (n_reads // ploidy, 1))
    else:
        reads = haplotypes[np.random.randint(0, ploidy, n_reads)]

    # introduce errors
    if errors:
        error_pos = np.random.choice(
            [True, False], p=[error_rate, 1 - error_rate], size=reads.shape
        )
        n_errors = error_pos.sum()
        error_calls = np.random.randint(0, 4, n_errors)
        reads[error_pos] = error_calls
        reads[reads >= n_alleles] = -1

    return reads


def metropolis_hastings_transitions(transitions, llks, priors):
    """Calculate the transition probabilities based on the
    Metropolis-Hastings algorithm.

    Parameters
    ----------
    transitions : array_like
        A binary square matrix indicating posible transitions among states.
    llk : array_like
        Log-likelihood of each state.
    priors : array_like
        Prior probability of each state.

    Returns
    -------
    probabilities : array_like
        A square transition probability matrix.
    """
    # ratio of likelihoods
    llk_ratios = llks[None, :] - llks[:, None]
    lk_ratios = np.exp(llk_ratios)

    # ratio of priors
    prior_ratios = priors[None, :] / priors[:, None]

    # proposal ratios for detailed balance
    proposal_ratios = transitions.sum(axis=0, keepdims=True) / transitions.sum(
        axis=-1, keepdims=True
    )
    proposal_ratios = 1 / proposal_ratios
    proposal_ratios *= transitions  # zero out illegal transitions

    # Metropolis-Hastings acceptance probabilities
    mh = lk_ratios * proposal_ratios * prior_ratios
    mh[mh > 1] = 1

    # probability of proposing each possible transition
    proposal_probability = transitions / np.sum(transitions, axis=-1, keepdims=True)

    # transition probability is proposal probability * acceptance probability
    mh *= proposal_probability

    # probability of no transition is the remainder
    np.fill_diagonal(mh, 1 - mh.sum(axis=-1))

    return mh
