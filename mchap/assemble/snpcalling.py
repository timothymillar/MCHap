import numpy as np
from itertools import combinations_with_replacement
from mchap.combinatorics import count_unique_genotypes
from mchap.assemble.likelihood import log_likelihood
from mchap.assemble.prior import log_genotype_prior
from mchap.jitutils import normalise_log_probs

__all__ = ["snp_posterior"]


def snp_posterior(
    reads, position, n_alleles, ploidy, error_rate, inbreeding=0, read_counts=None
):
    """Brute-force the posterior probability across all possible
    genotypes for a single SNP position.

    Parameters
    ----------
    reads : ndarray, int, shape (n_reads, n_base)
        Observed reads with base positions encoded
        as simple integers from 0 to n_nucl and -1
        indicating gaps.
    position : int
        Position of target SNP within reads.
    n_alleles : int
        Number of possible alleles for this SNP.
    ploidy : int
        Ploidy of organism.
    error_rate : float
        Expected base calling error rate.
    inbreeding : float
        Expected inbreeding coefficient of organism.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.

    Returns
    -------
    genotypes : ndarray, int, shape (n_genotypes, ploidy)
        SNP genotypes.
    probabilities : ndarray, float, shape (n_genotypes, )
        Probability of each genotype.

    """
    n_reads, n_pos = reads.shape
    if n_reads == 0:
        # handle no reads
        n_reads = 1
        reads = np.empty((n_reads, n_pos), dtype=np.int8)
        reads[:] = -1

    u_gens = count_unique_genotypes(n_alleles, ploidy)
    genotypes = np.zeros((u_gens, ploidy), dtype=np.int8) - 1
    log_probabilities = np.empty(u_gens, dtype=float)
    log_probabilities[:] = -np.inf

    alleles = np.arange(n_alleles)
    for j, genotype in enumerate(combinations_with_replacement(alleles, ploidy)):
        genotype = np.array(genotype)
        genotypes[j] = genotype
        _, dosage = np.unique(genotype, return_counts=True)
        lprior = log_genotype_prior(dosage, n_alleles, inbreeding=inbreeding)
        # treat as haplotypes with single position
        llk = log_likelihood(
            reads[:, position : position + 1],
            genotype[..., None],
            error_rate=error_rate,
            read_counts=read_counts,
        )

        log_probabilities[j] = lprior + llk

    # normalise
    probabilities = normalise_log_probs(log_probabilities)
    return genotypes, probabilities
