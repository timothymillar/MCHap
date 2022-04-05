import numpy as np
import pytest

from mchap import mset
from mchap.testing import simulate_reads
from mchap.assemble.likelihood import (
    log_likelihood,
    log_likelihood_structural_change,
    log_likelihood_cached,
    new_log_likelihood_cache,
)


def reference_likelihood(reads, genotype, error_rate):
    """Reference implementation of likelihood function."""
    n_nucl = genotype.max() + 1
    n_reads, n_pos = reads.shape
    ploidy, genotype_n_pos = genotype.shape

    # same number of base/variant positions in reads and haplotypes
    assert genotype_n_pos == n_pos

    # convert genotype to tensor of onehot row-vectors
    genotype_matrix = np.zeros((ploidy, n_pos, n_nucl), dtype=float)
    for h in range(ploidy):
        for j in range(n_pos):
            i = genotype[h, j]
            # allele is the index of the 1 in the one-hot vector
            genotype_matrix[h, j, i] = 1

    # convert reads to tensor of probabilities
    read_matrix = np.zeros((n_reads, n_pos, n_nucl), dtype=float)
    for r in range(n_reads):
        for j in range(n_pos):
            i = reads[r, j]
            if i < 0:
                read_matrix[r, j] = np.nan
            else:
                read_matrix[r, j] = error_rate / 3  # assumes SNV
                read_matrix[r, j, i] = 1 - error_rate

    # expand dimentions to use broadcasting
    reads = read_matrix.reshape((n_reads, 1, n_pos, n_nucl))
    genotype = genotype_matrix.reshape((1, ploidy, n_pos, n_nucl))

    # probability of match for each posible allele at each position
    # for every read-haplotype combination
    probs = reads * genotype

    # probability of match at each position for every read-haplotype combination
    # any nans along this axis indicate that ther is a gap in the read at this position
    probs = np.sum(probs, axis=-1)

    # joint probability of match at all positions for every read-haplotype combination
    # nans caused by read gaps should be ignored
    probs = np.nanprod(probs, axis=-1)

    # probability of a read being generated from a genotype is the mean of
    # probabilities of that read matching each haplotype within the genotype
    # this assumes that the reads are generated in equal probortions from each genotype
    probs = np.mean(probs, axis=-1)

    # the probability of a set of reads being generated from a genotype is the
    # joint probability of all reads within the set being generated from that genotype
    prob = np.prod(probs)

    # the likelihood is P(reads|genotype)
    return prob


def test_log_likelihood():
    reads = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=np.int8,
    )

    genotype = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int8)

    query = log_likelihood(reads, genotype, error_rate=0.2)
    answer = np.log(reference_likelihood(reads, genotype, error_rate=0.2))

    assert np.round(query, 10) == np.round(answer, 10)


def test_log_likelihood__read_counts():
    genotype = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    np.random.seed(42)
    reads = simulate_reads(genotype, n_reads=200)
    reads_unique, read_counts = mset.unique_counts(reads)

    expect = log_likelihood(reads, genotype, error_rate=0.0024)
    actual = log_likelihood(
        reads_unique, genotype, error_rate=0.0024, read_counts=read_counts
    )

    # may be slightly off due to float rounding
    np.testing.assert_almost_equal(actual, expect, decimal=10)


@pytest.mark.parametrize(
    "ploidy,n_base,n_reps",
    [
        [2, 3, 10],
        [2, 5, 10],
        [4, 3, 10],
        [4, 8, 10],
    ],
)
def test_log_likelihood__fuzz(ploidy, n_base, n_reps):
    np.random.seed(0)
    for _ in range(n_reps):
        # 'real' genotype
        haplotypes = np.random.randint(2, size=(ploidy, n_base))
        reads = simulate_reads(haplotypes)

        # 'proposed' genotype
        proposed = np.random.randint(2, size=(ploidy, n_base))

        actual = log_likelihood(reads, proposed, error_rate=0.0024)
        expect = np.log(reference_likelihood(reads, proposed, error_rate=0.0024))
        np.testing.assert_almost_equal(actual, expect, decimal=10)


@pytest.mark.parametrize(
    "ploidy,n_base,n_reps",
    [
        [2, 3, 10],
        [2, 5, 10],
        [4, 3, 10],
        [4, 8, 1000],
        [4, 40, 1000],  # beyond max cache size
    ],
)
def test_log_likelihood_cache__fuzz(ploidy, n_base, n_reps):
    np.random.seed(0)
    cache = new_log_likelihood_cache(ploidy, n_base, max_alleles=2, max_size=2**16)
    haplotypes = np.random.randint(2, size=(ploidy, n_base))
    reads = simulate_reads(haplotypes)
    for _ in range(n_reps):
        # 'proposed' genotype
        print(_, cache[0].shape, cache[-1], cache[0][-2])
        proposed = np.random.randint(2, size=(ploidy, n_base))
        expect = log_likelihood(reads, proposed, error_rate=0.0024)
        actual_1, _ = log_likelihood_cached(
            reads, proposed, error_rate=0.0024, cache=None
        )
        actual_2, cache = log_likelihood_cached(
            reads, proposed, error_rate=0.0024, cache=cache
        )
        assert expect == actual_1
        assert expect == actual_2


@pytest.mark.parametrize(
    "reads, genotype, haplotype_indices, interval, final_genotype",
    [
        pytest.param(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            [[0, 0, 0], [0, 0, 1]],
            [0, 1],
            None,
            [[0, 0, 0], [0, 0, 1]],
            id="2x-no-change",
        ),
        pytest.param(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            [[0, 0, 0], [0, 0, 1]],
            [1, 1],
            None,
            [[0, 0, 1], [0, 0, 1]],
            id="2x-overwrite",
        ),
    ],
)
def test_log_likelihood_structural_change(
    reads, genotype, haplotype_indices, interval, final_genotype
):

    reads = np.array(reads, dtype=np.int8)
    genotype = np.array(genotype, dtype=np.int8)
    haplotype_indices = np.array(haplotype_indices, dtype=int)
    final_genotype = np.array(final_genotype, dtype=np.int8)

    query = log_likelihood_structural_change(
        reads, genotype, haplotype_indices, error_rate=0.0024, interval=interval
    )
    answer = log_likelihood(reads, final_genotype, error_rate=0.0024)
    reference = np.log(reference_likelihood(reads, final_genotype, error_rate=0.0024))

    assert query == answer
    assert np.round(query, 10) == np.round(reference, 10)


def test_log_likelihood_structural_change__read_counts():
    genotype = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    np.random.seed(42)
    reads = simulate_reads(genotype, n_reads=200, error_rate=0.0024)
    reads_unique, read_counts = mset.unique_counts(reads)

    interval = (3, 8)
    indices = np.array([0, 2, 2, 3])

    expect = log_likelihood_structural_change(
        reads, genotype, indices, error_rate=0.0024, interval=interval
    )
    actual = log_likelihood_structural_change(
        reads_unique,
        genotype,
        indices,
        error_rate=0.0024,
        interval=interval,
        read_counts=read_counts,
    )

    # may be slightly off due to float rounding
    np.testing.assert_almost_equal(actual, expect, decimal=10)


@pytest.mark.parametrize(
    "ploidy,n_base,interval,indices,n_reps",
    [
        [4, 5, (0, 3), [1, 1, 2, 3], 10],
        [4, 5, (1, 4), [1, 0, 2, 3], 10],
        [4, 8, (0, 3), [1, 2, 2, 3], 10],
        [4, 8, (3, 7), [0, 1, 3, 2], 10],
    ],
)
def test_log_likelihood_structural_change__fuzz(
    ploidy, n_base, interval, indices, n_reps
):
    np.random.seed(0)
    indices = np.array(indices)
    for _ in range(n_reps):
        # 'real' genotype
        haplotypes = np.random.randint(2, size=(ploidy, n_base))
        reads = simulate_reads(haplotypes, error_rate=0.0024)

        # 'proposed' genotype
        proposed = np.random.randint(2, size=(ploidy, n_base))

        # log like of proposed change
        actual = log_likelihood_structural_change(
            reads, proposed, indices, error_rate=0.0024, interval=interval
        )

        # make change
        proposed[:, interval[0] : interval[1]] = proposed[
            indices, interval[0] : interval[1]
        ]
        expect = np.log(reference_likelihood(reads, proposed, error_rate=0.0024))

        np.testing.assert_almost_equal(actual, expect, decimal=10)
