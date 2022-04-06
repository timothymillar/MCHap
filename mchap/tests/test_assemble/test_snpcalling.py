import numpy as np
import pytest

from mchap.assemble.snpcalling import snp_posterior
from mchap.io.util import PFEIFFER_ERROR


def test_snp_posterior__zero_reads():
    reads = np.empty((0, 2), dtype=np.int8)

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])

    actual_genotypes, actual_probs = snp_posterior(
        reads,
        position=0,
        n_alleles=2,
        ploidy=4,
        inbreeding=0.0,
        error_rate=PFEIFFER_ERROR,
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs)


def test_snp_posterior__zero_reads__inbred():
    reads = np.empty((0, 2), dtype=np.int8)

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([0.1015625, 0.24375, 0.309375, 0.24375, 0.1015625])

    actual_genotypes, actual_probs = snp_posterior(
        reads,
        position=0,
        n_alleles=2,
        ploidy=4,
        inbreeding=0.1,
        error_rate=PFEIFFER_ERROR,
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs, decimal=5)


def test_snp_posterior__gap_reads():

    reads = np.array(
        [
            [
                -1,
                0,
            ],
            [
                -1,
                0,
            ],
            [
                -1,
                0,
            ],
        ],
        dtype=np.int8,
    )

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])

    actual_genotypes, actual_probs = snp_posterior(
        reads,
        position=0,
        n_alleles=2,
        ploidy=4,
        inbreeding=0.0,
        error_rate=PFEIFFER_ERROR,
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs)


def test_snp_posterior__gap_reads__inbred():

    reads = np.array(
        [
            [
                -1,
                0,
            ],
            [
                -1,
                0,
            ],
            [
                -1,
                0,
            ],
        ],
        dtype=np.int8,
    )

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([0.1015625, 0.24375, 0.309375, 0.24375, 0.1015625])

    actual_genotypes, actual_probs = snp_posterior(
        reads,
        position=0,
        n_alleles=2,
        ploidy=4,
        inbreeding=0.1,
        error_rate=PFEIFFER_ERROR,
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs, decimal=5)


@pytest.mark.parametrize(
    "use_read_counts",
    [
        False,
        True,
    ],
)
def test_snp_posterior__homozygous_deep(use_read_counts):

    read = np.array(
        [[0, 0]],
        dtype=float,
    )
    if use_read_counts:
        read_counts = np.array([100])
        reads = np.tile(read, (1, 1))
    else:
        read_counts = None
        reads = np.tile(read, (100, 1))

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    actual_genotypes, actual_probs = snp_posterior(
        reads,
        position=0,
        n_alleles=2,
        ploidy=4,
        inbreeding=0.0,
        read_counts=read_counts,
        error_rate=PFEIFFER_ERROR,
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs, decimal=10)


@pytest.mark.parametrize(
    "use_read_counts",
    [
        False,
        True,
    ],
)
def test_snp_posterior__homozygous_shallow(use_read_counts):

    read = np.array(
        [[0, 0]],
        dtype=float,
    )
    if use_read_counts:
        read_counts = np.array([2])
        reads = np.tile(read, (1, 1))
    else:
        read_counts = None
        reads = np.tile(read, (2, 1))

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array(
        [
            1.9980759475e-01,
            4.4980746626e-01,
            3.0019227676e-01,
            5.0192533741e-02,
            1.2849288638e-07,
        ]
    )

    actual_genotypes, actual_probs = snp_posterior(
        reads,
        position=0,
        n_alleles=2,
        ploidy=4,
        inbreeding=0.0,
        read_counts=read_counts,
        error_rate=PFEIFFER_ERROR,
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs, decimal=10)
