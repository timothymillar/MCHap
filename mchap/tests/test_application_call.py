import pathlib
import tempfile
import os
import sys
import pytest

from mchap.application.call import program


def reference_path():
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data/simple.fasta"
    return str(path)


@pytest.mark.parametrize(
    "input_vcf,bams,cli_extra,output_vcf",
    [
        (
            "simple.output.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.bam", "simple.sample3.bam"],
            [],
            "simple.output.call.vcf",
        ),
        (
            "simple.output.assemble.vcf",
            [
                "simple.sample1.broken.cram",
                "simple.sample2.broken.cram",
                "simple.sample3.broken.cram",
            ],
            ["--reference", reference_path()],
            "simple.output.call.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            [],
            "simple.output.mixed_depth.call.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            [
                "--report",
                "AFP",
            ],
            "simple.output.mixed_depth.call.frequencies.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            [
                "--report",
                "AOP",
            ],
            "simple.output.mixed_depth.call.occurrence.vcf",
        ),
        (  # test using mock input will low ref allele frequency at first allele
            "mock.input.frequencies.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            [
                "--haplotype-frequencies",
                "AFP",
                "--skip-rare-haplotypes",
                "0.1",
                "--report",
                "AFPRIOR",
                "AFP",
            ],
            "simple.output.mixed_depth.call.frequencies.skiprare.vcf",
        ),
        (  # test using mock input will low ref allele frequency at first allele
            "mock.input.frequencies.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            [
                "--haplotype-frequencies",
                "AFP",
                "--report",
                "AFPRIOR",
                "AFP",
            ],
            "simple.output.mixed_depth.call.frequencies.prior.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            [
                "--report",
                "GL",
                "--base-error-rate",
                "0.0",
                "--use-base-phred-scores",
            ],
            "simple.output.mixed_depth.call.likelihoods.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            [
                "--report",
                "GP",
                "--base-error-rate",
                "0.0",
                "--use-base-phred-scores",
            ],
            "simple.output.mixed_depth.call.posteriors.vcf",
        ),
    ],
)
@pytest.mark.parametrize("n_cores", [1, 2])
def test_Program__run_stdout(input_vcf, bams, cli_extra, output_vcf, n_cores):
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    VCF = str(path / input_vcf)
    BAMS = [str(path / bam) for bam in bams]

    command = (
        [
            "mchap",
            "call",
            "--bam",
        ]
        + BAMS
        + [
            "--ploidy",
            "4",
            "--haplotypes",
            VCF,
            "--mcmc-steps",
            "500",
            "--mcmc-burn",
            "100",
            "--mcmc-seed",
            "11",
            "--cores",
            str(n_cores),
        ]
        + cli_extra
    )

    prog = program.cli(command)

    # capture stdout in file
    _, out_filename = tempfile.mkstemp()
    stdout = sys.stdout
    sys.stdout = open(out_filename, "w")
    prog.run_stdout()
    sys.stdout.close()

    # replace stdout
    sys.stdout = stdout

    # compare output to expected
    with open(out_filename, "r") as f:
        actual = f.readlines()
    with open(str(path / output_vcf), "r") as f:
        expected = f.readlines()

    assert len(actual) == len(expected)

    if n_cores > 1:
        # output may be in different order
        actual.sort()
        expected.sort()

    for act, exp in zip(actual, expected):
        # file paths will make full line differ
        if act.startswith("##commandline"):
            assert exp.startswith("##commandline")
        elif act.startswith("##fileDate"):
            # new date should be greater than test vcf date
            assert exp.startswith("##fileDate")
            assert act > exp
        else:
            assert act == exp

    # cleanup
    os.remove(out_filename)
