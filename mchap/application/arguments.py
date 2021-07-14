import copy
from dataclasses import dataclass


from mchap.io import extract_sample_ids


@dataclass
class Argument(object):
    cli: str
    kwargs: dict

    def add_to(self, parser):
        raise NotImplementedError


@dataclass
class Parameter(Argument):
    def add_to(self, parser):
        """Add parameter to a parser object."""
        kwargs = copy.deepcopy(self.kwargs)
        parser.add_argument(
            self.cli,
            **kwargs,
        )
        return parser


@dataclass
class BooleanFlag(Argument):
    def add_to(self, parser):
        """Add boolean flag to a parser object."""
        dest = self.kwargs["dest"]
        action = self.kwargs["action"]
        if action == "store_true":
            default = False
        elif action == "store_false":
            default = True
        else:
            raise ValueError('Action must be "store_true" or "store_false".')
        parser.set_defaults(**{dest: default})
        parser.add_argument(
            self.cli,
            **self.kwargs,
        )
        return parser


haplotypes = Parameter(
    "--haplotypes",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Tabix indexed VCF file containing haplotype/MNP/SNP variants to be "
            "re-called among input samples."
        ),
    ),
)

region = Parameter(
    "--region",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Specify a single target region with the format contig:start-stop. "
            "This region will be a single variant in the output VCF. "
            "This argument can not be combined with the --targets argument."
        ),
    ),
)


region_id = Parameter(
    "--region-id",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Specify an identifier for the locus specified with the "
            "--region argument. This id will be reported in the output VCF."
        ),
    ),
)

targets = Parameter(
    "--targets",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Bed file containing multiple genomic intervals for haplotype assembly. "
            "First three columns (contig, start, stop) are mandatory. "
            "If present, the fourth column (id) will be used as the variant id in "
            "the output VCF."
            "This argument can not be combined with the --region argument."
        ),
    ),
)

variants = Parameter(
    "--variants",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Tabix indexed VCF file containing SNP variants to be used in "
            "assembly. Assembled haplotypes will only contain the reference and "
            "alternate alleles specified within this file."
        ),
    ),
)

reference = Parameter(
    "--reference",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help="Indexed fasta file containing the reference genome.",
    ),
)

bam = Parameter(
    "--bam",
    dict(
        type=str,
        nargs="*",
        default=[],
        help=(
            "A list of 0 or more bam files. "
            "All samples found within the listed bam files will be genotypes "
            "unless the --sample-list parameter is used."
        ),
    ),
)

bam_list = Parameter(
    "--bam-list",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "A file containing a list of bam file paths (one per line). "
            "This can optionally be used in place of or combined with the --bam "
            "parameter."
        ),
    ),
)

sample_bam = Parameter(
    "--sample-bam",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "A file containing a list of samples with bam file paths. "
            "Each line of the file should be a sample identifier followed "
            "by a tab and then a bam file path. "
            "This can optionally be used in place the --bam and --bam-list "
            "parameters. This is faster than using those parameters when running "
            "many small jobs. "
            "An error will be thrown if a sample is not found within its specified "
            "bam file."
        ),
    ),
)

ploidy = Parameter(
    "--ploidy",
    dict(
        type=int,
        nargs=1,
        default=[2],
        help=(
            "Default ploidy for all samples (default = 2). "
            "This value is used for all samples which are not specified using "
            "the --sample-ploidy parameter"
        ),
    ),
)

sample_ploidy = Parameter(
    "--sample-ploidy",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "A file containing a list of samples with a ploidy value "
            "used to indicate where their ploidy differs from the "
            "default value. Each line should contain a sample identifier "
            "followed by a tab and then an integer ploidy value."
        ),
    ),
)

sample_list = Parameter(
    "--sample-list",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "Optionally specify a file containing a list of samples to "
            "genotype (one sample id per line). "
            "This file also specifies the sample order in the output. "
            "If not specified, all samples in the input bam files will "
            "be genotyped."
        ),
    ),
)

inbreeding = Parameter(
    "--inbreeding",
    dict(
        type=float,
        nargs=1,
        default=[0.0],
        help=(
            "Default inbreeding coefficient for all samples (default = 0.0). "
            "This value is used for all samples which are not specified using "
            "the --sample-inbreeding parameter."
        ),
    ),
)

sample_inbreeding = Parameter(
    "--sample-inbreeding",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "A file containing a list of samples with an inbreeding coefficient "
            "used to indicate where their expected inbreeding coefficient "
            "default value. Each line should contain a sample identifier "
            "followed by a tab and then a inbreeding coefficient value "
            "within the interval [0, 1]."
        ),
    ),
)

base_error_rate = Parameter(
    "--base-error-rate",
    dict(
        nargs=1,
        type=float,
        default=[0.0],
        help=(
            "Expected base error rate of read sequences (default = 0.0). "
            "This is used in addition to base phred-scores by default "
            "however base phred-scores can be ignored using the "
            "--ignore-base-phred-scores flag."
        ),
    ),
)

ignore_base_phred_scores = BooleanFlag(
    "--ignore-base-phred-scores",
    dict(
        dest="ignore_base_phred_scores",
        action="store_true",
        help=(
            "Flag: Ignore base phred-scores as a source of base error rate. "
            "This can improve MCMC speed by allowing for greater de-duplication "
            "of reads however an error rate > 0.0 must be specified with the "
            "--base-error-rate argument."
        ),
    ),
)

haplotype_posterior_threshold = Parameter(
    "--haplotype-posterior-threshold",
    dict(
        type=float,
        nargs=1,
        default=[0.20],
        help=(
            "Posterior probability required for a haplotype to be included in "
            "the output VCF as an alternative allele. "
            "The posterior probability of each haplotype is assessed per individual "
            "and calculated as the probability of that haplotype being present "
            "with one or more copies in that individual."
            "A haplotype is included as an alternate allele if it meets this "
            "posterior probability threshold in at least one individual. "
            "This parameter is the main mechanism to control the number of "
            "alternate alleles in ech VCF record and hence the number of genotypes "
            "assessed when recalculating likelihoods and posterior distributions "
            "(default = 0.20)."
        ),
    ),
)

use_assembly_posteriors = BooleanFlag(
    "--use-assembly-posteriors",
    dict(
        dest="use_assembly_posteriors",
        action="store_true",
        help=(
            "Flag: Use posterior probabilities from each individuals "
            "assembly rather than recomputing posteriors based on the "
            "observed alleles across all samples. "
            "These posterior probabilities will be used to call genotypes "
            ", metrics related to the genotype, and the posterior "
            "distribution (GP field) if specified. "
            "This may lead to less robust genotype calls in the presence "
            "of multi-modality and hence it is recommended to run the "
            "simulation for longer or using parallel-tempering when "
            "using this option. "
            "This option may be more suitable than the default when calling "
            "haplotypes in unrelated individuals. "
        ),
    ),
)

genotype_likelihoods = BooleanFlag(
    "--genotype-likelihoods",
    dict(
        dest="genotype_likelihoods",
        action="store_true",
        help=("Flag: Report genotype likelihoods in the GL VCF field."),
    ),
)

genotype_posteriors = BooleanFlag(
    "--genotype-posteriors",
    dict(
        dest="genotype_posteriors",
        action="store_true",
        help=("Flag: Report genotype posterior probabilities in the GP VCF field."),
    ),
)

mapping_quality = Parameter(
    "--mapping-quality",
    dict(
        nargs=1,
        type=int,
        default=[20],
        help=("Minimum mapping quality of reads used in assembly (default = 20)."),
    ),
)

skip_duplicates = BooleanFlag(
    "--keep-duplicate-reads",
    dict(
        dest="skip_duplicates",
        action="store_false",
        help=(
            "Flag: Use reads marked as duplicates in the assembly "
            "(these are skipped by default)."
        ),
    ),
)

skip_qcfail = BooleanFlag(
    "--keep-qcfail-reads",
    dict(
        dest="skip_qcfail",
        action="store_false",
        help=(
            "Flag: Use reads marked as qcfail in the assembly "
            "(these are skipped by default)."
        ),
    ),
)

skip_supplementary = BooleanFlag(
    "--keep-supplementary-reads",
    dict(
        dest="skip_supplementary",
        action="store_false",
        help=(
            "Flag: Use reads marked as supplementary in the assembly "
            "(these are skipped by default)."
        ),
    ),
)

mcmc_chains = Parameter(
    "--mcmc-chains",
    dict(
        type=int,
        nargs=1,
        default=[2],
        help="Number of independent MCMC chains per assembly (default = 2).",
    ),
)


mcmc_temperatures = Parameter(
    "--mcmc-temperatures",
    dict(
        type=float,
        nargs="*",
        default=[1.0],
        help=(
            "A list of inverse-temperatures to use for parallel tempered chains. "
            "These values must be between 0 and 1 and will automatically be sorted in "
            "ascending order. The cold chain value of 1.0 will be added automatically if "
            "it is not specified."
        ),
    ),
)

sample_mcmc_temperatures = Parameter(
    "--sample-mcmc-temperatures",
    dict(
        type=str,
        nargs=1,
        default=[None],
        help=(
            "A file containing a list of samples with mcmc (inverse) temperatures. "
            "Each line of the file should start with a sample identifier followed by "
            "tab seperated numeric values between 0 and 1. "
            "The number of temperatures specified may vary between samples. "
            "Samples not listed in this file will use the default values specified "
            "with the --mcmc-temperatures argument."
        ),
    ),
)

mcmc_steps = Parameter(
    "--mcmc-steps",
    dict(
        type=int,
        nargs=1,
        default=[1500],
        help="Number of steps to simulate in each MCMC chain (default = 1500).",
    ),
)

mcmc_burn = Parameter(
    "--mcmc-burn",
    dict(
        type=int,
        nargs=1,
        default=[500],
        help="Number of initial steps to discard from each MCMC chain (default = 500).",
    ),
)

mcmc_fix_homozygous = Parameter(
    "--mcmc-fix-homozygous",
    dict(
        type=float,
        nargs=1,
        default=[0.999],
        help=(
            "Fix alleles that are homozygous with a probability greater "
            "than or equal to the specified value (default = 0.999). "
            "The probability of that a variant is homozygous in a sample is "
            "assessed independently for each variant prior to MCMC simulation. "
            'If an allele is "fixed" it is not allowed vary within the MCMC thereby '
            "reducing computational complexity."
        ),
    ),
)

mcmc_seed = Parameter(
    "--mcmc-seed",
    dict(
        type=int,
        nargs=1,
        default=[42],
        help=("Random seed for MCMC (default = 42). "),
    ),
)

mcmc_recombination_step_probability = Parameter(
    "--mcmc-recombination-step-probability",
    dict(
        type=float,
        nargs=1,
        default=[0.5],
        help=(
            "Probability of performing a recombination sub-step during "
            "each step of the MCMC. (default = 0.5)."
        ),
    ),
)

mcmc_partial_dosage_step_probability = Parameter(
    "--mcmc-partial-dosage-step-probability",
    dict(
        type=float,
        nargs=1,
        default=[0.5],
        help=(
            "Probability of performing a within-interval dosage sub-step during "
            "each step of the MCMC. (default = 0.5)."
        ),
    ),
)

mcmc_dosage_step_probability = Parameter(
    "--mcmc-dosage-step-probability",
    dict(
        type=float,
        nargs=1,
        default=[1.0],
        help=(
            "Probability of performing a dosage sub-step during "
            "each step of the MCMC. (default = 1.0)."
        ),
    ),
)

mcmc_chain_incongruence_threshold = Parameter(
    "--mcmc-chain-incongruence-threshold",
    dict(
        type=float,
        nargs=1,
        default=[0.60],
        help=(
            "Posterior phenotype probability threshold for identification of "
            "incongruent posterior modes (default = 0.60)."
        ),
    ),
)

mcmc_llk_cache_threshold = Parameter(
    "--mcmc-llk-cache-threshold",
    dict(
        type=int,
        nargs=1,
        default=[100],
        help=(
            "Threshold for determining whether to cache log-likelihoods "
            "during MCMC to improve performance. This value is computed as "
            "ploidy * variants * unique-reads (default = 100). "
            "If set to 0 then log-likelihoods will be cached for all samples "
            "including those with few observed reads which is inefficient and "
            "can slow the MCMC. "
            "If set to -1 then log-likelihood caching will be disabled for all "
            "samples."
        ),
    ),
)

read_group_field = Parameter(
    "--read-group-field",
    dict(
        nargs=1,
        type=str,
        default=["SM"],
        help=(
            'Read group field to use as sample id (default = "SM"). '
            "The chosen field determines tha sample ids required in other "
            "input files e.g. the --sample-list argument."
        ),
    ),
)

cores = Parameter(
    "--cores",
    dict(
        type=int,
        nargs=1,
        default=[1],
        help=("Number of cpu cores to use (default = 1)."),
    ),
)

DEFAULT_PARSER_ARGUMENTS = [
    bam,
    bam_list,
    sample_bam,
    sample_list,
    ploidy,
    sample_ploidy,
    inbreeding,
    sample_inbreeding,
    base_error_rate,
    ignore_base_phred_scores,
    mapping_quality,
    skip_duplicates,
    skip_qcfail,
    skip_supplementary,
    read_group_field,
    genotype_likelihoods,
    genotype_posteriors,
    cores,
]

CALL_EXACT_PARSER_ARGUMENTS = [
    haplotypes,
] + DEFAULT_PARSER_ARGUMENTS

DEFAULT_MCMC_PARSER_ARGUMENTS = DEFAULT_PARSER_ARGUMENTS + [
    mcmc_chains,
    mcmc_steps,
    mcmc_burn,
    mcmc_seed,
    mcmc_chain_incongruence_threshold,
]

CALL_MCMC_PARSER_ARGUMENTS = [
    haplotypes,
] + DEFAULT_MCMC_PARSER_ARGUMENTS

ASSEMBLE_MCMC_PARSER_ARGUMENTS = (
    [
        region,
        region_id,
        targets,
        variants,
        reference,
    ]
    + DEFAULT_MCMC_PARSER_ARGUMENTS
    + [
        mcmc_fix_homozygous,
        mcmc_llk_cache_threshold,
        mcmc_recombination_step_probability,
        mcmc_dosage_step_probability,
        mcmc_partial_dosage_step_probability,
        mcmc_temperatures,
        sample_mcmc_temperatures,
        haplotype_posterior_threshold,
    ]
)


def parse_sample_bam_paths(arguments):
    """Combine arguments relating to sample bam file specification.

    Parameters
    ----------
    arguments
        Parsed arguments containing some combination of
        arguments for "bam", "bam_list", "sample_bam", and
        "sample_list".

    Returns
    -------
    samples : list
        List of samples.
    sample_bam : dict
        Dict mapping samples to bam paths.
    """
    sample_bams = dict()

    # bam paths
    bams = []
    if hasattr(arguments, "bam"):
        bams = arguments.bam
    if hasattr(arguments, "bam_list"):
        path = arguments.bam_list[0]
        if path:
            with open(arguments.bam_list[0]) as f:
                bams += [line.strip() for line in f.readlines()]
        if len(bams) != len(set(bams)):
            raise IOError("Duplicate input bams")
    sample_bams.update(extract_sample_ids(bams, id=arguments.read_group_field[0]))

    # sample-bams map
    if hasattr(arguments, "sample_bam"):
        # only use values in sample_bam file
        path = arguments.sample_bam[0]
        if path and len(sample_bams) > 0:
            raise IOError(
                "The --sample-bam argument cannot be combined with --bam or --bam-list."
            )
        elif path:
            with open(path) as f:
                for line in f.readlines():
                    sample, bam = line.strip().split("\t")
                    sample_bams[sample] = bam

    # samples list
    if hasattr(arguments, "sample_list"):
        path = arguments.sample_list[0]
        if path:
            with open(path) as f:
                samples = [line.strip() for line in f.readlines()]
            # remove non-listed samples
            sample_bams = {s: sample_bams[s] for s in samples if s in sample_bams}
        else:
            samples = list(sample_bams.keys())
    else:
        samples = list(sample_bams.keys())
    if len(samples) != len(set(samples)):
        raise IOError("Duplicate input samples")

    return samples, sample_bams


def parse_sample_value_map(arguments, samples, default, sample_map, type):
    """Combine arguments specified for a default value and sample-value map file.

    Parameters
    ----------
    arguments
        Parsed arguments containing some the default value argument
        and the optionally the sample-value map file argument.
    samples : list
        List of sample names
    default : str
        Name of argument with default value.
    sample_map : str
        Path of file containing tab-seperated per sample values.
    type : type
        Type of the specified values.

    Returns
    -------
    sample_values : dict
        Dict mapping samples to values.
    """
    sample_value = dict()
    assert hasattr(arguments, default)
    # sample value map
    if hasattr(arguments, sample_map):
        path = getattr(arguments, sample_map)[0]
        if path:
            with open(path) as f:
                for line in f.readlines():
                    sample, value = line.strip().split("\t")
                    sample_value[sample] = type(value)
    # default value
    default_value = getattr(arguments, default)[0]
    for sample in samples:
        if sample in sample_value:
            pass
        else:
            sample_value[sample] = default_value
    return sample_value


def parse_sample_temperatures(arguments, samples):
    """Parse inverse temperatures for MCMC simulation
    with parallel-tempering.

    Parameters
    ----------
    arguments
        Parsed arguments containing the "mcmc_temperatures"
        argument and optionally the "sample_mcmc_temperatures"
        argument.
    samples : list
        List of samples.

    Returns
    -------
    sample_temperatures : dict
        Dict mapping each sample to a list of temperatures (floats).

    """
    assert hasattr(arguments, "mcmc_temperatures")
    # per sample mcmc temperatures
    sample_mcmc_temperatures = dict()
    if hasattr(arguments, "sample_mcmc_temperatures"):
        path = arguments.sample_mcmc_temperatures[0]
        if path:
            with open(path) as f:
                for line in f.readlines():
                    values = line.strip().split("\t")
                    sample = values[0]
                    temps = [float(v) for v in values[1:]]
                    temps.sort()
                    assert temps[0] > 0.0
                    assert temps[-1] <= 1.0
                    if temps[-1] != 1.0:
                        temps.append(1.0)
                    sample_mcmc_temperatures[sample] = temps

    # default mcmc temperatures
    temps = arguments.mcmc_temperatures
    temps.sort()
    assert temps[0] > 0.0
    assert temps[-1] <= 1.0
    if temps[-1] != 1.0:
        temps.append(1.0)
    for sample in samples:
        if sample in sample_mcmc_temperatures:
            pass
        else:
            sample_mcmc_temperatures[sample] = temps
    return sample_mcmc_temperatures


def collect_default_program_arguments(arguments):
    # must have some source of error in reads
    if arguments.ignore_base_phred_scores:
        if arguments.base_error_rate[0] == 0.0:
            raise ValueError(
                "Cannot ignore base phred scores if --base-error-rate is 0"
            )
    # merge sample specific data with defaults
    samples, sample_bams = parse_sample_bam_paths(arguments)
    sample_ploidy = parse_sample_value_map(
        arguments,
        samples,
        default="ploidy",
        sample_map="sample_ploidy",
        type=int,
    )
    sample_inbreeding = parse_sample_value_map(
        arguments,
        samples,
        default="inbreeding",
        sample_map="sample_inbreeding",
        type=float,
    )
    return dict(
        samples=samples,
        sample_bams=sample_bams,
        sample_ploidy=sample_ploidy,
        sample_inbreeding=sample_inbreeding,
        read_group_field=arguments.read_group_field[0],
        base_error_rate=arguments.base_error_rate[0],
        ignore_base_phred_scores=arguments.ignore_base_phred_scores,
        mapping_quality=arguments.mapping_quality[0],
        skip_duplicates=arguments.skip_duplicates,
        skip_qcfail=arguments.skip_qcfail,
        skip_supplementary=arguments.skip_supplementary,
        report_genotype_likelihoods=arguments.genotype_likelihoods,
        report_genotype_posterior=arguments.genotype_posteriors,
        n_cores=arguments.cores[0],
    )


def collect_call_exact_program_arguments(arguments):
    data = collect_default_program_arguments(arguments)
    data["vcf"] = arguments.haplotypes[0]
    data["random_seed"] = None
    return data


def collect_default_mcmc_program_arguments(arguments):
    data = collect_default_program_arguments(arguments)
    data.update(
        dict(
            mcmc_chains=arguments.mcmc_chains[0],
            mcmc_steps=arguments.mcmc_steps[0],
            mcmc_burn=arguments.mcmc_burn[0],
            mcmc_incongruence_threshold=arguments.mcmc_chain_incongruence_threshold[0],
            random_seed=arguments.mcmc_seed[0],
        )
    )
    return data


def collect_call_mcmc_program_arguments(arguments):
    data = collect_default_mcmc_program_arguments(arguments)
    data["vcf"] = arguments.haplotypes[0]
    return data


def collect_assemble_mcmc_program_arguments(arguments):
    # target and regions cant be combined
    if (arguments.targets[0] is not None) and (arguments.region[0] is not None):
        raise ValueError("Cannot combine --targets and --region arguments.")
    data = collect_default_mcmc_program_arguments(arguments)
    sample_mcmc_temperatures = parse_sample_temperatures(
        arguments, samples=data["samples"]
    )
    data.update(
        dict(
            bed=arguments.targets[0],
            vcf=arguments.variants[0],
            ref=arguments.reference[0],
            sample_mcmc_temperatures=sample_mcmc_temperatures,
            region=arguments.region[0],
            region_id=arguments.region_id,
            # mcmc_alpha,
            # mcmc_beta,
            mcmc_fix_homozygous=arguments.mcmc_fix_homozygous[0],
            mcmc_recombination_step_probability=arguments.mcmc_recombination_step_probability[
                0
            ],
            mcmc_partial_dosage_step_probability=arguments.mcmc_partial_dosage_step_probability[
                0
            ],
            mcmc_dosage_step_probability=arguments.mcmc_dosage_step_probability[0],
            mcmc_llk_cache_threshold=arguments.mcmc_llk_cache_threshold[0],
            haplotype_posterior_threshold=arguments.haplotype_posterior_threshold[0],
        )
    )
    return data
