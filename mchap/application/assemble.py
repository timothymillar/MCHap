import sys
import argparse
import numpy as np
from dataclasses import dataclass
import pysam
from itertools import islice
from collections import Counter
import multiprocessing as mp

from mchap.assemble.mcmc.denovo import DenovoMCMC
from mchap.encoding import character, integer
from mchap.io.vcf.util import vcfstr
from mchap.io import \
    read_bed4, \
    extract_sample_ids, \
    extract_read_variants, \
    add_nan_read_if_empty, \
    encode_read_alleles, \
    encode_read_distributions, \
    qual_of_prob, \
    vcf
from mchap.io.biotargetsfile import read_biotargets

import warnings
warnings.simplefilter('error', RuntimeWarning)

@dataclass
class program(object):
    bed: str
    vcf: str
    ref: str
    bams: list
    samples: list
    sample_ploidy: dict
    sample_inbreeding: dict
    call_best_genotype: bool = False
    call_filtered: bool = False
    read_group_field: str = 'SM'
    read_error_rate: float = 0.0
    mcmc_temperatures: tuple = (1.0, )
    mcmc_chains: int = 1
    mcmc_steps: int = 1000
    mcmc_burn: int = 500
    mcmc_alpha: float = 1.0
    mcmc_beta: float = 3.0
    mcmc_fix_homozygous: float = 0.999
    mcmc_recombination_step_probability: float = 0.5
    mcmc_partial_dosage_step_probability: float = 0.5
    mcmc_dosage_step_probability: bool = 1.0
    depth_filter_threshold: float = 5.0
    read_count_filter_threshold: int = 5
    probability_filter_threshold: float = 0.95
    kmer_filter_k: int = 3
    kmer_filter_theshold: float = 0.90
    incongruence_filter_threshold: float = 0.60
    n_cores: int = 1
    precision: int = 3
    random_seed: int = 42
    cli_command: str=None

    @classmethod
    def cli(cls, command):
        """Program initialisation from cli command
        
        e.g. `program.cli(sys.argv)`
        """
        parser = argparse.ArgumentParser(
            'MCMC haplotype assembly'
        )

        parser.add_argument(
            '--targets',
            type=str,
            nargs=1,
            default=[None],
            help=('Bed file containing genomic intervals for haplotype assembly. '
            'First three columns (contig, start, stop) are mandatory. '
            'If present, the fourth column (id) will be used as the variant id in '
            'the output VCF.'),
        )

        parser.add_argument(
            '--variants',
            type=str,
            nargs=1,
            default=[None],
            help=('Tabix indexed VCF file containing SNP variants to be used in '
            'assembly. Assembled haplotypes will only contain the reference and '
            'alternate alleles specified within this file.'),
        )

        parser.add_argument(
            '--reference',
            type=str,
            nargs=1,
            default=[None],
            help='Indexed fasta file containing the reference genome.',
        )

        parser.add_argument(
            '--bam',
            type=str,
            nargs='*',
            default=[],
            help=('A list of 0 or more bam files. '
            'Haplotypes will be assembled for all samples found within all '
            'listed bam files unless the --sample-list parameter is used.'),
        )

        parser.add_argument(
            '--bam-list',
            type=str,
            nargs=1,
            default=[None],
            help=('A file containing a list of bam file paths (one per line). '
            'This can optionally be used in place of or combined with the --bam '
            'parameter.'),
        )

        parser.add_argument(
            '--ploidy',
            type=int,
            nargs=1,
            default=[2],
            help=('Default ploidy for all samples (default = 2). '
            'This value is used for all samples which are not specified using '
            'the --sample-ploidy parameter'),
        )

        parser.add_argument(
            '--sample-ploidy',
            type=str,
            nargs=1,
            default=[None],
            help=(
                'A file containing a list of samples with a ploidy value '
                'used to indicate where their ploidy differs from the '
                'default value. Each line should contain a sample identifier '
                'followed by a tab and then an integer ploidy value.'
            ),
        )

        parser.add_argument(
            '--sample-list',
            type=str,
            nargs=1,
            default=[None],
            help=('Optionally specify a file containing a list of samples to '
                'haplotype (one sample id per line). '
                'This file also specifies the sample order in the output. '
                'If not specified, all samples in the input bam files will '
                'be haplotyped.'),
        )

        parser.add_argument(
            '--inbreeding',
            type=float,
            nargs=1,
            default=[0.0],
            help=('Default inbreeding coefficient for all samples (default = 0.0). '
            'This value is used for all samples which are not specified using '
            'the --sample-inbreeding parameter'),
        )

        parser.add_argument(
            '--sample-inbreeding',
            type=str,
            nargs=1,
            default=[None],
            help=(
                'A file containing a list of samples with an inbreeding coefficient '
                'used to indicate where their expected inbreeding coefficient '
                'default value. Each line should contain a sample identifier '
                'followed by a tab and then a inbreeding coefficient value '
                'within the interval [0, 1]'
            ),
        )

        parser.add_argument(
            '--error-rate',
            nargs=1,
            type=float,
            default=[0.0],
            help=('Expected base-call error rate of sequences '
            'in addition to base phred scores (default = 0.0). '
            'By default only the phred score of each base call is used to '
            'calculate its probability of an incorrect call. '
            'The --error-rate value is added to that probability.')
        )

        parser.set_defaults(call_best_genotype=False)
        parser.add_argument(
            '--best-genotype',
            dest='call_best_genotype',
            action='store_true',
            help=('Flag: allways call the best supported complete genotype '
            'within a called phenotype. This may result in calling genotypes '
            'with a posterior probability less than --filter-probability '
            'however a phenotype probability of >= --filter-probability '
            'is still required.')
        )

        parser.set_defaults(call_filtered=False)
        parser.add_argument(
            '--call-filtered',
            dest='call_filtered',
            action='store_true',
            help=('Flag: include genotype calls for filtered samples. '
            'Sample filter tags will still indicate samples that have '
            'been filtered. '
            'WARNING: this can result in a large VCF file with '
            'un-informative genotype calls.')
        )

        parser.add_argument(
            '--mcmc-chains',
            type=int,
            nargs=1,
            default=[2],
            help='Number of independent MCMC chains per assembly (default = 2).'
        )

        parser.add_argument(
            '--mcmc-temperatures',
            type=float,
            nargs='*',
            default=[1.0],
            help=('A list of inverse-temperatures to use for parallel tempered chains. '
            'These values must be between 0 and 1 and will automatically be sorted in '
            'ascending order. The cold chain value of 1.0 will be added automatically if '
            'it is not specified.')
        )

        parser.add_argument(
            '--mcmc-steps',
            type=int,
            nargs=1,
            default=[1000],
            help='Number of steps to simulate in each MCMC chain (default = 1000).'
        )

        parser.add_argument(
            '--mcmc-burn',
            type=int,
            nargs=1,
            default=[500],
            help='Number of initial steps to discard from each MCMC chain (default = 500).'
        )

        parser.add_argument(
            '--mcmc-fix-homozygous',
            type=float,
            nargs=1,
            default=[0.999],
            help=('Fix alleles that are homozygous with a probability greater '
            'than or equal to the specified value (default = 0.999). '
            'The probability of that a variant is homozygous in a sample is '
            'assessed independently for each variant prior to MCMC simulation. '
            'If an allele is "fixed" it is not allowed vary within the MCMC thereby '
            'reducing computational complexity.')
        )

        parser.add_argument(
            '--mcmc-seed',
            type=int,
            nargs=1,
            default=[42],
            help=('Random seed for MCMC (default = 42). ')
        )

        parser.add_argument(
            '--mcmc-recombination-step-probability',
            type=float,
            nargs=1,
            default=[0.5],
            help=('Probability of performing a recombination sub-step during '
            'each step of the MCMC. (default = 0.5).')
        )

        parser.add_argument(
            '--mcmc-partial-dosage-step-probability',
            type=float,
            nargs=1,
            default=[0.5],
            help=('Probability of performing a within-interval dosage sub-step during '
            'each step of the MCMC. (default = 0.5).')
        )

        parser.add_argument(
            '--mcmc-dosage-step-probability',
            type=float,
            nargs=1,
            default=[1.0],
            help=('Probability of performing a dosage sub-step during '
            'each step of the MCMC. (default = 1.0).')
        )

        parser.add_argument(
            '--filter-depth',
            type=float,
            nargs=1,
            default=[5.0],
            help=('Minimum sample read depth required to include an assembly '
            'result (default = 5.0). '
            'Read depth is measured as the mean of read depth across each '
            'variable position.')
        )

        parser.add_argument(
            '--filter-read-count',
            type=float,
            nargs=1,
            default=[5.0],
            help=('Minimum number of read (pairs) required within a target '
            'interval in order to include an assembly result (default = 5).')
        )

        parser.add_argument(
            '--filter-probability',
            type=float,
            nargs=1,
            default=[0.95],
            help=('Minimum sample assembly posterior probability required to call '
            'a phenotype i.e. a set of unique haplotypes of unknown dosage '
            '(default = 0.95). '
            'Genotype dosage will be called or partially called if it also exceeds '
            'this threshold. '
            'See also the --best-genotype flag.')
        )

        parser.add_argument(
            '--filter-kmer-k',
            type=int,
            nargs=1,
            default=[3],
            help=('Size of variant kmer used to filter assembly results (default = 3).')
        )

        parser.add_argument(
            '--filter-kmer',
            type=float,
            nargs=1,
            default=[0.90],
            help=('Minimum kmer representation required at each position in assembly '
            'results (default = 0.90).')
        )

        parser.add_argument(
            '--filter-chain-incongruence',
            type=float,
            nargs=1,
            default=[0.60],
            help=('Posterior phenotype probability threshold for identification of '
            'incongruent posterior modes (default = 0.60).')
        )

        parser.add_argument(
            '--read-group-field',
            nargs=1,
            type=str,
            default=['SM'],
            help=('Read group field to use as sample id (default = "SM"). '
            'The chosen field determines tha sample ids required in other '
            'input files e.g. the --sample-list argument.')
        )

        parser.add_argument(
            '--cores',
            type=int,
            nargs=1,
            default=[2],
            help=('Number of cpu cores to use (default = 2).')
        )

        if len(command) < 3:
            parser.print_help()
            sys.exit(1)
        args = parser.parse_args(command[2:])

        # bam paths
        bams = args.bam
        if args.bam_list[0]:
            with open(args.bam_list[0]) as f:
                bams += [line.strip() for line in f.readlines()]
        if len(bams) != len(set(bams)):
            raise IOError('Duplicate input bams')

        # samples
        if args.sample_list[0]:
            with open(args.sample_list[0]) as f:
                samples = [line.strip() for line in f.readlines()]
        else:
            # read samples from bam headers
            samples = list(extract_sample_ids(bams, id=args.read_group_field[0]).keys())
        if len(samples) != len(set(samples)):
            raise IOError('Duplicate input samples')
        
        # sample ploidy where it differs from default
        sample_ploidy = dict()
        if args.sample_ploidy[0]:
            with open(args.sample_ploidy[0]) as f:
                for line in f.readlines():
                    sample, ploidy = line.strip().split('\t')
                    sample_ploidy[sample] = int(ploidy)

        # default ploidy
        for sample in samples:
            if sample in sample_ploidy:
                pass
            else:
                sample_ploidy[sample] = args.ploidy[0]

        # sample inbreeding where it differs from default
        sample_inbreeding = dict()
        if args.sample_inbreeding[0]:
            with open(args.sample_inbreeding[0]) as f:
                for line in f.readlines():
                    sample, inbreeding = line.strip().split('\t')
                    sample_ploidy[sample] = float(inbreeding)

        # default inbreeding
        for sample in samples:
            if sample in sample_inbreeding:
                pass
            else:
                sample_inbreeding[sample] = args.inbreeding[0]

        # add cold chain temperature if not present and sort
        temperatures = args.mcmc_temperatures
        temperatures.sort()
        assert temperatures[0] >= 0.0
        assert temperatures[-1] <= 1.0
        if temperatures[-1] != 1.0:
            temperatures.append(1.0)

        return cls(
            bed=args.targets[0],
            vcf=args.variants[0],
            ref=args.reference[0],
            bams=bams,
            samples=samples,
            sample_ploidy=sample_ploidy,
            sample_inbreeding=sample_inbreeding,
            call_best_genotype=args.call_best_genotype,
            call_filtered=args.call_filtered,
            read_group_field=args.read_group_field[0],
            read_error_rate=args.error_rate[0],
            mcmc_chains=args.mcmc_chains[0],
            mcmc_temperatures=tuple(temperatures),
            mcmc_steps=args.mcmc_steps[0],
            mcmc_burn=args.mcmc_burn[0],
            #mcmc_alpha,
            #mcmc_beta,
            mcmc_fix_homozygous=args.mcmc_fix_homozygous[0],
            mcmc_recombination_step_probability=args.mcmc_recombination_step_probability[0],
            mcmc_partial_dosage_step_probability=args.mcmc_partial_dosage_step_probability[0],
            mcmc_dosage_step_probability=args.mcmc_dosage_step_probability[0],
            depth_filter_threshold=args.filter_depth[0],
            read_count_filter_threshold=args.filter_read_count[0],
            probability_filter_threshold=args.filter_probability[0],
            kmer_filter_k=args.filter_kmer_k[0],
            kmer_filter_theshold=args.filter_kmer[0],
            incongruence_filter_threshold=args.filter_chain_incongruence[0],
            n_cores=args.cores[0],
            cli_command=command,
            random_seed=args.mcmc_seed[0]
        )

    def loci(self):
        bed = read_bed4(self.bed)
        for b in bed:
            yield b.set_sequence(self.ref).set_variants(self.vcf)

    def header(self):

        # io
        samples = self.samples

        with pysam.Fastafile(self.ref) as fasta:
            contigs = tuple(
                vcf.ContigHeader(c, l)
                for c, l in zip(fasta.references, fasta.lengths)
            )

        # define vcf template
        meta_fields=(
            vcf.headermeta.fileformat('v4.3'),
            vcf.headermeta.filedate(),
            vcf.headermeta.source(),
            vcf.headermeta.phasing('None'),
            vcf.headermeta.commandline(self.cli_command),
            vcf.headermeta.randomseed(self.random_seed),
        )

        filters=(
            vcf.filters.SamplePassFilter(),
            vcf.filters.SampleKmerFilter(self.kmer_filter_k, self.kmer_filter_theshold),
            vcf.filters.SampleDepthFilter(self.depth_filter_threshold),
            vcf.filters.SampleReadCountFilter(self.read_count_filter_threshold),
            vcf.filters.SamplePhenotypeProbabilityFilter(self.probability_filter_threshold),
            vcf.filters.SampleChainPhenotypeIncongruenceFilter(self.incongruence_filter_threshold),
            vcf.filters.SampleChainPhenotypeCNVFilter(self.incongruence_filter_threshold),
        )

        info_fields=(
            vcf.infofields.AN,
            vcf.infofields.AC,
            vcf.infofields.NS,
            vcf.infofields.END,
            vcf.infofields.SNVPOS,
        )

        format_fields=(
            vcf.formatfields.GT,
            vcf.formatfields.GQ,
            vcf.formatfields.PHQ,
            vcf.formatfields.DP,
            vcf.formatfields.RCOUNT,
            vcf.formatfields.RCALLS,
            vcf.formatfields.MEC,
            vcf.formatfields.FT,
            vcf.formatfields.GPM,
            vcf.formatfields.PPM, 
            vcf.formatfields.MPED,
        )

        vcf_header = vcf.VCFHeader(
            meta=meta_fields,
            contigs=contigs,
            filters=filters,
            info_fields=info_fields,
            format_fields=format_fields,
            samples=tuple(samples),
        )

        return vcf_header

    def _assemble_locus(self, header, sample_bams, locus):
        # label sample filters
        kmer_filter = header.filters[1]
        depth_filter = header.filters[2]
        count_filter = header.filters[3]
        prob_filter = header.filters[4]
        incongruence_filter = header.filters[5]
        cnv_filter = header.filters[6]

        # arrays of sample data in order
        n_samples = len(self.samples)
        sample_read_calls = np.empty(n_samples, dtype='O')
        sample_genotype = np.empty(n_samples, dtype='O')
        sample_phenotype_dist = np.empty(n_samples, dtype='O')
        sample_RCOUNT = np.empty(n_samples, dtype='O')
        sample_DP = np.empty(n_samples, dtype='O')
        sample_FT = np.empty(n_samples, dtype='O')
        sample_GPM = np.empty(n_samples, dtype='O')
        sample_PPM = np.empty(n_samples, dtype='O')
        sample_RCALLS = np.empty(n_samples, dtype='O')
        sample_GQ = np.empty(n_samples, dtype='O')
        sample_PHQ = np.empty(n_samples, dtype='O')
        sample_MEC = np.empty(n_samples, dtype='O')

        # loop over samples
        for i, sample in enumerate(self.samples):

            # path to bam for this sample
            path = sample_bams[sample]

            # extract read data
            read_chars, read_quals = extract_read_variants(locus, path, id=self.read_group_field)[sample]

            # get read stats
            read_count = read_chars.shape[0]
            sample_RCOUNT[i] = read_count
            read_variant_depth = character.depth(read_chars)
            if len(read_variant_depth) == 0:
                # no variants to score depth
                sample_DP[i] = np.nan
            else:
                sample_DP[i] = np.round(np.mean(read_variant_depth))

            # if no reads insert nan read TODO: more elegant solution
            read_chars, read_quals = add_nan_read_if_empty(locus, read_chars, read_quals)

            # encode reads as alleles and probabilities
            read_calls = encode_read_alleles(locus, read_chars)
            sample_read_calls[i] = read_calls
            read_dists = encode_read_distributions(
                locus, 
                read_calls, 
                read_quals, 
                error_rate=self.read_error_rate, 
            )

            # assemble haplotypes
            trace = DenovoMCMC(
                ploidy=self.sample_ploidy[sample],
                inbreeding=self.sample_inbreeding[sample],
                steps=self.mcmc_steps,
                chains=self.mcmc_chains,
                fix_homozygous=self.mcmc_fix_homozygous,
                recombination_step_probability=self.mcmc_recombination_step_probability,
                partial_dosage_step_probability=self.mcmc_partial_dosage_step_probability,
                dosage_step_probability=self.mcmc_dosage_step_probability,
                temperatures=self.mcmc_temperatures,
                random_seed=self.random_seed,
            ).fit(read_dists).burn(self.mcmc_burn)

            # posterior mode phenotype is a collection of genotypes
            phenotype = trace.posterior().mode_phenotype()

            # call genotype (array(ploidy, vars), probs)
            if self.call_best_genotype:
                genotype = phenotype.mode_genotype()
            else:
                genotype = phenotype.call_phenotype(self.probability_filter_threshold)

            # per chain modes for QC
            chain_modes = [dist.mode_phenotype() for dist in trace.chain_posteriors()]

            # apply filters
            filterset = vcf.filters.FilterCallSet((
                prob_filter(phenotype.probabilities.sum()),
                depth_filter(read_variant_depth),
                count_filter(read_count),
                kmer_filter(read_calls, genotype[0]),
                incongruence_filter(chain_modes),
                cnv_filter(chain_modes),
            ))
            sample_FT[i] = filterset

            # store sample format calls
            sample_GPM[i] = np.round(genotype[1], self.precision)
            sample_PPM[i] = np.round(phenotype.probabilities.sum(), self.precision)
            sample_RCALLS[i] = np.sum(read_calls >= 0)
            sample_GQ[i] = qual_of_prob(genotype[1])
            sample_PHQ[i] = qual_of_prob(phenotype.probabilities.sum())
            sample_MEC[i] =  integer.minimum_error_correction(read_calls, genotype[0]).sum()

            # Null out the genotype and phenotype arrays
            if (not self.call_filtered) and filterset.failed:
                genotype[0][:] = -1
                phenotype.genotypes[:] = -1
            
            # store genotype and phenotype
            sample_genotype[i] = genotype[0]
            sample_phenotype_dist[i] = phenotype

        # labeling alleles
        vcf_haplotypes, vcf_haplotype_counts = vcf_sort_haplotypes(sample_genotype)
        vcf_alleles = locus.format_haplotypes(vcf_haplotypes)
        vcf_REF = vcf_alleles[0]
        vcf_ALTS = vcf_alleles[1:]

        # vcf info fields
        info_AC = vcf_haplotype_counts[1:]
        info_AN = np.sum(vcf_haplotype_counts > 0)
        info_NS = len(self.samples)
        if not self.call_filtered:
            info_NS -= sum(ft.failed for ft in sample_FT)
        info_END = locus.stop
        info_SNVPOS = np.subtract(locus.positions, locus.start) + 1

        # additional sample data requiring sorted alleles
        sample_GT = np.empty(n_samples, dtype='O')
        sample_MPED = np.empty(n_samples, dtype='O')
        for i, sample in enumerate(self.samples):
            sample_GT[i] = vcf_genotype_string(sample_genotype[i], vcf_haplotypes)
            dosage_expected = vcf_expected_dosage(
                sample_phenotype_dist[i].genotypes,
                sample_phenotype_dist[i].probabilities,
                vcf_haplotypes,
            )
            sample_MPED[i] = np.round(dosage_expected, self.precision)

        # vcf line formating
        vcf_INFO = format_vcf_info_field(
            AN=info_AN,
            AC=info_AC,
            NS=info_NS,
            END=info_END,
            SNVPOS=info_SNVPOS,
        )

        vcf_FORMAT = format_vcf_sample_field(
            GT=sample_GT,
            GQ=sample_GQ,
            PHQ=sample_PHQ,
            DP=sample_DP,
            RCOUNT=sample_RCOUNT,
            RCALLS=sample_RCALLS,
            MEC=sample_MEC,
            FT=sample_FT,
            GPM=sample_GPM,
            PPM=sample_PPM,
            MPED=sample_MPED,
        )

        return format_vcf_line(
            chrom=locus.contig, 
            pos=locus.start + 1,  # 0-based BED to 1-based VCF 
            id=locus.name, 
            ref=vcf_REF, 
            alt=vcf_ALTS, 
            qual=None, 
            filter=None, 
            info=vcf_INFO, 
            format=vcf_FORMAT,
        )

    def run(self):
        header = self.header()
        sample_bams = extract_sample_ids(self.bams, id=self.read_group_field)
        pool = mp.Pool(self.n_cores)
        jobs = ((header, sample_bams, locus) for locus in self.loci())
        records = pool.starmap(self._assemble_locus, jobs)
        return list(header.lines()) + records

    def _worker(self, header, sample_bams, locus, queue):
        line = str(self._assemble_locus(header, sample_bams, locus))
        queue.put(line)
        return line

    def _writer(self, queue):
        while True:
            line = queue.get()
            if line == 'KILL':
                break
            sys.stdout.write(line + '\n')
            sys.stdout.flush()

    def _run_stdout_single_core(self):
        header = self.header()
        sample_bams = extract_sample_ids(self.bams, id=self.read_group_field)
        for line in header.lines():
            sys.stdout.write(line + '\n')
        for locus in self.loci():
            line = self._assemble_locus(header, sample_bams, locus)
            sys.stdout.write(line + '\n')

    def run_stdout(self):
        if self.n_cores <= 1:
            self._run_stdout_single_core()

        header = self.header()
        sample_bams = extract_sample_ids(self.bams, id=self.read_group_field)

        for line in header.lines():
            sys.stdout.write(line + '\n')
        sys.stdout.flush()

        manager = mp.Manager()
        queue = manager.Queue()
        pool = mp.Pool(self.n_cores)

        # start writer process
        writer = pool.apply_async(self._writer, (queue,))

        jobs = []
        for locus in self.loci():
            job = pool.apply_async(self._worker, (header, sample_bams, locus, queue))
            jobs.append(job)

        for job in jobs:
            job.get()

        queue.put('KILL')
        pool.close()
        pool.join()


def vcf_sort_haplotypes(genotypes, dtype=np.int8):
    """Sort unique haplotypes from multiple genotype arrays from
    most to least frequent with the reference allele first

    Parameters
    ----------
    genotypes : list
        List of ndarrays each with shape (ploidy, n_positions).
    dtype : type
        Numpy dtype for returned array.
    
    Returns
    -------
    haplotypes : ndarray, int, shape (n_haplotypes, n_positions)
        Unique haplotypes sorted by frequency with reference allele first.
    counts : ndarray, int, shape (n_haplotypes, )
        Count of each haplotype

    """
    haplotypes = np.concatenate(genotypes)
    _, n_pos = haplotypes.shape
    
    # count observed haps
    counts = Counter(tuple(hap) for hap in haplotypes)
    
    # ref and null haps are special values
    ref = (0, ) * n_pos
    null = (-1, ) * n_pos
    
    # remove null haps from count if present
    if null in counts:
        _ = counts.pop(null)
    
    # seperate ref count
    if ref not in counts:
        ref_count = 0
    else:
        ref_count = counts.pop(ref)

    # order by frequency then insert ref first
    pairs = counts.most_common()
    pairs = [(ref, ref_count)] + pairs
    
    # convert back to arrays
    haplotypes = np.array([a for a, _ in pairs], dtype=dtype)
    counts = np.array([c for _, c in pairs])
    
    return haplotypes, counts


def vcf_genotype_string(genotype, haplotypes):
    """Convert a genotype array to a VCF genotype (GT) string

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_positions)
        Integer encoded genotype.
    haplotypes : ndarray, int, shape (n_haplotypes, n_positions)
        Unique haplotypes sorted by frequency with reference allele first.

    Returns
    -------
    string : str
        VCF genotype string.
    """
    assert genotype.dtype == haplotypes.dtype
    labels = {h.tobytes(): i for i, h in enumerate(haplotypes)}
    alleles = [labels.get(h.tobytes(), -1) for h in genotype]
    alleles.sort()
    chars = [str(a) for a in alleles if a >= 0]
    chars += ['.' for a in alleles if a < 0]
    return '/'.join(chars)


def vcf_expected_dosage(genotypes, probabilities, haplotypes):
    """Calculate the expected floating point dosage based on a phenotype distribution.

    Parameters
    ----------
    genotypes : ndarray, int, shape (n_genotypes, ploidy, n_positions)
        Integer encoded genotypes containing the same alleles in different dosage.
    probabilities : ndarray, int, shape (n_genotypes, ).
        Probability of each genotype.

    Returns
    -------
    expected_dosage: ndarray, float, shape (n_alleles, )
        Expected count of each allele.
    """
    assert genotypes.dtype == haplotypes.dtype

    # if all alleles are null then no dosage
    if np.all(genotypes < 0):
        return np.array([np.nan])

    # label genotype alleles
    labels = {h.tobytes(): i for i, h in enumerate(haplotypes)}
    alleles = [[labels[h.tobytes()] for h in g] for g in genotypes]

    # values for sorting genotypes, this works because all
    # genotypes share the same alleles in different dosage
    vals = [np.sum(g) for g in alleles]
    
    # normalised probability of each dosage
    probs = probabilities[np.argsort(vals)]
    probs /= probs.sum()
    
    # per genotype allele counts
    uniques, counts = zip(*[np.unique(g, return_counts=True) for g in alleles])
    counts = np.array(counts)
    
    # assert all dosage variants contain same alleles
    for i in range(1, len(uniques)):
        np.testing.assert_array_equal(uniques[0], uniques[i])
    
    # expectation of dosage
    expected = np.sum(counts * probs[:, None], axis=0)
    
    return expected


def format_vcf_info_field(**kwargs):
    """Format key-value pairs into a VCF info field.

    Parameters
    ----------
    kwargs
        Key value pairs of info field codes to values.
    
    Returns
    -------
    string : str
        VCF info field.
    """
    parts = ['{}={}'.format(k, vcfstr(v)) for k, v in kwargs.items()]
    return ';'.join(parts)


def format_vcf_sample_field(**kwargs):
    """Format key-value pairs into a VCF format field.

    Parameters
    ----------
    kwargs
        Key value pairs of info field codes to arrays of values per sample.
    
    Returns
    -------
    string : str
        VCF format and sample columns.
    """
    fields, arrays = zip(*kwargs.items())
    fields = ':'.join(fields)
    lengths = np.array([len(a) for a in arrays])
    length = lengths[0]
    assert np.all(lengths == length)
    sample_data = np.empty(length, dtype='O')
    for i in range(length):
        sample_data[i] = ':'.join((vcfstr(a[i]) for a in arrays))
    sample_data = '\t'.join(sample_data)
    return '{}\t{}'.format(fields, sample_data)


def format_vcf_line(
    *, 
    chrom=None, 
    pos=None, 
    id=None, 
    ref=None, 
    alt=None, 
    qual=None, 
    filter=None, 
    info=None, 
    format=None,
):
    """Format a VCF record line.

    Parameters
    ----------
    chrom : str
        Variant chromosome or contig.
    pos : int
        Variant position.
    id : str
        Variant ID.
    ref : str
        Reference allele.
    alt : list, str
        Alternate alleles.
    qual : int
        Variant quality.
    filter : str
        Variant filter codes.
    info : str
        Variant INFO string.
    format : str
        Variant format codes and sample values.

    Returns
    -------
    line : str
        VCF record line.
    """
    fields = [chrom, pos, id, ref, alt, qual, filter, info, format]
    return '\t'.join(vcfstr(f) for f in fields)
