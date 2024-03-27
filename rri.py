import re
import subprocess
from PIL import Image
from io import StringIO
import random
import shutil


import pandas as pd
import RNA
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import infrared.rna

import rrikindp

PARAMS = {
    "SD": "AGGAGC",  # Shine-Dalgarno motif
    "SD_pos": 139,  # Shine-Dalgarno at pos 189-194 in rPos sequence (1-index)
    "AUG_pos": 151,  # Start of AUG sequence (1-index)
    "BP_SPAN": 200,
}
"""Some predefined parameters to avoid passing different value to functions
The param values are used if corresponding function argument is set to None
All indices are one based
"""

def dGopen(seq, region, constraint=None, max_bp_span=None):
    """Computes the opening energy for region [i,j] (1-index) optionally while also constraining a second region to be unstructured

    Args:
        seq: RNA sequence of interest
        region: region (i, j)
        constraint: constraint region (i, j)
        max_bp_span: Maximum basepair span
    """
    md = RNA.md()
    # md.uniq_ML = 1
    # md.compute_bpp = 0
    span = int(max_bp_span) if max_bp_span is not None else PARAMS["BP_SPAN"]
    if span > 0:
        md.max_bp_span = span
    # fold compound and constraint
    cons = "." * len(seq)
    if constraint:
        i = constraint[0] - 1
        j = constraint[1]
        cons = cons[:i] + "x" * (j - i) + cons[j:]
    fc = RNA.fold_compound(seq, md)
    fc.hc_add_from_db(cons, RNA.CONSTRAINT_DB_DEFAULT)
    # print(cons)
    pf = fc.pf()[1]
    i = region[0] - 1
    j = region[1]
    cons = cons[:i] + "x" * (j - i) + cons[j:]
    fc.hc_add_from_db(cons, RNA.CONSTRAINT_DB_DEFAULT)
    pf_cons = fc.pf()[1]
    # print(cons)

    return pf_cons - pf


class EnergyLandscape(rrikindp.EM):
    def __init__(
        self,
        seq_a,
        seq_b,
        bps,
        id_a="id_a",
        id_b="id_b",
        str_a="",
        str_b="str_b",
        accessibility_from_pf=True,
        dangles=True,
        temperature=37.0,
    ):
        self.rna_a = rrikindp.RnaSequence(id_a, seq_a)
        self.rna_b = rrikindp.RnaSequence(id_b, seq_b)
        self.interaction = rrikindp.Interaction(self.rna_a, self.rna_b)
        self.interaction.basePairs = bps
        self.interaction_length = len(bps)
        super().__init__(
            self.interaction,
            seq_a,
            seq_b,
            str_a,
            str_b,
            accessibility_from_pf,
            dangles,
            temperature,
        )

    def get_full_E(self):
        return self.get_e(0, self.interaction_length - 1) / 100

    def get_full_hybridE(self):
        return self.get_hybride_e(0, self.interaction_length - 1) / 100

    def get_full_ED(self):
        return self.get_accessibility(0, self.interaction_length - 1) / 100

    def get_full_ED1(self):
        return self.get_ED1(0, self.interaction_length - 1) / 100

    def get_full_ED2(self):
        return self.get_ED2(0, self.interaction_length - 1) / 100

    def get_seed_Es(self, seed_length):
        seed_energies = [
            self.get_e(k, k + seed_length - 1) / 100
            for k in range(0, self.interaction_length - seed_length + 1)
        ]
        return seed_energies

    def get_seed_EDs(self, seed_length):
        seed_unpairingE1 = [
            self.get_accessibility(k, k + seed_length - 1) / 100
            for k in range(0, self.interaction_length - seed_length + 1)
        ]
        return seed_unpairingE

    def get_seed_ED1s(self, seed_length):
        seed_unpairingE1 = [
            self.get_ED1(k, k + seed_length - 1) / 100
            for k in range(0, self.interaction_length - seed_length + 1)
        ]
        return seed_unpairingE1

    def get_seed_ED2s(self, seed_length):
        seed_unpairingE1 = [
            self.get_ED2(k, k + seed_length - 1) / 100
            for k in range(0, self.interaction_length - seed_length + 1)
        ]
        return seed_unpairingE2

    def get_min_barrier_Es(self, seed_length):
        barriers = [
            self.get_minBarrier(k, self.interaction_length, seed_length) / 100
            for k in range(0, self.interaction_length - seed_length + 1)
        ]
        return barriers

    def get_states(self):
        states = []
        for i in range(0, self.interaction_length):
            for j in range(i, self.interaction_length):
                energy = self.get_e(i, j) / 100
                states.append([i, j, energy])
        return states

    def get_seed_barrier_state(self, seed_length):
        n = self.interaction_length
        states = self.get_states()
        min_barrier_energies = self.get_min_barrier_Es(seed_length)
        min_barrier = min(min_barrier_energies)
        seed_k = min_barrier_energies.index(min_barrier)
        seed_l = seed_k + seed_length - 1
        barrier_k = -1
        barrier_l = -1
        for s in states:
            if s[2] == min_barrier and s[0] <= seed_k and s[1] >= seed_l:
                barrier_k = s[0]
                barrier_l = s[1]

        return (seed_k+1, seed_l+1), (barrier_k+1, barrier_l+1)


def _get_rec(x, y, n):
    """Helper function to decide the rectangle position
    """
    z = 1
    if y == 0:
        y = 0.1
    if x == n:
        z = 0.9
    return y, z

def plot_landscape(
    states,  # as generated by RRIkinDP
    figure_path=None,  # output file; if None a default path is set
    energy="E",  # free energy to plot: E, Ehybrid, ED1, ED2 or ED
    structure=None,  # interaction as string as returned by RRIkinDP.utilities.get_string_representations()
    annotate=None,  # plot energy values per structure; True/False; default: True if less than 16 bps in interaction
    e_min=None,  # min value represented by color scale (kcal/mol)
    e_max=None,  # max value represented by color scale (kcal/mol)
    figsize=None,  # figure size in inches
    seqId1="Seq1",  # name of first RNA
    seqId2="Seq2",  # name of second RNA
    path=None,
    barrier=None,
):
    """Plot energy landscape from RRIkinDP states as heatmap."""

    # read states(zero based indices)
    df = pd.DataFrame(states, columns=["k", "l", "E"])

    # set color scale limits
    if e_min is None:
        e_min = df[energy].min()
    if e_max is None:
        e_max = df[energy].max()

    # states list to energy landscape
    energy_df = df.pivot(index="k", columns="l", values=energy)

    # only annotate energies if the interaction has fewer than 15 base pairs
    if annotate is None:
        if len(energy_df) > 15:
            annotate = False
        else:
            annotate = True

    # set up plot
    if figsize is None:
        x = len(energy_df) / 4.5 + 4
        y = len(energy_df) / 4.5 + 2
        figsize = (x, y)

    f, ax = plt.subplots(figsize=figsize)  # 9, 7)

    # generate heatmap
    ax = sns.heatmap(
        energy_df,
        vmin=e_min,
        vmax=e_max,
        center=0,
        cmap="RdBu_r",
        linewidths=0.2,
        linecolor="lightgrey",
        annot=annotate,
        fmt="2.1f",
        cbar_kws={
            "label": f"free energy term: {energy} in kcal/mol",
            "pad": 0.1,
            "shrink": 1.0,
        },
        square=True,
        ax=ax,
        # cbar=False,
    )

    if path is not None:
        # Start
        # ax.add_patch(Rectangle((path[0][1]-1, path[0][0]-1), 1, 1, fill=False, edgecolor='limegreen', lw=3))
        ax.add_patch(Rectangle((path[0][1]-1, path[0][0]-1), 1, 1, fill=False, edgecolor='gold', lw=2))
        y, x = zip(*path)
        ax.plot([t-0.5 for t in x], [t-0.5 for t in y], color='limegreen', ls='--', lw=2)
        # ax.plot([t-0.5 for t in x], [t-0.5 for t in y], color='gold', ls='--', lw=2)
        # End
        # ax.add_patch(Rectangle((path[-1][1]-1, path[-1][0]-1), 1, 1, fill=False, edgecolor='forestgreen', lw=3))
        ax.add_patch(Rectangle((path[-1][1]-1, path[-1][0]-1), 1, 1, fill=False, edgecolor='darkorange', lw=2))
        # barrier
        if barrier is not None:
            # ax.add_patch(Rectangle((barrier[1]-1, barrier[0]-1), 1, 1, fill=False, edgecolor='darkorange', lw=3, ls=':'))
            ax.add_patch(Rectangle((barrier[1]-1, barrier[0]-1), 1, 1, fill=False, edgecolor='forestgreen', lw=2, ls=':'))

    
    # set title
    plt.title(f"Energy Landscape: {seqId1} + {seqId2}")

    # set axis lables
    ax.set_ylabel("start base pair")
    ax.set_xlabel("end base pair")

    # draw axis ticks on all four sides
    plt.tick_params(labeltop=True, labelright=True, top=True, right=True)

    # include structure information in plot
    if structure is not None:
        # plot full interaction structure
        props = dict(facecolor="white", alpha=0.7)  # boxstyle='round')
        ax.text(
            0.05,
            0.15,
            structure,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=props,
            fontfamily="monospace",
        )

        # label axis with base pairs in addition to one based indices
        struct = structure.split("\n")
        labels = [
            [struct[0][e], struct[2][e]] for e, c in enumerate(struct[1]) if c == "|"
        ]
        xticklabels = [f"{i+1}\n" + "\nI\n".join(l) for i, l in enumerate(labels)]
        yticklabels = [f"{i+1} " + " ─ ".join(l) for i, l in enumerate(labels)]
        ax.set_xticks([t + 0.5 for t in range(0, len(energy_df))])
        ax.set_yticks([t + 0.5 for t in range(0, len(energy_df))])
        ax.set_xlim(0, len(energy_df))
        ax.set_ylim(len(energy_df), 0)
        ax.set_xticklabels(
            xticklabels,
            fontdict={
                "fontsize": 8,
                "horizontalalignment": "center",
            },
        )
        ax.set_yticklabels(
            yticklabels,
            rotation="horizontal",
            fontsize=8,
        )
    # if no structure information: change zero based to one based indices
    else:
        yticklabels = [
            int(item.get_text()) + 1
            for item in ax.get_yticklabels()
            if item.get_position()[0] == 0
        ]
        ax.set_yticklabels(yticklabels, rotation="horizontal")
        xticklabels = [
            int(item.get_text()) + 1
            for item in ax.get_xticklabels()
            if item.get_position()[1] == 0
        ]
        ax.set_xticklabels(xticklabels)

    # write plot to file
    if figure_path is not None:
        f.savefig(figure_path)
    else:
        id1 = seqId1.replace(" ", "_")
        id2 = seqId2.replace(" ", "_")
        figure_path = f"{id1}-{id2}-{energy}.pdf"
        f.savefig(figure_path)

    # close plot to avoid to many open plots if function is called multiple times
    # plt.close(f)


def intarna_to_bplist(bplist_string, zero_based=False):
    """Convert base pair list from intarna string format to python list.

    Args:
        bplist_string (str): string with base pairs as for example returned by
            IntaRNA. For example:
            '(134,56):(135,55):(136,54):(137,53):(138,52):(139,51)'
        zero_based (bool): if true 1-based intarna position indices get
            converted to 0-based indices

    Returns:
        List of base pairs. Each base pair is represented as a tuple of the
        pairing positions. For example:
        [(134, 56), (135, 55), (136, 54), (137, 53), (138, 52), (139, 51)]
        or if zero_based = True:
        [(133, 55), (134, 54), (135, 53), (136, 52), (137, 51), (138, 50)]
    """
    b = 0
    if zero_based:
        b = 1
    return [
        (
            int(item.split(",")[0].strip("(")) - b,
            int(item.split(",")[1].strip(")")) - b,
        )
        for item in bplist_string.split(":")
    ]


def get_string_representations(seq1, seq2, bp_list, id1="Seq1", id2="Seq2"):
    """Get interaction represented as a single string with three lines.

    Args:
        seq1 (str): full sequence of first RNA
        seq2 (str): full sequence of second RNA
        bp_list (list): list of interacting base pairs in tuple. Note that indices should be one based
        id1 (str): name of first RNA
        id2 (str): name of second RNA

    Returns:
        Within the three line string representation, the first and third line
        repesenting the seqeunce of the two pairing RNAs within the
        interaction site. The sequences contain gaps such that the paring
        positions are aligned. The sequence directions are annotated with 5'
        and 3'. The subsequence is annotated after the sequence id by the
        (one based) index of the first and last nucleotide within the
        interaction site. Base pairs are marked by pipes in the
        corresponding positions within the second line. Interior loops and
        buldges within the interaction site correspond to spaces within the
        second line.

        Examples (missing tailing spaces):

        5'-UACGGC-3' ArcZ[50:55]
           ||||||
        3'-AUGUCG-5' CyaR[34:29]

        5'-GAUUUCCUGGUGUAACGAAUUUUUUAAGUGC-3' DsrA[10:40]
           ||||||||  |||||||||||||  ||||||
        3'-CUAAAGGGGAACAUUGCUUAAAGU-UUUACG-5' rpoS[104:75]
    """

    # introduce gaps such that pairing sequence positions are aligned
    # and introduce pipes to mark pairing positions

    gapped_seq1 = ""  # firs line
    gapped_seq2 = ""  # third line
    bps_as_string = ""  # second line
    for i in range(len(bp_list) - 1):
        len_a_frag = -bp_list[i][0] + bp_list[i + 1][0]
        len_b_frag = bp_list[i][1] - bp_list[i + 1][1]
        fragment_length = max(len_a_frag, len_b_frag)
        gapped_seq1 += (
            seq1[bp_list[i][0] - 1 : bp_list[i + 1][0] - 1]
            + (fragment_length - len_a_frag) * "-"
        )
        gapped_seq2 += (
            seq2[bp_list[i][1] - 1 : bp_list[i + 1][1] - 1 : -1]
            + (fragment_length - len_b_frag) * "-"
        )
        bps_as_string += "|" + (fragment_length - 1) * " "
    gapped_seq1 += seq1[bp_list[-1][0] - 1]
    gapped_seq2 += seq2[bp_list[-1][1] - 1]
    bps_as_string += "|"

    # annotate sequences
    gapped_seq1 = f"5'-{gapped_seq1}-3' {id1}[{bp_list[0][0]},{bp_list[-1][0]}]"
    gapped_seq2 = f"3'-{gapped_seq2}-5' {id2}[{bp_list[0][1]},{bp_list[-1][1]}]"
    bps_as_string = f"   {bps_as_string}    "

    # unify length of lines
    length = max([len(gapped_seq1), len(gapped_seq2)])
    gapped_seq1 = gapped_seq1.ljust(length)
    gapped_seq2 = gapped_seq2.ljust(length)
    bps_as_string = bps_as_string.ljust(length)

    # lines = [gapped_seq1, gapped_seq2, bps_as_string]
    # length = max([len(l) in lines])
    # lines = [l.ljust(length) for l in lines]
    # gapped_seq1, gapped_seq2, bps_as_string = lines

    return "\n".join([gapped_seq1, bps_as_string, gapped_seq2])


def run_intarna(
    seq1,
    seq2,
    id1="target",
    id2="query",
    temperature=37.0,
    intarna_args=[],
    out_file=None,
    intarna_executable="IntaRNA",
    outMode="C",
    outCsvCols="id1,id2,start1,end1,start2,end2,seq1,seq2,"
    + "bpList,E,Etotal,ED1,ED2,Pu1,Pu2,E_init,E_loops,E_dangleL,"
    + "E_dangleR,E_endL,E_endR,E_hybrid,E_norm,E_add,P_E,hybridDPfull",
):
    """Run IntaRNA.
        Only tested with csv output format.
        Provide following intarna arguments through function arguments
        and not through intarna_args variable:
        - --out  as out_file
        - --outMode as outMode
        - -t as seq1
        - -q as seq2
        - --tId  as id1
        - --qId as id2

    Args:
        seq1: sequence 1
        seq2: sequence 2
        id1: sequence 1 identity
        id2: sequence 2 identity
        temperature: temperature
        intarna_args: additional interna arguments
        out_file: path to store output
        intarna_executable: callable IntaRNA tool
        outMode: IntaRNA output mode
        outCsVCols: output csv format
    """
    # set up intarna arguments
    intarna_args = [
        intarna_executable,
        "-t",
        seq1,
        "-q",
        seq2,
        "--tId",
        id1,
        "--qId",
        id2,
        "--temperature",
        str(temperature),
        "--outMode",
        outMode,
        "--outCsvCols=" + outCsvCols,
    ] + intarna_args
    if out_file is not None:
        intarna_args.append("--out")
        intarna_args.append(out_file)

    # call intarna
    cp = subprocess.run(
        intarna_args,
        universal_newlines=True,  # TODO: needed? pd needs stream anyway
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        capture_output=True,
    )
    # prepare return format
    if cp.returncode != 0:
        print("IntaRNA returncode is " + str(cp.returncode))
        print(cp)

    if outMode == "C":
        if out_file is None:
            df = pd.read_csv(StringIO(cp.stdout), sep=";", comment="#")
        else:
            df = pd.read_csv(out_file, sep=";", comment="#")
        return df
    else:
        if out_file is None:
            return cp.stdout
        else:
            with open(out_file, "r") as out_handle:
                intarna_output = out_handle.read()
            return intarna_output


def gen_intarna_args(max_bp_span=None):
    span = int(max_bp_span) if max_bp_span is not None else PARAMS["BP_SPAN"]
    Intarna_args = [
        "--noseed",  # turn off seed heuristic
        "--tAccW",
        "0",  # window size for accessibility prediction; 0 == full sequence
        "--tAccL",
        "0",  # max bp span for accessibility prediction; 0 == full sequence
        "--qAccW",
        "0",  # window size for accessibility prediction; 0 == full sequence
        "--qAccL",
        f"{span}",  # max bp span for accessibility prediction; 0 == full sequence
        "--tIntLenMax",
        "50",  # maximum interaction length on target
        "--qIntLenMax",
        "50",  # maximum interaction length on query
        "--mode",
        "H",  # M ... exact mode; H ... heuristic, default
        "--model",
        "S",  # S ... single side, mfe; X ... single side, mfe, seed extension
        "--outNumber",
        "1",  # set to 1 of you only want the mfe interaction
        "--outMaxE",
        "5",  # maximum interaction energy to be returned in kcal/mol
    ]
    return Intarna_args


def RNAup(seq1, seq2, structure=False):
    """Predict binding region using RNAup for given sequences
    RNAup reorders sequences by length in descending order

    Args:
        seq1: sequence 1
        seq2: sequence 2
        structure: add predicted structure into output if enable

    Returns:
        Tuple of (i, j, k, l, eb), where (i, j) is binding region of longer sequence (mRNA),
        (k, l) is the one for small RNA and eb is binding energy
    """
    input = seq1 + "\n" + seq2 + "\n"
    rnaup = subprocess.run(
        ["RNAup", "-b", "-w 40"], input=input.encode(), capture_output=True
    )

    if structure:
        match = re.search(
            r"([(.)]+)&([(.)]+) +(\d+),(\d+) +: +(\d+),(\d+).*\((-?\d+\.\d+) =",
            rnaup.stdout.decode(),
        )
        groups = match.groups()
        return [eval(groups[i]) for i in range(2, len(groups))] + [groups[0], groups[1]]
    match = re.search(
        r" (\d+),(\d+) +: +(\d+),(\d+).*\((-?\d+\.\d+) =", rnaup.stdout.decode()
    )
    # careful RNAup reorders sequences, such that the long sequence comes first (e.g. mRNA then sRNA)!
    return [eval(i) for i in match.groups()]


def bp_list_from_rnaup(seq1, seq2, base_one=True):
    """Return interaction basepair (i, j) list predicted by RNAup for two given sequences seq1, seq2
    i is position in mRNA (longer one) and j is the position in shorter one

    Args:
        seq1: sequence 1
        seq2: sequence 2
        base_one: use 1-index
    """
    i, j, k, l, _, str1, str2 = RNAup(seq1, seq2, structure=True)
    str1_paired = [i for i, x in enumerate(str) if x == "("]
    str2_paired = [i for i, x in enumerate(str) if x == ")"]
    assert len(str1_paired) == len(str2_paired)
    return [
        (i + x - int(base_one), k + y - int(base_one))
        for x, y in zip(str1_paired, str2_paired)
    ]


def prediction_unbound(target, query, max_bp_span=None):
    """Return unbound MFE structure for target and query sequences

    Args:
        target: target mRNA sequence to bind
        query: small RNA binding on mRNA
        max_bp_span: maximum basepair span value of MFE prediction for target sequence
    """
    span = int(max_bp_span) if max_bp_span is not None else PARAMS["BP_SPAN"]
    if span == 0:
        md = RNA.md()
    else:
        md = RNA.md(max_bp_span=span)
    fc = RNA.fold_compound(target, md)
    return fc.mfe()[0], RNA.fold(query)[0]


def prediction_bound(target, query, *, mode="RNAup", max_bp_span=None):
    """Return bound MFE structure for target and query sequences

    Args:
        target: target mRNA sequence to bind
        query: small RNA binding on mRNA
        mode: interection prediction tool, RNAup for IntaRNA
        max_bp_span: maximum basepair span value of MFE prediction for target sequence.
    """
    mode = mode.lower()
    if mode == "rnaup":
        end_pos, target_str, query_str = _binding_from_rnaup(target, query)
    elif mode == "intarna":
        end_pos, target_str, query_str = _binding_from_intarna(target, query)
    else:
        raise ValueError("Value of mode should be either RNAup or IntaRNA")

    # Compute MFE for free area in target
    constraint = "x" * end_pos + "." * (len(target) - end_pos)
    span = int(max_bp_span) if max_bp_span is not None else PARAMS["BP_SPAN"]
    if span == 0:
        md = RNA.md()
    else:
        md = RNA.md(max_bp_span=span)
    fc = RNA.fold_compound(target, md)
    fc.hc_add_from_db(constraint)
    mfe, _ = fc.mfe()
    return target_str[:end_pos] + mfe[end_pos:], query_str


def _binding_from_rnaup(target, query):
    """Return prediction at binding region using RNAup

    Args:
        target: target mRNA sequence to bind
        query: small RNA binding on mRNA

    Return:
        Tuple: (end position of target binding region, target binding structure, query binding structure)
    """
    i, j, k, l, _, target_binding, query_binding = RNAup(target, query, structure=True)
    return (
        j,
        "." * (i - 1) + target_binding.replace("(", ")") + "." * (len(target) - j),
        "." * (k - 1) + query_binding.replace(")", "(") + "." * (len(query) - l),
    )


def _binding_from_intarna(target, query):
    """Return prediction at binding region using intarna

    Args:
        target: target mRNA sequence to bind
        query: small RNA binding on mRNA

    Return:
        Tuple: (end position of target binding region, target binding structure, query binding structure)
    """
    df = run_intarna(target, query, intarna_args=gen_intarna_args(), out_file=None)
    # list of binding positions (target, query)
    bp_list = intarna_to_bplist(df.to_dict(orient="records")[0]["bpList"])
    target_str = ["."] * len(target)
    query_str = ["."] * len(query)
    max_i = 0
    # Structure for binding area
    for i, j in bp_list:
        query_str[j] = "("
        target_str[i] = ")"
        max_i = max(i, max_i)

    return max_i, "".join(target_str), "".join(query_str)


def draw_RNAplot(
    sRNA_seq,
    mRNA_seq,
    sRNA_str,
    mRNA_str,
    output=None,
    SD_pos=None,
    SD_len=None,
    AUG_pos=None,
    show=True,
):
    """Draw small RNA and mRNA
    The function concatenate two RNAs with "  &  "

    Args:
        sRNA_seq: sequence of small RNA
        mRNA_seq: sequence of mRNA
        sRNA_str: structure of small RNA in dbn
        mRNA_str: structure of mRNA in dbn
        output: path to save the plot. Display the draw if not given
        SD_pos: Shine-Dalgarno position
        SD_len: Shine-Dalgarno motif length
        AUG_pos: AUG position
        show: display plot if True (for Jupyter)
    """
    SD_pos = SD_pos if SD_pos is not None else PARAMS["SD_pos"]
    SD_len = SD_len if SD_len is not None else len(PARAMS["SD"])
    AUG_pos = AUG_pos if AUG_pos is not None else PARAMS["AUG_pos"]

    seq = sRNA_seq + "&" + mRNA_seq
    ss = sRNA_str + "&" + mRNA_str

    input = seq + "\n" + ss + "\n"

    sRNA_len = len(sRNA_seq)
    mRNA_len = len(mRNA_seq)

    annot = "{} {} 9 0.8 0.8 1 omark ".format(
        sRNA_len + SD_pos + 1, sRNA_len + SD_pos + 29 + 1
    )
    annot = annot + "{} {} 10 BLUE omark ".format(
        sRNA_len + SD_pos + 1, sRNA_len + SD_pos + 6
    )
    annot = annot + "{} {} 10 GREEN omark ".format(
        sRNA_len + AUG_pos + 1, sRNA_len + AUG_pos + 3
    )

    plot = subprocess.run(
        ["RNAplot", "-t 4", "--pre", annot], input=input.encode(), capture_output=True
    )

    if output is not None:
        shutil.copy("rna.ps", output)

    if show:
        img = Image.open("rna.ps")
        display(img)


def draw_RNAplot_mRNA(
    mRNA_seq, mRNA_str, output=None, SD_pos=None, SD_len=None, AUG_pos=None, show=True
):
    """Similar to draw_RNAplot but for mRNA only

    Args:
        mRNA_seq: sequence of mRNA
        mRNA_str: structure of mRNA in dbn
        output: path to save the plot. Display the draw if not given
        SD_pos: Shine-Dalgarno position
        SD_len: Shine-Dalgarno motif length
        AUG_pos: AUG position
        show: display plot if True (for Jupyter)
    """
    SD_pos = SD_pos if SD_pos is not None else PARAMS["SD_pos"]
    SD_len = SD_len if SD_len is not None else len(PARAMS["SD"])
    AUG_pos = AUG_pos if AUG_pos is not None else PARAMS["AUG_pos"]

    seq = mRNA_seq
    ss = mRNA_str

    input = seq + "\n" + ss + "\n"

    mRNA_len = len(mRNA_seq)

    annot = "{} {} 9 0.8 0.8 1 omark ".format(SD_pos + 1, SD_pos + 29 + 1)
    annot = annot + "{} {} 10 BLUE omark ".format(SD_pos + 1, SD_pos + 6)
    annot = annot + "{} {} 10 GREEN omark ".format(AUG_pos + 1, AUG_pos + 3)

    plot = subprocess.run(
        ["RNAplot", "-t 4", "--pre", annot], input=input.encode(), capture_output=True
    )

    if output is not None:
        shutil.copy("rna.ps", output)

    if show:
        img = Image.open("rna.ps")
        display(img)


def plot_energylandscape(
    sRNA,
    mRNA,
    id1="sRNA",
    id2="mRNA",
    figure_path=None,
    print_info=False,
    temperature=37.0,
    figsize=None,
    verbose = False,
    path=None,
    barrier=None
):
    df = run_intarna(
        sRNA,
        mRNA,
        id1="sRNA",
        id2="mRNA",
        temperature=temperature,
        intarna_args=gen_intarna_args(),
    )
    mfe_interaction = df.to_dict(orient="records")[0]
    if print_info:
        print(mfe_interaction)
    # compute kinetic features of best interaction
    bp_list = intarna_to_bplist(mfe_interaction["bpList"], zero_based=True)
    seed_length = 5
    # TODO: catch if no interaction predicted
    el = EnergyLandscape(sRNA, mRNA, bp_list, temperature=temperature)
    if verbose:
        min_barrier_energies = el.get_min_barrier_Es(seed_length)
        seed_stabilities = el.get_seed_Es(seed_length)
        min_barrier = min(min_barrier_energies)
        seed_stability_min_barrier = seed_stabilities[
            min_barrier_energies.index(min_barrier)
        ]
        rrikindp_full_E = el.get_full_E()

        print(
            f"- free energy of the full interaction with RRIkinDP: {rrikindp_full_E}kcal/mol"
        )
        print(
            f"- free energy of the barrier state: {min_barrier}kcal/mol"
        )
        print(
            f"- free energy of {seed_length}bps seed from which the minimum barrier folding paths starts: {seed_stability_min_barrier}kcal/mol"
        )
        print(
            f"- seed resulting in the minimum barrier state folding path starts at base pair {min_barrier_energies.index(min_barrier)+1}"
        )
        print(
            f"The interaction range on the sRNA is {mfe_interaction['start1']}-{mfe_interaction['end1']}"
        )
        print(
            f"The interaction range on the mRNA is {mfe_interaction['start2']}-{mfe_interaction['end2']}"
        )
        print(f"Base pair list: {mfe_interaction['bpList']}")
        print(f'interaction in db notation: {mfe_interaction["hybridDPfull"]}')
        print(
            f'E: {mfe_interaction["E"]}, ED1: {mfe_interaction["ED1"]}, ED2: {mfe_interaction["ED2"]}, E_loops: {mfe_interaction["E_loops"]}'
        )
    
    states = el.get_states()
    structures = get_string_representations(
        sRNA, mRNA, intarna_to_bplist(mfe_interaction["bpList"]), id1=id1, id2=id2
    )
    plot_landscape(
        states,
        structure=structures,
        figure_path=figure_path,
        seqId1=id1,
        seqId2=id2,
        figsize=figsize,
        path=path,
        barrier=barrier,
    )


# translation table AA to iupac codon
tt = [
    ["Ala", "A ", ["GCN"]],
    ["Ile", "I", ["AUH"]],
    ["Arg", "R", ["CGN", "AGR"]],
    ["Leu", "L", ["CUN", "UUR"]],
    ["Asn", "N", ["AAY"]],
    ["Lys", "K", ["AAR"]],
    ["Asp", "D", ["GAY"]],
    ["Met", "M", ["AUG"]],
    ["Asp", "B", ["RAY"]],
    ["Phe", "F", ["UUY"]],
    ["Cys", "C", ["UGY"]],
    ["Pro", "P", ["CCN"]],
    ["Gln", "Q", ["CAR"]],
    ["Ser", "S", ["UCN", "AGY"]],
    ["Glu", "E", ["GAR"]],
    ["Thr", "T", ["ACN"]],
    ["Gln", "Z", ["SAR"]],
    ["Trp", "W", ["UGG"]],
    ["Gly", "G", ["GGN"]],
    ["Tyr", "Y", ["UAY"]],
    ["His", "H", ["CAY"]],
    ["Val", "V", ["GUN"]],
    ["START", "START", ["HUG"]],
    ["STOP", "STOP", ["URA", "UAG"]],
]

translation_dict = {}
for aa, aa_short, iupac_codons in tt:
    translation_dict[aa_short] = iupac_codons


def protein2iupacRNA(protein_seq, ACGU_only=False):
    """Convert protein sequence to RNA sequence
    Note that a random three letters is chosen if AA can be translated to more than one.

    Args:
        protein_seq: protein sequence
        ACGU_only: flag to use ACGU only, randomly pick standard compatible letter for each IUPAC letter
    """
    cds = [random.choice(translation_dict[aa]) for aa in protein_seq]
    cds = "".join(cds)
    if ACGU_only:
        return "".join(random.choice(infrared.rna._iupac_nucleotides[x]) for x in cds)
    return cds
