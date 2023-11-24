""" Core definition of document describing molecular chemical reactions """
from typing import List, Optional, Tuple, Union

from pydantic import Field

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

from emmet.core.mpid import MPID, MPculeID  # TODO: consider switching to MPculeID-based IDs
from emmet.core.structure import MoleculeMetadata

from rapter_eggs.schema.calc_types import LevelOfTheory, TaskType
from rapter_eggs.schema.task import filter_task_type
from rapter_eggs.schema.pes import (
    evaluate_lot,
    PESPointDoc,
)


__author__ = "Evan Spotte-Smith <ewcspottesmith@lbl.gov>"


METALS = ["Li", "Na", "Mg", "Ca", "Zn", "Al"]


def find_common_reaction_lot_opt(
    endpoint1: PESPointDoc,
    endpoint2: PESPointDoc,
    transition_state: PESPointDoc,
) -> Optional[str]:
    """
    Identify the highest level of theory (LOT) used in a reaction.

    :param endpoint1: PESPointDoc for the first endpoint
    :param endpoint2: PESPointDoc for the second endpoint
    :param transition_state: PESPointDoc for the transition-state of this
        reaction
    :return: String representation of the best common level of theory.
    """
    lots_end1 = sorted(endpoint1.best_entries.keys(), key=lambda x: evaluate_lot(x))
    lots_end2 = sorted(endpoint2.best_entries.keys(), key=lambda x: evaluate_lot(x))
    lots_ts = sorted(
        transition_state.best_entries.keys(), key=lambda x: evaluate_lot(x)
    )

    for lot in lots_ts:
        if lot in lots_end1 and lot in lots_end2:
            return lot

    return None


def find_common_reaction_lot_sp(
    endpoint1: PESPointDoc,
    endpoint2: PESPointDoc,
    transition_state: PESPointDoc,
) -> Optional[str]:
    """
    Identify the highest level of theory (LOT) used for single-point energy correction
    in a reaction.

    :param endpoint1: PESPointDoc for the first endpoint
    :param endpoint2: PESPointDoc for the second endpoint
    :param transition_state: PESPointDoc for the transition-state of this
        reaction
    :return: String representation of the best common level of theory.
    """

    sp_end1 = filter_task_type(endpoint1.entries, TaskType.Single_Point)
    sp_end2 = filter_task_type(endpoint2.entries, TaskType.Single_Point)
    sp_ts = filter_task_type(transition_state.entries, TaskType.Single_Point)

    lots_end1 = sorted(
        list({e["level_of_theory"] for e in sp_end1}), key=lambda x: evaluate_lot(x)
    )
    lots_end2 = sorted(
        list({e["level_of_theory"] for e in sp_end2}), key=lambda x: evaluate_lot(x)
    )
    lots_ts = sorted(
        list({e["level_of_theory"] for e in sp_ts}), key=lambda x: evaluate_lot(x)
    )
    for lot in lots_ts:
        if lot in lots_end1 and lot in lots_end2:
            return lot

    return None


def find_freq_entry(doc: PESPointDoc, lot: Union[LevelOfTheory, str]):
    """
    Find an entry in a PESPointDoc (minimum or transition-state) with frequency
    information.

    :param doc: PESPointDoc to be searched through
    :param lot: level of theory. Only entries in doc with this LOT will be
        accepted
    :return: entry dict, or None
    """

    if isinstance(lot, LevelOfTheory):
        lot_value = lot.value
    else:
        lot_value = lot

    possible_entries = list()
    for e in doc.entries:
        if e["level_of_theory"] != lot_value:
            continue
        elif e["input"]["gen_variables"].get("ifreq", 0) != 1:
            continue
        elif e["output"].get("zero_point_energy") is None:
            continue

        possible_entries.append(e)

    if len(possible_entries) == 0:
        return None

    return sorted(possible_entries, key=lambda x: x["energy"])[0]


def bonds_metal_nometal(mg: MoleculeGraph):
    """
    Extract the bonds (with and without counting metal coordination) from a
    MoleculeGraph.

    TODO: Should this functionality just be in pymatgen?

    :param mg: MoleculeGraph
    :return:
        - bonds: List of tuples (a, b) representing bonds, where a and b are
            the 0-based atom indices
        - bonds_nometal: List of tuples (a, b) representing non-coordinate
            bonds, where a and b are the 0-based atom indices
    """

    bonds = list()
    for bond in mg.graph.edges():
        bonds.append(tuple(sorted([bond[0], bond[1]])))

    m_inds = [
        e for e in range(len(mg.molecule)) if str(mg.molecule.species[e]) in METALS
    ]

    bonds_nometal = list()
    for bond in bonds:
        if not any([m in bond for m in m_inds]):
            bonds_nometal.append(bond)

    return (bonds, bonds_nometal)


def bond_species(mol: Molecule, bond: Tuple[int, int]):
    """
    Get the elements involved in a bond
    :param mol: Molecule
    :param bond: Tuple (a, b), where a and b are 0-based atom indices representing
        the atoms in the given bond
    :return:
    """

    return "-".join(sorted([str(mol.species[bond[0]]), str(mol.species[bond[1]])]))


class ReactionDoc(MoleculeMetadata):

    reaction_id: MPID = Field(..., description="Unique identifier for this reaction.")

    reactant_id: MPID = Field(
        ..., description="Unique ID for the reactants for this reaction."
    )
    product_id: MPID = Field(
        ..., description="Unique ID for the products for this reaction."
    )
    transition_state_id: MPID = Field(
        ..., description="Unique ID of the transition-state for this reaction."
    )

    deprecated: bool = Field(False, description="Is this reaction deprecated?")

    reactant_structure: Molecule = Field(
        None, description="Molecule object describing the reactants of this reaction."
    )
    reactant_molecule_graph: MoleculeGraph = Field(
        None,
        description="Structural and bonding information for the reactants of this reaction.",
    )
    reactant_molecule_graph_nometal: MoleculeGraph = Field(
        None,
        description="Structural and bonding information for the reactants of this reaction, "
        "removing all metals.",
    )
    reactant_bonds: List[Tuple[int, int]] = Field(
        [],
        description="List of bonds in the reactants in the form (a, b), where a and b are 0-indexed "
        "atom indices",
    )
    reactant_bonds_nometal: List[Tuple[int, int]] = Field(
        [],
        description="List of bonds in the reactants in the form (a, b), where a and b are 0-indexed "
        "atom indices, with all metal ions removed",
    )
    reactant_coord_hash: str = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash of the reactant using the atom coordinates as the graph "
        "node attribute.",
    )
    reactant_species_hash: str = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash of the reactant using the atom species as the "
                    "graph node attribute."
    )
    reactant_species_hash_nometal: str = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash of the reactant using the atom species as the "
                    "graph node attribute, where metal bonds are excluded."
    )
    reactant_energy: float = Field(
        None,
        description="Electronic energy of the reactants of this reaction (units: eV).",
    )
    reactant_zpe: float = Field(
        None,
        description="Vibrational zero-point energy of the reactants of this reaction (units: eV).",
    )
    reactant_enthalpy: float = Field(
        None, description="Enthalpy of the reactants of this reaction (units: eV)."
    )
    reactant_entropy: float = Field(
        None, description="Entropy of the reactants of this reaction (units: eV/K)."
    )
    reactant_free_energy: float = Field(
        None,
        description="Gibbs free energy of the reactants of this reaction at 298.15K (units: eV).",
    )
    reactant_charges: List[Optional[float]] = Field(
        None,
        description="Atomic partial charges of the reactants of this reaction based on the electrostatic potential "
                    "method."
    )

    # Product properties
    product_structure: Molecule = Field(
        None, description="Molecule object describing the products of this reaction."
    )
    product_molecule_graph: MoleculeGraph = Field(
        None,
        description="Structural and bonding information for the products of this reaction.",
    )
    product_molecule_graph_nometal: MoleculeGraph = Field(
        None,
        description="Structural and bonding information for the products of this reaction, "
        "removing all metals.",
    )
    product_bonds: List[Tuple[int, int]] = Field(
        [],
        description="List of bonds in the products in the form (a, b), where a and b are 0-indexed "
        "atom indices",
    )
    product_bonds_nometal: List[Tuple[int, int]] = Field(
        [],
        description="List of bonds in the products in the form (a, b), where a and b are 0-indexed "
        "atom indices, with all metal ions removed",
    )
    product_coord_hash: str = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash of the product using the atom coordinates as the graph "
        "node attribute.",
    )
    product_species_hash: str = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash of the product using the atom species as the "
                    "graph node attribute."
    )
    product_species_hash_nometal: str = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash of the product using the atom species as the "
                    "graph node attribute, where metal bonds are excluded."
    )
    product_energy: float = Field(
        None,
        description="Electronic energy of the products of this reaction (units: eV).",
    )
    product_zpe: float = Field(
        None,
        description="Vibrational zero-point energy of the products of this reaction (units: eV).",
    )
    product_enthalpy: float = Field(
        None, description="Enthalpy of the products of this reaction (units: eV)."
    )
    product_entropy: float = Field(
        None, description="Entropy of the products of this reaction (units: eV/K)."
    )
    product_free_energy: float = Field(
        None,
        description="Gibbs free energy of the products of this reaction at 298.15K (units: eV).",
    )
    product_charges: List[Optional[float]] = Field(
        None,
        description="Atomic partial charges of the products of this reaction based on the electrostatic potential "
                    "method."
    )

    # TS properties
    transition_state_structure: Molecule = Field(
        None,
        description="Molecule object describing the transition-state of this reaction.",
    )
    transition_state_energy: float = Field(
        None,
        description="Electronic energy of the transition-state of this reaction (units: eV).",
    )
    transition_state_zpe: float = Field(
        None,
        description="Vibrational zero-point energy of the transition-state of this reaction (units: eV).",
    )
    transition_state_enthalpy: float = Field(
        None,
        description="Enthalpy of the transition-state of this reaction (units: eV).",
    )
    transition_state_entropy: float = Field(
        None,
        description="Entropy of the transition-state of this reaction (units: eV/K).",
    )
    transition_state_free_energy: float = Field(
        None,
        description="Gibbs free energy of the transition-state of this reaction at 298.15K (units: eV).",
    )
    transition_state_charges: List[Optional[float]] = Field(
        None,
        description="Atomic partial charges of the transition-state of this reaction using the electrostatic potential "
                    "method."
    )

    # Reaction thermodynamics
    dE: float = Field(
        None, description="Electronic energy change of this reaction (units: eV)."
    )
    dH: float = Field(None, description="Enthalpy change of this reaction (units: eV).")
    dS: float = Field(
        None, description="Entropy change of this reaction (units: eV/K)."
    )
    dG: float = Field(None, description="Gibbs free energy (units: eV).")

    # Reaction barrier
    dE_barrier: float = Field(
        None,
        description="Electronic energy barrier (TS - reactant) of this reaction (units: eV).",
    )
    dH_barrier: float = Field(
        None,
        description="Enthalpy barrier (TS - reactant) of this reaction " "(units: eV).",
    )
    dS_barrier: float = Field(
        None,
        description="Entropy barrier (TS - reactant) of this reaction (units: eV/K).",
    )
    dG_barrier: float = Field(
        None, description="Gibbs free energy barrier (TS - reactant) " "(units: eV)."
    )

    # Bonding changes
    bonds_broken: List[Tuple[int, int]] = Field(
        [],
        description="List of bonds broken during the reaction in the form (a, b), where a and b are"
        "0-indexed atom indices.",
    )
    bond_types_broken: List[str] = Field(
        [],
        description="List of types of bonds being broken during the reaction, e.g. C-O for a "
        "carbon-oxygen bond.",
    )
    bonds_broken_nometal: List[Tuple[int, int]] = Field(
        [],
        description="List of bonds broken during the reaction in the form (a, b), where a and b are"
        "0-indexed atom indices, with all metal ions removed.",
    )
    bond_types_broken_nometal: List[str] = Field(
        [],
        description="List of types of bonds being broken during the reaction, e.g. C-O for a "
        "carbon-oxygen bond. This excludes bonds involving metal ions.",
    )
    bonds_formed: List[Tuple[int, int]] = Field(
        [],
        description="List of bonds formed during the reaction in the form (a, b), where a and b are"
        "0-indexed atom indices",
    )
    bond_types_formed: List[str] = Field(
        [],
        description="List of types of bonds being formed during the reaction, e.g. C-O for a "
        "carbon-oxygen bond.",
    )
    bonds_formed_nometal: List[Tuple[int, int]] = Field(
        [],
        description="List of bonds formed during the reaction in the form (a, b), where a and b are"
        "0-indexed atom indices, with all metal ions removed",
    )
    bond_types_formed_nometal: List[str] = Field(
        [],
        description="List of types of bonds being formed during the reaction, e.g. C-O for a "
        "carbon-oxygen bond. This excludes bonds involving metal ions.",
    )

    similar_reactions: List[MPID] = Field(
        None,
        description="Reactions that are similar to this one (for instance, because the same types "
        "of bonds are broken or formed)",
    )

    @classmethod
    def from_docs(
        cls,
        endpoint1: PESPointDoc,
        endpoint2: PESPointDoc,
        transition_state: PESPointDoc,
        deprecated: bool = False,
        **kwargs
    ):  # type: ignore[override]
        """
        Define a reaction based on reactant & product complexes and a
        transition-state

        :param endpoint1: PESPointDoc describing one endpoint of this reaction
        :param products: PESPointDoc describing the other endpoint of this
            reaction
        :param transition_state: PESPointDoc describing the TS of this
            reaction
        :param deprecated: Bool. Is this reaction deprecated?
        :param kwargs:
        :return: ReactionDoc
        """

        # Find best common LevelOfTheory
        # Use that LOT to calculate thermodynamic properties
        # Decide which endpoint is reactant/product based on ∆G
        # Take deltas of everything
        # Extract basic information (IDs, structures)
        # Make MoleculeGraphs

        # Find best level of theory - optimization
        chosen_lot_opt = find_common_reaction_lot_opt(
            endpoint1, endpoint2, transition_state
        )

        if chosen_lot_opt is None:
            raise ValueError(
                "Endpoints and Transition-State have no LevelOfTheory in common! Cannot compare."
            )

        end1_best = endpoint1.best_entries[chosen_lot_opt]
        end2_best = endpoint2.best_entries[chosen_lot_opt]
        ts_best = transition_state.best_entries[chosen_lot_opt]

        # If possible, find an entry with frequency information
        # This includes ZPE, H, S
        # This can be the same as the x_best entries above!
        end1_freq = find_freq_entry(endpoint1, chosen_lot_opt)
        end2_freq = find_freq_entry(endpoint2, chosen_lot_opt)
        ts_freq = find_freq_entry(transition_state, chosen_lot_opt)

        add_freq = True
        if any([x is None for x in [end1_freq, end2_freq, ts_freq]]):
            add_freq = False
        else:
            for freq in [end1_freq, end2_freq, ts_freq]:
                temps = [t["temperature"] for t in freq["output"]["thermo"]]
                if 298.15 not in temps:
                    add_freq = False
                    break

        # Find best level of theory - single-point
        chosen_lot_sp = find_common_reaction_lot_sp(
            endpoint1, endpoint2, transition_state
        )

        # If there are high-quality single-points, use them for energy
        if (
            chosen_lot_sp is not None
            and evaluate_lot(chosen_lot_sp) < evaluate_lot(chosen_lot_opt)
        ):
            end1_sp = filter_task_type(
                endpoint1.entries,
                TaskType.Single_Point,
                sort_by=lambda x: (x["level_of_theory"] != chosen_lot_sp, x["energy"]),
            )[0]
            end2_sp = filter_task_type(
                endpoint2.entries,
                TaskType.Single_Point,
                sort_by=lambda x: (x["level_of_theory"] != chosen_lot_sp, x["energy"]),
            )[0]
            ts_sp = filter_task_type(
                transition_state.entries,
                TaskType.Single_Point,
                sort_by=lambda x: (x["level_of_theory"] != chosen_lot_sp, x["energy"]),
            )[0]

            # Convert to eV
            # TODO: should we have a constant conversion factor?
            end1_e = end1_sp["energy"] * 27.2114
            end2_e = end2_sp["energy"] * 27.2114
            ts_e = ts_sp["energy"] * 27.2114

            end1_charges = [ap["esp_charge"] for ap in end1_sp["output"].get("atom_properties", list())]
            end2_charges = [ap["esp_charge"] for ap in end2_sp["output"].get("atom_properties", list())]
            ts_charges = [ap["esp_charge"] for ap in ts_sp["output"].get("atom_properties", list())]
        else:
            end1_e = end1_best["energy"] * 27.2114
            end2_e = end2_best["energy"] * 27.2114
            ts_e = ts_best["energy"] * 27.2114

            end1_charges = [ap["esp_charge"] for ap in end1_best["output"].get("atom_properties", list())]
            end2_charges = [ap["esp_charge"] for ap in end2_best["output"].get("atom_properties", list())]
            ts_charges = [ap["esp_charge"] for ap in ts_best["output"].get("atom_properties", list())]

        # TS thermo and structural information
        ts_id = transition_state.molecule_id
        ts_structure = transition_state.molecule

        # Endpoint thermo and structural information
        # endpoint_1 is the reactant
        if end1_e > end2_e:
            rct_id = endpoint1.molecule_id
            rct_structure = endpoint1.molecule
            rct_e = end1_e
            rct_freq = end1_freq
            rct_coord_hash = endpoint1.coord_hash
            rct_species_hash = endpoint1.species_hash
            rct_species_hash_nometal = endpoint1.species_hash_nometal
            rct_charges = end1_charges

            pro_id = endpoint2.molecule_id
            pro_structure = endpoint2.molecule
            pro_e = end2_e
            pro_freq = end2_freq
            pro_coord_hash = endpoint2.coord_hash
            pro_species_hash = endpoint2.species_hash
            pro_species_hash_nometal = endpoint2.species_hash_nometal
            pro_charges = end2_charges
        # endpoint_2 is the reactant
        else:
            rct_id = endpoint2.molecule_id
            rct_structure = endpoint2.molecule
            rct_e = end2_e
            rct_freq = end2_freq
            rct_coord_hash = endpoint2.coord_hash
            rct_species_hash = endpoint2.species_hash
            rct_species_hash_nometal = endpoint2.species_hash_nometal
            rct_charges = end2_charges

            pro_id = endpoint1.molecule_id
            pro_structure = endpoint1.molecule
            pro_e = end1_e
            pro_freq = end1_freq
            pro_coord_hash = endpoint1.coord_hash
            pro_species_hash = endpoint1.species_hash
            pro_species_hash_nometal = endpoint1.species_hash_nometal
            pro_charges = end1_charges

        dE = pro_e - rct_e
        dE_barrier = ts_e - rct_e

        ts_zpe = None
        ts_h = None
        ts_s = None
        ts_g = None
        rct_zpe = None
        rct_h = None
        rct_s = None
        rct_g = None
        pro_zpe = None
        pro_h = None
        pro_s = None
        pro_g = None

        # Get thermo (at 298.15K, where relevant)
        if add_freq:
            ts_zpe = ts_freq["output"]["zero_point_energy"] * 0.043363
            for thermo in ts_freq["output"]["thermo"]:
                if thermo["temperature"] == 298.15:
                    ts_h = thermo["enthalpy"]["total_enthalpy"] * 0.043363
                    ts_s = thermo["entropy"]["total_entropy"] * 0.000043363
                    ts_g = ts_e + ts_zpe + ts_h - 298.15 * ts_s
                    break

            rct_zpe = rct_freq["output"]["zero_point_energy"] * 0.043363
            for thermo in rct_freq["output"]["thermo"]:
                if thermo["temperature"] == 298.15:
                    rct_h = thermo["enthalpy"]["total_enthalpy"] * 0.043363
                    rct_s = thermo["entropy"]["total_entropy"] * 0.000043363
                    rct_g = rct_e + rct_zpe + rct_h - 298.15 * rct_s
                    break

            pro_zpe = pro_freq["output"]["zero_point_energy"] * 0.043363
            for thermo in pro_freq["output"]["thermo"]:
                if thermo["temperature"] == 298.15:
                    pro_h = thermo["enthalpy"]["total_enthalpy"] * 0.043363
                    pro_s = thermo["entropy"]["total_entropy"] * 0.000043363
                    pro_g = pro_e + pro_zpe + pro_h - 298.15 * pro_s
                    break

            rct_H = rct_e + rct_zpe + rct_h  # type: ignore
            ts_H = ts_e + ts_zpe + ts_h  # type: ignore
            pro_H = pro_e + pro_zpe + pro_h  # type: ignore

            dH = pro_H - rct_H  # type: ignore
            dH_barrier = ts_H - rct_H  # type: ignore

            dS = pro_s - rct_s  # type: ignore
            dS_barrier = ts_s - rct_s  # type: ignore

            dG = pro_g - rct_g  # type: ignore
            dG_barrier = ts_g - rct_g  # type: ignore

        else:
            dH = None
            dH_barrier = None
            dS = None
            dS_barrier = None
            dG = None
            dG_barrier = None

        # Bonding information
        rct_mg = metal_edge_extender(
            MoleculeGraph.with_local_env_strategy(rct_structure, OpenBabelNN())
        )
        rct_bonds, rct_bonds_nometal = bonds_metal_nometal(rct_mg)

        pro_mg = metal_edge_extender(
            MoleculeGraph.with_local_env_strategy(pro_structure, OpenBabelNN())
        )
        pro_bonds, pro_bonds_nometal = bonds_metal_nometal(pro_mg)

        #  Use set differences to identify which bonds are not present in rct/pro
        rct_bond_set = set(rct_bonds)
        rct_bond_nometal_set = set(rct_bonds_nometal)
        pro_bond_set = set(pro_bonds)
        pro_bond_nometal_set = set(pro_bonds_nometal)

        bonds_formed = sorted(pro_bond_set - rct_bond_set)
        bonds_broken = sorted(rct_bond_set - pro_bond_set)
        bonds_formed_nometal = sorted(pro_bond_nometal_set - rct_bond_nometal_set)
        bonds_broken_nometal = sorted(rct_bond_nometal_set - pro_bond_nometal_set)

        # Get bond types, as in C-O or Li-F
        bond_types_formed = sorted(
            set(map(lambda x: bond_species(ts_structure, x), bonds_formed))
        )
        bond_types_broken = sorted(
            set(map(lambda x: bond_species(ts_structure, x), bonds_broken))
        )
        bond_types_formed_nometal = sorted(
            set(map(lambda x: bond_species(ts_structure, x), bonds_formed_nometal))
        )
        bond_types_broken_nometal = sorted(
            set(map(lambda x: bond_species(ts_structure, x), bonds_broken_nometal))
        )

        rct_mg_nometal = MoleculeGraph.with_edges(
            rct_structure, {e: dict() for e in rct_bonds_nometal}
        )
        pro_mg_nometal = MoleculeGraph.with_edges(
            pro_structure, {e: dict() for e in pro_bonds_nometal}
        )

        reaction_id = "-".join([str(rct_id), str(ts_id), str(pro_id)])

        return cls.from_molecule(
            meta_molecule=ts_structure,
            deprecated=deprecated,
            reaction_id=reaction_id,
            reactant_id=rct_id,
            product_id=pro_id,
            transition_state_id=ts_id,
            reactant_structure=rct_structure,
            reactant_molecule_graph=rct_mg,
            reactant_molecule_graph_nometal=rct_mg_nometal,
            reactant_bonds=rct_bonds,
            reactant_bonds_nometal=rct_bonds_nometal,
            reactant_coord_hash=rct_coord_hash,
            reactant_species_hash=rct_species_hash,
            reactant_species_hash_nometal=rct_species_hash_nometal,
            reactant_energy=rct_e,
            reactant_zpe=rct_zpe,
            reactant_enthalpy=rct_h,
            reactant_entropy=rct_s,
            reactant_free_energy=rct_g,
            reactant_charges=rct_charges,
            product_structure=pro_structure,
            product_molecule_graph=pro_mg,
            product_molecule_graph_nometal=pro_mg_nometal,
            product_bonds=pro_bonds,
            product_bonds_nometal=pro_bonds_nometal,
            product_coord_hash=pro_coord_hash,
            product_species_hash=pro_species_hash,
            product_species_hash_nometal=pro_species_hash_nometal,
            product_energy=pro_e,
            product_zpe=pro_zpe,
            product_enthalpy=pro_h,
            product_entropy=pro_s,
            product_free_energy=pro_g,
            product_charges=pro_charges,
            transition_state_structure=ts_structure,
            transition_state_energy=ts_e,
            transition_state_zpe=ts_zpe,
            transition_state_enthalpy=ts_h,
            transition_state_entropy=ts_s,
            transition_state_free_energy=ts_g,
            transition_state_charges=ts_charges,
            dE=dE,
            dE_barrier=dE_barrier,
            dH=dH,
            dH_barrier=dH_barrier,
            dS=dS,
            dS_barrier=dS_barrier,
            dG=dG,
            dG_barrier=dG_barrier,
            bonds_broken=bonds_broken,
            bond_types_broken=bond_types_broken,
            bonds_broken_nometal=bonds_broken_nometal,
            bond_types_broken_nometal=bond_types_broken_nometal,
            bonds_formed=bonds_formed,
            bond_types_formed=bond_types_formed,
            bonds_formed_nometal=bonds_formed_nometal,
            bond_types_formed_nometal=bond_types_formed_nometal,
            **kwargs
        )
