""" Core definition of document describing points on a Potential Energy Surface """
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import Field

from pymatgen.core.structure import Molecule
from pymatgen.analysis.molecule_matcher import MoleculeMatcher

from emmet.core.mpid import MPID, MPculeID
from emmet.core.utils import get_molecule_id
from emmet.core.material import CoreMoleculeDoc, PropertyOrigin
from emmet.core.structure import MoleculeMetadata

from rapter_eggs.settings import (
    JAGUAR_FUNCTIONAL_QUALITY_SCORES,
    JAGUAR_BASIS_QUALITY_SCORES,
    JAGUAR_SOLVENT_MODEL_QUALITY_SCORES
)
from rapter_eggs.schema.calc_types import CalcType, LevelOfTheory, TaskType
from rapter_eggs.schema.task import TaskDocument


__author__ = "Evan Spotte-Smith <ewcspottesmith@lbl.gov>"


def evaluate_lot(
    lot: Union[LevelOfTheory, str],
    funct_scores: Dict[str, int] = JAGUAR_FUNCTIONAL_QUALITY_SCORES,
    basis_scores: Dict[str, int] = JAGUAR_BASIS_QUALITY_SCORES,
    solvent_scores: Dict[str, int] = JAGUAR_SOLVENT_MODEL_QUALITY_SCORES,
):
    """
    Score the various components of a level of theory (functional, basis set,
    and solvent model), where a lower score is better than a higher score.

    :param lot: Level of theory to be evaluated
    :param funct_scores: Scores for various density functionals
    :param basis_scores: Scores for various basis sets
    :param solvent_scores: Scores for various implicit solvent models
    :return:
    """

    if isinstance(lot, LevelOfTheory):
        lot_comp = lot.value.split("/")
    else:
        lot_comp = lot.split("/")

    return (
        -1 * funct_scores.get(lot_comp[0], 0),
        -1 * basis_scores.get(lot_comp[1], 0),
        -1 * solvent_scores.get(lot_comp[2].split("(")[0], 0),
    )


def evaluate_task(
    task: TaskDocument,
    funct_scores: Dict[str, int] = JAGUAR_FUNCTIONAL_QUALITY_SCORES,
    basis_scores: Dict[str, int] = JAGUAR_BASIS_QUALITY_SCORES,
    solvent_scores: Dict[str, int] = JAGUAR_SOLVENT_MODEL_QUALITY_SCORES,
):
    """
    Helper function to order optimization calcs by
    - Level of theory
    - Electronic energy

    Note that lower scores indicate a higher quality.

    :param task: Task to be evaluated
    :param funct_scores: Scores for various density functionals
    :param basis_scores: Scores for various basis sets
    :param solvent_scores: Scores for various implicit solvent models
    :return:
    """

    lot = task.level_of_theory

    lot_eval = evaluate_lot(
        lot,
        funct_scores=funct_scores,
        basis_scores=basis_scores,
        solvent_scores=solvent_scores,
    )

    return (
        -1 * int(task.is_valid),
        sum(lot_eval),
        task.output.scf_energy,
    )


class PESPointDoc(CoreMoleculeDoc, MoleculeMetadata):

    calc_types: Mapping[str, CalcType] = Field(  # type: ignore
        None,
        description="Calculation types for all the calculations that make up this point on a PES",
    )
    task_types: Mapping[str, TaskType] = Field(
        None,
        description="Task types for all the calculations that make up this point on a PES",
    )
    levels_of_theory: Mapping[str, LevelOfTheory] = Field(
        None,
        description="Levels of theory types for all the calculations that make up this point on a PES",
    )

    origins: List[PropertyOrigin] = Field(
        None,
        description="List of property origins for tracking the provenance of properties",
    )

    entries: List[Dict[str, Any]] = Field(
        None,
        description="Dictionary representations of all task documents for this point on a PES",
    )

    best_entries: Mapping[LevelOfTheory, Dict[str, Any]] = Field(
        None,
        description="Mapping for tracking the best entries at each level of theory for Jaguar calculations",
    )

    similar_points: List[MPID] = Field(
        None,
        description="List of IDs with of points on a PES that are similar to this one",
    )

    frequencies: List[Optional[float]] = Field(
        None,
        description="Vibrational frequencies of this point on the PES (units: cm^-1)",
    )

    vibrational_frequency_modes: List[List[List[Optional[float]]]] = Field(
        None, description="Normal mode vectors of the molecule (units: Angstrom)"
    )

    freq_entry: Dict[str, Any] = Field(
        None,
        description="Dictionary representation of the task document used to obtain characteristic vibrational "
        "frequencies for this point on a PES",
    )

    coord_hash: str = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash using the atom coordinates as the graph "
        "node attribute.",
    )

    species_hash: str = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash using the atom species as the "
                    "graph node attribute."
    )

    species_hash_nometal: str = Field(
        None,
        description="Weisfeiler Lehman (WL) graph hash using the atom species as the "
                    "graph node attribute, where metal bonds are excluded."
    )

    @classmethod
    def from_tasks(
        cls,
        task_group: List[TaskDocument],
    ) -> "PESPointDoc":

        """
        Converts a group of tasks into one document describing a point on a
        Potential Energy Surface (PES)

        Args:
            task_group: List of task document
        """
        if len(task_group) == 0:
            raise Exception("Must have more than one task in the group.")

        entries = [t.entry for t in task_group]

        # Metadata
        last_updated = max(task.last_updated for task in task_group)
        calc_ids = list({task.calcid for task in task_group})

        deprecated_tasks = {task.calcid for task in task_group if not task.is_valid}
        levels_of_theory = {task.calcid: task.level_of_theory for task in task_group}
        task_types = {task.calcid: task.task_type for task in task_group}
        calc_types = {task.calcid: task.calc_type for task in task_group}

        initial_structures = list()
        for task in task_group:
            if isinstance(task.input["molecule"], Molecule):
                initial_structures.append(task.input["molecule"])
            else:
                mol = Molecule.from_dict(task.input["molecule"])
                initial_structures.append(mol)  # type: ignore

        # If we're dealing with single-atoms, process is much different
        if all([len(m) == 1 for m in initial_structures]):
            sorted_tasks = sorted(task_group, key=evaluate_task)

            coord_hash = sorted_tasks[0].coord_hash
            species_hash = sorted_tasks[0].species_hash
            species_hash_nometal = sorted_tasks[0].species_hash_nometal

            molecule = sorted_tasks[0].output.molecule

            molecule_id = "{}-{}-{}-{}".format(
                coord_hash,
                molecule.composition.alphabetical_formula.replace(' ', ''),
                str(int(molecule.charge)).replace("-", "m"),
                str(mol.spin_multiplicity)
            )

            # Output molecules. No geometry should change for a single atom
            initial_molecules = [molecule]

            # Deprecated
            deprecated = all(task.calcid in deprecated_tasks for task in task_group)

            # Origins
            origins = [
                PropertyOrigin(
                    name="molecule",
                    task_id=point_id,
                    last_updated=sorted_tasks[0].last_updated,
                )
            ]

            # No frequencies for single atom
            freq_entry = None
            frequencies = None
            frequency_modes = None

            # entries
            best_entries = {}
            all_lots = set(levels_of_theory.values())
            for lot in all_lots:
                relevant_calcs = sorted(
                    [
                        doc
                        for doc in task_group
                        if doc.level_of_theory == lot and doc.is_valid
                    ],
                    key=evaluate_task,
                )

                if len(relevant_calcs) > 0:
                    best_task_doc = relevant_calcs[0]
                    entry = best_task_doc.entry
                    entry["calcid"] = entry["entry_id"]
                    entry["entry_id"] = point_id
                    best_entries[lot] = entry

        else:
            geometry_optimizations = [
                task
                for task in task_group
                if task.task_type
                in [
                    TaskType.Geometry_Optimization,
                    TaskType.Transition_State_Geometry_Optimization,
                ]  # type: ignore
            ]

            best_structure_calc = sorted(geometry_optimizations, key=evaluate_task)[0]
            coord_hash = best_structure_calc.coord_hash
            species_hash = best_structure_calc.species_hash
            species_hash_nometal = best_structure_calc.species_hash_nometal
            molecule = best_structure_calc.output.molecule

            molecule_id = "{}-{}-{}-{}".format(
                coord_hash,
                molecule.composition.alphabetical_formula.replace(' ', ''),
                str(int(molecule.charge)).replace("-", "m"),
                str(mol.spin_multiplicity)
            )

            freq_tasks = sorted(
                [
                    task
                    for task in task_group
                    if task.input["gen_variables"].get("ifreq", 0) != 0 and task.success
                ],
                key=evaluate_task,
            )
            if len(freq_tasks) == 0:
                frequencies = None
                frequency_modes = None
                freq_entry = None
            else:
                frequencies = freq_tasks[0].output.frequencies
                frequency_modes = freq_tasks[0].output.vibrational_frequency_modes
                freq_entry = freq_tasks[0].entry

            mm = MoleculeMatcher()
            initial_molecules = [
                group[0] for group in mm.group_molecules(initial_structures)
            ]

            # Deprecated
            deprecated = all(
                task.calcid in deprecated_tasks for task in geometry_optimizations
            )
            deprecated = deprecated or best_structure_calc.calcid in deprecated_tasks

            # Origins
            origins = [
                PropertyOrigin(
                    name="molecule",
                    task_id=best_structure_calc.calcid,
                    last_updated=best_structure_calc.last_updated,
                )
            ]

            # entries
            best_entries = dict()
            all_lots = set(levels_of_theory.values())
            for lot in all_lots:
                relevant_calcs = sorted(
                    [
                        doc
                        for doc in geometry_optimizations
                        if doc.level_of_theory == lot and doc.is_valid
                    ],
                    key=evaluate_task,
                )

                if len(relevant_calcs) > 0:
                    best_task_doc = relevant_calcs[0]
                    entry = best_task_doc.entry
                    best_entries[lot] = entry

        for entry in entries:
            entry["entry_id"] = molecule_id

        return cls.from_molecule(
            molecule=molecule,
            freq_entry=freq_entry,
            frequencies=frequencies,
            vibrational_frequency_modes=frequency_modes,
            molecule_id=molecule_id,
            initial_molecules=initial_molecules,
            last_updated=last_updated,
            task_ids=calc_ids,
            calc_types=calc_types,
            levels_of_theory=levels_of_theory,
            task_types=task_types,
            deprecated=deprecated,
            deprecated_tasks=deprecated_tasks,
            origins=origins,
            entries=entries,
            best_entries=best_entries,
            coord_hash=coord_hash,
            species_hash=species_hash,
            species_hash_nometal=species_hash_nometal
        )

    @classmethod
    def construct_deprecated_pes_point(
        cls,
        task_group: List[TaskDocument],
    ) -> "PESPointDoc":
        """
        Converts a group of tasks into a deprecated PESPointDoc

        Args:
            task_group: List of task document
        """
        if len(task_group) == 0:
            raise Exception("Must have more than one task in the group.")

        # Metadata
        last_updated = max(task.last_updated for task in task_group)
        created_at = min(task.last_updated for task in task_group)
        calc_ids = list({task.calcid for task in task_group})

        deprecated_tasks = {task.calcid for task in task_group}
        levels_of_theory = {task.calcid: task.level_of_theory for task in task_group}
        task_types = {task.calcid: task.task_type for task in task_group}
        calc_types = {task.calcid: task.calc_type for task in task_group}

        # Choose arbitrary task
        chosen_task = sorted(task_group, key=lambda x: x.calcid)[0]

        if isinstance(chosen_task.input["molecule"], dict):
            molecule = Molecule.from_dict(chosen_task.input["molecule"])
        else:
            molecule = chosen_task.input["molecule"]

        coord_hash = chosen_task.coord_hash
        species_hash = chosen_task.species_hash
        species_hash_nometal = chosen_task.species_hash_nometal

        # Molecule ID
        molecule_id = "{}-{}-{}-{}".format(
            coord_hash,
            molecule.composition.alphabetical_formula.replace(' ', ''),
            str(int(molecule.charge)).replace("-", "m"),
            str(mol.spin_multiplicity)
        )

        # Deprecated
        deprecated = True

        return cls.from_molecule(
            molecule=molecule,
            molecule_id=molecule_id,
            last_updated=last_updated,
            created_at=created_at,
            task_ids=calc_ids,
            calc_types=calc_types,
            levels_of_theory=levels_of_theory,
            task_types=task_types,
            deprecated=deprecated,
            deprecated_tasks=deprecated_tasks,
        )


def best_lot(
    doc: PESPointDoc,
    funct_scores: Dict[str, int] = JAGUAR_FUNCTIONAL_QUALITY_SCORES,
    basis_scores: Dict[str, int] = JAGUAR_BASIS_QUALITY_SCORES,
    solvent_scores: Dict[str, int] = JAGUAR_SOLVENT_MODEL_QUALITY_SCORES,
) -> LevelOfTheory:
    """

    Return the best level of theory used within a MoleculeDoc

    :param doc: PESPointDoc
    :param funct_scores: Scores for various density functionals
    :param basis_scores: Scores for various basis sets
    :param solvent_scores: Scores for various implicit solvent models

    :return: LevelOfTheory
    """

    sorted_lots = sorted(
        doc.best_entries.keys(),
        key=lambda x: evaluate_lot(x, funct_scores, basis_scores, solvent_scores),
    )

    return sorted_lots[0]
