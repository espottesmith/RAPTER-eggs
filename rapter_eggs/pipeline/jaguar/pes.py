from datetime import datetime
from itertools import chain
from math import ceil
from typing import Dict, Iterable, Iterator, List, Optional


from maggma.builders import Builder
from maggma.stores import Store
from maggma.utils import grouper

from emmet.builders.settings import EmmetBuildSettings
from emmet.core.utils import group_molecules, jsanitize
from emmet.core.jaguar.pes import (
    best_lot,
    evaluate_lot,
    PESPointDoc,
)
from emmet.core.jaguar.task import TaskDocument


__author__ = "Evan Spotte-Smith <ewcspottesmith@lbl.gov>"


SETTINGS = EmmetBuildSettings()


def evaluate_point(
    pes_point: PESPointDoc,
    funct_scores: Dict[str, int] = SETTINGS.JAGUAR_FUNCTIONAL_QUALITY_SCORES,
    basis_scores: Dict[str, int] = SETTINGS.JAGUAR_BASIS_QUALITY_SCORES,
    solvent_scores: Dict[str, int] = SETTINGS.JAGUAR_SOLVENT_MODEL_QUALITY_SCORES,
):
    """
    Helper function to order optimization calcs by
    - Level of theory
    - Electronic energy

    :param mol_doc: Molecule to be evaluated
    :param funct_scores: Scores for various density functionals
    :param basis_scores: Scores for various basis sets
    :param solvent_scores: Scores for various implicit solvent models
    :return:
    """

    best = best_lot(pes_point, funct_scores, basis_scores, solvent_scores)

    lot_eval = evaluate_lot(best, funct_scores, basis_scores, solvent_scores)

    return (
        -1 * int(pes_point.deprecated),
        sum(lot_eval),
        pes_point.best_entries[best]["energy"],
    )


def filter_and_group_tasks(
    tasks: List[TaskDocument], settings: EmmetBuildSettings
) -> Iterator[List[TaskDocument]]:
    """
    Groups tasks by identical structure
    """

    filtered_tasks = [
        task
        for task in tasks
        if any(
            allowed_type is task.task_type
            for allowed_type in settings.JAGUAR_ALLOWED_TASK_TYPES
        )
    ]

    molecules = list()

    for idx, task in enumerate(filtered_tasks):
        if task.output.molecule:
            m = task.output.molecule
        else:
            m = task.input["molecule"]
        m.index = idx  # type: ignore
        molecules.append(m)

    grouped_molecules = group_molecules(molecules)
    for group in grouped_molecules:
        grouped_tasks = [filtered_tasks[mol.index] for mol in group]  # type: ignore
        yield grouped_tasks


class PESPointBuilder(Builder):
    """
    The PESPointBuilder matches Jaguar task documents that represent minima of
    a potential energy surface (no imaginary frequencies, or one negligible
    imaginary frequency) or transition-states (one imaginary frequency, or two with 
    one very small imaginary mode) by composition and collects tasks associated with
    identical structures.

    The process is as follows:

        1.) Find all documents with the same formula
        2.) Select only task documents for the task_types we can select
        properties from
        3.) Aggregate task documents based on nuclear geometry
        4.) Create PESPointDocs, filtering based on the characteristic
        frequencies calculated in the tasks
    """

    def __init__(
        self,
        tasks: Store,
        minima: Store,
        ts: Store,
        query: Optional[Dict] = None,
        settings: Optional[EmmetBuildSettings] = None,
        negative_threshold: float = -75.0,
        **kwargs,
    ):
        """
        Args:
            tasks:  Store of task documents
            minima: Store of PES minima to prepare
            ts: Store of transition-states (TS) to prepare
            query: dictionary to limit tasks to be analyzed
            settings: EmmetSettings to use in the build process
            negative_threshold: Threshold for imaginary frequencies. Points
                with one imaginary frequency >= this value will be considered
                as valid.
        """

        self.tasks = tasks
        self.minima = minima
        self.ts = ts
        self.query = query if query else dict()
        self.settings = EmmetBuildSettings.autoload(settings)
        self.negative_threshold = negative_threshold
        self.kwargs = kwargs

        super().__init__(sources=[tasks], targets=[minima, ts], **kwargs)

    def ensure_indexes(self):
        """
        Ensures indices on the collections needed for building
        """

        # Basic search index for tasks
        self.tasks.ensure_index("calcid")
        self.tasks.ensure_index("last_updated")
        self.tasks.ensure_index("success")
        self.tasks.ensure_index("formula_alphabetical")
        self.tasks.ensure_index("coord_hash")
        self.tasks.ensure_index("species_hash")
        self.tasks.ensure_index("species_hash_nometal")

        # Search index for minima
        self.minima.ensure_index("molecule_id")
        self.minima.ensure_index("last_updated")
        self.minima.ensure_index("task_ids")
        self.minima.ensure_index("formula_alphabetical")
        self.minima.ensure_index("coord_hash")
        self.minima.ensure_index("species_hash")
        self.minima.ensure_index("species_hash_nometal")

        # Search index for ts
        self.ts.ensure_index("molecule_id")
        self.ts.ensure_index("last_updated")
        self.ts.ensure_index("task_ids")
        self.ts.ensure_index("formula_alphabetical")
        self.ts.ensure_index("coord_hash")
        self.ts.ensure_index("species_hash")
        self.ts.ensure_index("species_hash_nometal")

    def prechunk(self, number_splits: int) -> Iterable[Dict]:  # pragma: no cover
        """Prechunk the PESPointBuilder for distributed computation"""

        temp_query = dict(self.query)
        temp_query["success"] = True

        self.logger.info("Finding tasks to process")
        all_tasks = list(
            self.tasks.query(temp_query, [self.tasks.key, "species_hash"])
        )

        processed_tasks = set(self.minima.distinct("task_ids")) | set(self.ts.distinct("task_ids"))
        to_process_tasks = {d[self.tasks.key] for d in all_tasks} - processed_tasks
        to_process_hashes = set()
        for d in all_tasks:
            if d[self.tasks.key] in to_process_tasks:
                hash = d.get("species_hash")
                if hash:
                    to_process_hashes.add(hash)

        N = ceil(len(to_process_hashes) / number_splits)

        for hash_chunk in grouper(to_process_hashes, N):
            yield {"query": {"species_hash": {"$in": list(hash_chunk)}}}

    def get_items(self) -> Iterator[List[Dict]]:
        """
        Gets all items to process into PESPointBuilder.
        This does no datetime checking; relying on on whether
        task_ids are included in the minima Store

        Returns:
            generator or list relevant tasks and molecules to process into documents
        """

        self.logger.info("PES point builder started")
        self.logger.info(
            f"Allowed task types: {[task_type.value for task_type in self.settings.JAGUAR_ALLOWED_TASK_TYPES]}"
        )

        self.logger.info("Setting indexes")
        self.ensure_indexes()

        # Save timestamp to mark buildtime
        self.timestamp = datetime.utcnow()

        # Get all processed tasks
        temp_query = dict(self.query)
        temp_query["success"] = True

        self.logger.info("Finding tasks to process")
        all_tasks = list(
            self.tasks.query(temp_query, [self.tasks.key, "species_hash"])
        )

        processed_tasks = set(self.minima.distinct("task_ids")) | set(self.ts.distinct("task_ids"))
        to_process_tasks = {d[self.tasks.key] for d in all_tasks} - processed_tasks
        to_process_hashes = set()
        for d in all_tasks:
            if d[self.tasks.key] in to_process_tasks:
                hash = d.get("species_hash")
                if hash:
                    to_process_hashes.add(hash)

        self.logger.info(f"Found {len(to_process_tasks)} unprocessed tasks")
        self.logger.info(f"Found {len(to_process_hashes)} unprocessed structures")

        # Set total for builder bars to have a total
        self.total = len(to_process_hashes)

        projected_fields = [
            "calcid",
            "tags",
            "additional_data",
            "charge",
            "spin_multiplicity",
            "nelectrons",
            "errors",
            "success",
            "walltime",
            "input",
            "output",
            "last_updated",
            "job_type",
            "formula_alphabetical",
            "coord_hash",
            "species_hash",
            "species_hash_nometal",
            "name"
        ]

        for hash in to_process_hashes:
            tasks_query = dict(temp_query)
            tasks_query["species_hash"] = hash
            tasks = list(
                self.tasks.query(criteria=tasks_query, properties=projected_fields)
            )

            to_send = list()
            for t in tasks:
                # TODO: Validation
                # basic validation here ensures that tasks that do not have the requisite
                # information to form TaskDocuments do not snag building
                try:
                    task = TaskDocument(**t)
                    t["is_valid"] = True
                    to_send.append(task)
                except Exception as e:
                    self.logger.info(
                        f"Processing task {t['calcid']} failed with Exception - {e}"
                    )
                    t["is_valid"] = False

            yield to_send

    def process_item(self, items: List[TaskDocument]) -> List[Dict]:
        """
        Process the tasks into a PESPointDoc

        Args:
            tasks [dict] : a list of task docs

        Returns:
            [dict] : a list of new PESPointDoc
        """

        tasks = items
        hash = tasks[0].species_hash
        task_ids = [task.calcid for task in tasks]
        self.logger.debug(f"Processing {hash} : {task_ids}")
        docs = list()

        for group in filter_and_group_tasks(tasks, self.settings):
            try:
                docs.append(PESPointDoc.from_tasks(group))
            except Exception as e:
                failed_ids = list({t_.calcid for t_ in group})
                doc = PESPointDoc.construct_deprecated_pes_point(group)
                doc.warnings.append(str(e))
                docs.append(doc)
                self.logger.warn(
                    f"Failed making PESPointDoc for {failed_ids}."
                    f" Inserted as deprecated doc: {doc.molecule_id}"
                )

        self.logger.debug(f"Produced {len(docs)} docs for {hash}")

        return jsanitize([doc.dict() for doc in docs], allow_bson=True)

    def update_targets(self, items: List[Dict]):
        """
        Inserts the new minima into the minima and ts collections

        Args:
            items [[dict]]: A list of PESPointDocs to update
        """

        docs = list(chain.from_iterable(items))  # type: ignore

        true_minima = list()
        true_ts = list()

        for item in docs:
            item.update({"_bt": self.timestamp})
            frequencies = item.get("frequencies")

            # For minima
            # Assume a species with no frequencies is a valid minimum
            if frequencies is None or len(frequencies) < 2:
                true_minima.append(item)
            # All positive, or one small negative frequency
            elif frequencies[0] >= self.negative_threshold and frequencies[1] > 0:
                true_minima.append(item)
            # For TS
            elif frequencies[0] <= self.negative_threshold:
                if len(frequencies) == 2 and frequencies[1] >= self.negative_threshold:
                    true_ts.append(item)
                elif frequencies[1] >= self.negative_threshold and frequencies[2] > 0:
                    true_ts.append(item)
            else:
                continue

        molecule_ids_minima = list({item["molecule_id"] for item in true_minima})
        molecule_ids_ts = list({item["molecule_id"] for item in true_ts})

        if len(true_minima) > 0:
            self.logger.info(f"Updating {len(true_minima)} minima")
            self.minima.remove_docs({self.minima.key: {"$in": molecule_ids_minima}})
            self.minima.update(
                docs=true_minima,
                key=["molecule_id"],
            )
        else:
            self.logger.info("No PES minima to update")

        if len(true_ts) > 0:
            self.logger.info(f"Updating {len(true_ts)} ts")
            self.ts.remove_docs({self.ts.key: {"$in": molecule_ids_ts}})
            self.ts.update(
                docs=true_ts,
                key=["molecule_id"],
            )
        else:
            self.logger.info("No TS to update")
