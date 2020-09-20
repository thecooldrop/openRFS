"""
This module contains classes encapsulating computation of metrics used to evaluate the results of target tracking
algorithms.
"""

import scipy.optimize
import scipy.spatial


class MotMetrics:
    """
    The class encapsulating the computation of MOTA and MOTP target tracking metrics.
    """

    def compute(self, hypotheses_seq, ground_truths_seq, cutoff):
        """
        This function computes the MOTA and MOTP metrics and returns them together with intermediate values used in
        obtaining this metric. In total this method returns five values including the MOTA, MOTP, miss ratio,
        false positive ratio and miss-match ratios.

        The algorithm for computing these metrics is described in :

        Bernardin, K., Stiefelhagen, R. Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. J Image
        Video Proc 2008, 246309 (2008). https://doi.org/10.1155/2008/246309

        Expected inputs are sequences of maps, where each entry in the sequence corresponds to single measurement/frame
        of the detection process. Each map is expected to contain two keys, named "detection" and "label" mapped to
        matrices containing the predicted states and corresponding labels.

        :param hypotheses_seq: A sequence of hypotheses maps
        :param ground_truths_seq: A sequence of ground truth states and corresponding labels
        :param cutoff: A parameter denoting the maximum distance between ground-truth and hypothesis, under which they
                       can still be associated one to another. The parameter is described in cited paper.
        :return: A 5-tuple consisting of MOTA, MOTP, miss ratio, false-positive ratio and mismatch ratio
        """
        self._validate_inputs(hypotheses_seq, ground_truths_seq, cutoff)

        label_mapping = dict()
        misses = dict()
        false_positives = dict()
        mismatches = dict()
        object_count = dict()
        match_count = dict()
        match_distances = dict()

        for i in range(len(hypotheses_seq)):
            # find matching between previous and current timesteps
            previous_matching = label_mapping.get(i - 1, dict())
            current_true_labels = ground_truths_seq[i]["label"]
            current_hypothesis_labels = hypotheses_seq[i]["label"]
            current_true_states = ground_truths_seq[i]["detection"]
            current_hypothesis_states = hypotheses_seq[i]["detection"]
            distances = scipy.spatial.distance.cdist(current_true_states, current_hypothesis_states)
            new_matching = self._match_to(previous_matching, current_true_labels, current_hypothesis_labels, distances,
                                          cutoff)

            # find matches for remaining matched objects
            unmatched_true_labels = [label for label in current_true_labels if label not in new_matching]
            unmatched_hypothesis_labels = [label for label in current_hypothesis_labels if
                                           label not in new_matching.values()]
            unmatched_true_label_idxs = [current_true_labels.index(label) for label in unmatched_true_labels]
            unmatched_hypothesis_label_idxs = [current_hypothesis_labels.index(label) for label in
                                               unmatched_hypothesis_labels]
            unmatched_true_states = current_true_states[unmatched_true_label_idxs, :]
            unmatched_hypothesis_states = current_hypothesis_states[unmatched_hypothesis_label_idxs, :]
            unmatched_distances = scipy.spatial.distance.cdist(unmatched_true_states, unmatched_hypothesis_states)
            row_idx, col_idx = scipy.optimize.linear_sum_assignment(unmatched_distances)
            for c_row_idx, c_col_idx in zip(row_idx, col_idx):
                if unmatched_distances[c_row_idx, c_col_idx] < cutoff:
                    new_matching[unmatched_true_labels[c_row_idx]] = unmatched_hypothesis_labels[c_col_idx]

            miss_labels = [label for label in current_true_labels if label not in new_matching]
            false_positive_labels = [label for label in current_hypothesis_labels if label not in new_matching.values()]
            mismatch_labels = [label for label in current_true_labels
                               if label in previous_matching
                               and label in new_matching
                               and not previous_matching[label] == new_matching[label]]
            current_object_count = len(ground_truths_seq[i]["label"])
            current_match_count = len(new_matching)
            current_match_distances = [distances[current_true_labels.index(key), current_hypothesis_labels.index(value)]
                                       for key, value in new_matching.items()]

            misses[i] = len(miss_labels)
            false_positives[i] = len(false_positive_labels)
            mismatches[i] = len(mismatch_labels)
            object_count[i] = current_object_count
            match_count[i] = current_match_count
            match_distances[i] = current_match_distances

        total_misses = sum(misses.values())
        total_false_positives = sum(false_positives.values())
        total_mismatches = sum(mismatches.values())
        total_objects = sum(object_count.values())
        total_matches = sum(match_count.values())
        total_distances = sum([sum(val) for val in match_distances.values()])

        motp = total_distances / total_matches

        miss_ratio = total_misses / total_objects
        false_positives_ratio = total_false_positives / total_objects
        mismatch_ratio = total_mismatches / total_objects
        mota = 1 - (miss_ratio + false_positives_ratio + mismatch_ratio)

        return mota, motp, miss_ratio, false_positives_ratio, mismatch_ratio

    @staticmethod
    def _validate_inputs(hypotheses_seq, ground_truths_seq, cutoff):
        """
        Checks that the inputs are compatible and valid
        :param hypotheses_seq:
        :param ground_truths_seq:
        :param cutoff:
        :raise
        """
        if not len(hypotheses_seq) == ground_truths_seq:
            raise ValueError("The lengths of hypothesis and ground-truth sequences has to be the same")

        if not cutoff >= 0:
            raise ValueError("The cutoff distance has to be non-negative value")

    @staticmethod
    def _match_to(matching, key_labels, value_labels, distances, cutoff):
        """
        Computes a matching between previously matched labels and current labels. Note that the keys of the matching
        are labels attached to ground-truth states, while values are the labels attached to hypothesized states. Here is
        how this method works:

        For each of the ground truth states which are matched in given matching, check if they can still be matched to
        the same hypothesis label as in given matching. They can be associated to same hypothesis label only if this
        label still exists, and the mutual distance between the truth and hypothesis labeled states is less then cutoff.

        Note that the mutual distances between key_labels and value_labels are given in distances matrix.

        :param matching: A dictionary matching truth-labels to hypothesis labels
        :param key_labels: A list of ground-truth labels
        :param value_labels: A list of hypothesis-labels
        :param distances: Mutual distances between hypothesis and ground-truth labels
        :param cutoff: The maximum mutual distance between ground-truth and hypothesis at which they can still be
                       assigned to each other.
        :return: A dictionary mapping key_labels to value_labels under conditions described above
        """
        new_matching = dict()
        for key, val in matching.items():
            if key in key_labels and val in value_labels and distances[key_labels.index(key), value_labels.index(val)] < cutoff:
                new_matching[key] = val
        return new_matching
