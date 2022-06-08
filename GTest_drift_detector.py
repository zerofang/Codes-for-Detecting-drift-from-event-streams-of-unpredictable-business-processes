import copy

import numpy as np
import pandas as pd
import math
from scipy.stats import chi2_contingency


class DriftDetector:
    def __init__(self, _case_id_col, _activity_col, _label_col, _timestamp_col, _label_num, _init_window_size=1100,
                 _GTest_threshold=0.05, _GTest_phi=0.5):
        """
        Parameter initialization
        :param _case_id_col: the column name of 'case id' in business process log (the log must be in pandas dataframe format)
        :param _activity_col: the column name of 'activity' in business process log
        :param _label_col: the column name of the label (like next activity) in business process log
        :param _timestamp_col: the column name of timestamp in business process log
        :param _label_num:  how many kinds of labels (like the number of activities) are in business process log
        :param _init_window_size: the initial window size
        :param _GTest_threshold: determine the sensitivity of drift detection algorithm, smaller = more sensitive
        :param _GTest_phi: determine the sensitivity or latency of drift reporting, smaller = more sensitive and smaller latency
        """
        self.init_window_size = _init_window_size
        self.GTest_threshold = _GTest_threshold
        self.GTest_phi = _GTest_phi
        self.case_id_col = _case_id_col
        self.activity_col = _activity_col
        self.label_col = _label_col
        self.timestamp_col = _timestamp_col
        self.label_num = _label_num

        self.w = self.init_window_size
        self.drift_points = []
        self.pbt_event = -1
        self.pbt_win_size = -1
        self.pbt_len = 0

    def extract_next_activity(self, group):
        group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        group[self.label_col] = group[self.activity_col].shift(axis=0, periods=-1).fillna("#")
        return group

    def get_relation_matrix(self, sub_log, activity_to_index):
        sub_log = sub_log[[self.case_id_col, self.activity_col, self.timestamp_col]]
        sub_log = sub_log.groupby(self.case_id_col, as_index=False).apply(self.extract_next_activity)
        sub_log.drop(sub_log[sub_log[self.label_col] == "#"].index, inplace=True)
        sub_log_sorted = sub_log.sort_values(by=[self.case_id_col, self.timestamp_col], ascending=(True, True))

        relation_matrix = np.zeros((self.label_num, self.label_num))
        for index, row in sub_log_sorted.iterrows():
            prev_activity = activity_to_index[row[self.activity_col]]
            next_activity = activity_to_index[row[self.label_col]]
            relation_matrix[prev_activity][next_activity] += 1

        return relation_matrix

    def build_contingency_matrix_dict(self, sub_log, activity_to_index):
        """
        build contingency matrix dict
        :param sub_log: the log must be in pandas dataframe format
        :param activity_to_index: a dict mapping activity names to numeric indexes
        :return: reference dictionary and detection dictionary
        """
        ref_sub_log = sub_log.iloc[:self.w]
        det_sub_log = sub_log.iloc[self.w:]

        ref_label_num = len(ref_sub_log[self.activity_col].unique())
        det_label_num = len(det_sub_log[self.activity_col].unique())
        self.w = max(pow(max(ref_label_num, det_label_num), 2) * 5, 500)

        ref_relation_matrix = self.get_relation_matrix(ref_sub_log, activity_to_index)
        det_relation_matrix = self.get_relation_matrix(det_sub_log, activity_to_index)

        ref_relation_dict = {}
        det_relation_dict = {}

        for i in range(0, self.label_num):
            for j in range(0, self.label_num):
                if ref_relation_matrix[i][j] == 0 and det_relation_matrix[i][j] == 0 and ref_relation_matrix[j][
                    i] == 0 and det_relation_matrix[j][i] == 0:
                    continue

                if i == j and ref_relation_matrix[i][j] > 0:
                    key = str(i) + "_loop_" + str(j)
                    ref_relation_dict[key] = ref_relation_matrix[i][j]

                if i == j and det_relation_matrix[i][j] > 0:
                    key = str(i) + "_loop_" + str(j)
                    det_relation_dict[key] = det_relation_matrix[i][j]

                if i == j:
                    continue

                if ref_relation_matrix[i][j] > 10 and ref_relation_matrix[j][i] <= 10:
                    key = str(i) + "_db_" + str(j)
                    ref_relation_dict[key] = ref_relation_matrix[i][j]

                if det_relation_matrix[i][j] > 10 and det_relation_matrix[j][i] <= 10:
                    key = str(i) + "_db_" + str(j)
                    det_relation_dict[key] = det_relation_matrix[i][j]

                if ref_relation_matrix[i][j] > 10 and ref_relation_matrix[j][i] > 10:
                    key = str(min(i, j)) + "_pl_" + str(max(i, j))
                    ref_relation_dict[key] = min(ref_relation_matrix[i][j], ref_relation_matrix[j][i])

                if det_relation_matrix[i][j] > 10 and det_relation_matrix[j][i] > 10:
                    key = str(min(i, j)) + "_pl_" + str(max(i, j))
                    det_relation_dict[key] = min(det_relation_matrix[i][j], det_relation_matrix[j][i])

        for key in ref_relation_dict.keys():
            if key not in det_relation_dict.keys():
                det_relation_dict[key] = 0

        for key in det_relation_dict.keys():
            if key not in ref_relation_dict.keys():
                ref_relation_dict[key] = 0
        return ref_relation_dict, det_relation_dict

    def GTest(self, ref_list, det_list):
        try:
            _, p, _, _ = chi2_contingency(np.array([ref_list, det_list]))
        except:
            ref_sum = sum(ref_list)
            det_sum = sum(det_list)
            return 1 / (ref_sum + det_sum)
        else:
            return p

    def detect(self, ref_relation_dict, det_relation_dict, event_id):
        ref_list = []
        det_list = []
        for key, value in ref_relation_dict.items():
            ref_list.append(value)
            det_list.append(det_relation_dict[key])

        p_value = self.GTest(ref_list, det_list)
        if p_value < self.GTest_threshold:
            # print("-------found a drift--------")
            # print(p_value)
            # print(event_id)
            self.pbt_len += 1
            if self.pbt_event == -1:
                self.pbt_event = event_id
                self.pbt_win_size = self.w
            # The latency of drift reporting is about math.floor(self.GTest_phi * self.w)
            if self.pbt_len == math.floor(self.GTest_phi * self.w):
                self.drift_points.append(event_id - self.pbt_len)
                return True, event_id - self.pbt_len
        else:
            self.pbt_len = 0
            self.pbt_event = -1
            self.pbt_win_size = -1
        return False, self.pbt_len

    def get_drift_points(self):
        return self.drift_points

    def get_start_event_id(self, event_id):
        return event_id - 2 * self.w


if __name__ == '__main__':
    df = pd.read_csv("Loan_Assessment_Concept_Drift.csv")
    df_len = df.shape[0]
    detector = DriftDetector("Case_ID", "Activity", "Complete_Timestamp", "Next_Activity", 20)
    index_to_activity = dict(enumerate(list(df["Activity"].unique())))
    activity_to_index = dict(zip(index_to_activity.values(), index_to_activity.keys()))
    print(activity_to_index)

    # The actual drift points are [4244, 7870, 12198, 18364, 22611, 26629, 30876, 35162, 39332, 43415, 50644]
    for current_event_id in range(2500, df_len + 1):
        start_event_id = detector.get_start_event_id(current_event_id)
        sub_log = df.iloc[start_event_id:current_event_id, :].copy().reset_index()
        ref_relation_dict, det_relation_dict = detector.build_contingency_matrix_dict(sub_log, activity_to_index)
        is_drift, drift_point = detector.detect(ref_relation_dict, det_relation_dict, current_event_id)
        if is_drift:
            print("Found a drift point: " + str(drift_point) + ", current event id = " + str(current_event_id))

    print(detector.get_drift_points())
