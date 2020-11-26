import numpy as np

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

import EIkMeans_lib as eikm


class HANG(BaseDriftDetector):
    """ Early Drift Detection Method.

    Notes
    -----
    EDDM (Early Drift Detection Method) [1]_ aims to improve the
    detection rate of gradual concept drift in DDM, while keeping
    a good performance against abrupt concept drift.

    This method works by keeping track of the average distance
    between two errors instead of only the error rate. For this,
    it is necessary to keep track of the running average distance
    and the running standard deviation, as well as the maximum
    distance and the maximum standard deviation.

    The algorithm works similarly to the DDM algorithm, by keeping
    track of statistics only. It works with the running average
    distance (:math:`p_i^'`) and the running standard deviation (:math:`s_i^'`), as
    well as :math:`p^'_{max}` and :math:`s^'_{max}`, which are the values of :math:`p_i^'` and :math:`s_i^'`
    when :math:`(p_i^' + 2 * s_i^')` reaches its maximum.

    Like DDM, there are two threshold values that define the
    borderline between no change, warning zone, and drift detected.
    These are as follows:

    * if :math:`(p_i^' + 2 * s_i^')/(p^'_{max} + 2 * s^'_{max}) < \alpha` -> Warning zone
    * if :math:`(p_i^' + 2 * s_i^')/(p^'_{max} + 2 * s^'_{max}) < \beta` -> Change detected

    :math:`\alpha` and :math:`\beta` are set to 0.95 and 0.9, respectively.

    References
    ----------
    .. [1] Early Drift Detection Method. Manuel Baena-Garcia, Jose Del Campo-Avila,
       Raúl Fidalgo, Albert Bifet, Ricard Gavalda, Rafael Morales-Bueno. In Fourth
       International Workshop on Knowledge Discovery from Data Streams, 2006.

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.drift_detection.eddm import EDDM
    >>> eddm = EDDM()
    >>> # Simulating a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 1500, simulating an 
    >>> # increase in error rate
    >>> for i in range(999, 1500):
    ...     data_stream[i] = 0
    >>> # Adding stream elements to EDDM and verifying if drift occurred
    >>> for i in range(2000):
    ...     eddm.add_element(data_stream[i])
    ...     if eddm.detected_warning_zone():
    ...         print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    ...     if eddm.detected_change():
    ...         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

    """

    def __init__(self, the_k = 4, initail_window_len=200, compare_window_len=30, m_test=20, the_alpha=0.05):
        super().__init__()
        self.the_alpha = the_alpha
        self.the_k = the_k
        self.initial_window_len = initail_window_len
        self.initial_window = []
        self.initial_k_list = None
        self.compare_window_len = compare_window_len
        self.compare_window_one = []
        self.W_one_k_list = None
        self.W_one_unique_count = None
        self.compare_window_two = []
        self.W_two_k_list = None
        self.W_two_unique_count = None
        self.compare_window_three = []
        self.W_three_k_list = None
        self.W_three_unique_count = None
        self.m_test = m_test
        self.sample_num = 0
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.the_alpha = 0.05
        self.the_k = 4
        self.initial_window_len = 200
        self.initial_window = []
        self.initial_k_list = None
        self.compare_window_len = 30
        self.compare_window_one = []
        self.W_one_k_list = None
        self.W_one_unique_count = None
        self.compare_window_two = []
        self.W_two_k_list = None
        self.W_two_unique_count = None
        self.compare_window_three = []
        self.W_three_k_list = None
        self.W_three_unique_count = None
        self.m_test = 20
        self.sample_num = 0

    def add_element(self, prediction):
        """ Add a new element to the statistics
        
        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).
        
        Returns
        -------
        EDDM
            self
        
        Notes
        -----
        After calling this method, to verify if change was detected or if  
        the learner is in the warning zone, one should call the super method 
        detected_change, which returns True if concept drift was detected and
        False otherwise.
         
        """

        if self.in_concept_change:
            self.reset()

        self.in_concept_change = False

        self.sample_num += 1

        prediction = [prediction, 0.0]
        print("prediction: ", prediction)
        if self.sample_num < self.initial_window_len:
            self.initial_window.append(prediction)
        else:
            # 学习模型
            cp_inst = eikm.EIkMeans(self.the_k)
            self.initial_window = np.array(self.initial_window)
            self.initial_k_list = cp_inst.build_partition(self.initial_window, self.m_test)
            print("initial_k_list: ", self.initial_k_list)
            if self.sample_num < self.initial_window_len + self.compare_window_len:
                self.W_one_unique_count = cp_inst.getHist([prediction], self.initial_k_list,
                                                                             self.W_one_unique_count)
            else:
                if self.sample_num < self.initial_window_len + 2 * self.compare_window_len:
                    self.W_two_unique_count = cp_inst.getHist([prediction], self.initial_k_list,
                                                                                 self.W_two_unique_count)
                    self.compare_window_two.append(prediction)
                else:
                    if self.sample_num < self.initial_window_len + 3 * self.compare_window_len:
                        self.W_three_unique_count = cp_inst.getHist([prediction], self.initial_k_list,
                                                                                         self.W_three_unique_count)
                        results = np.zeros(1)
                        results[0], best_location_type, updated_W_one_unique_count, updated_W_two_unique_count = \
                            cp_inst.drift_detection_new(self.initial_k_list, self.W_one_unique_count,
                                                        self.W_two_unique_count, self.compare_window_two[0],
                                                        [prediction], self.the_alpha)
                        if results[0] == 1:
                            self.in_concept_change = True
                        else:
                            if best_location_type == 0:
                                self.W_one_unique_count = updated_W_one_unique_count
                                self.W_two_unique_count = updated_W_two_unique_count
                                self.compare_window_two = self.compare_window_two.pop(0)
                            else:
                                self.W_one_unique_count = updated_W_one_unique_count
                                self.W_two_unique_count = updated_W_two_unique_count
                    else:
                        self.W_one_unique_count = cp_inst.combineWindow(self.W_one_unique_count, self.W_two_unique_count)
                        self.sample_num -= self.compare_window_len * 2


        # if prediction == 1.0:
        #     self.in_warning_zone = False
        #     self.delay = 0
        #     self.m_num_errors += 1
        #     self.m_lastd = self.m_d
        #     self.m_d = self.m_n - 1
        #     distance = self.m_d - self.m_lastd
        #     old_mean = self.m_mean
        #     self.m_mean = self.m_mean + (float(distance) - self.m_mean) / self.m_num_errors
        #     self.estimation = self.m_mean
        #     self.m_std_temp = self.m_std_temp + (distance - self.m_mean) * (distance - old_mean)
        #     std = np.sqrt(self.m_std_temp/self.m_num_errors)
        #     m2s = self.m_mean + 2 * std
        #
        #     if self.m_n < self.FDDM_MIN_NUM_INSTANCES:
        #         return
        #
        #     if m2s > self.m_m2s_max:
        #         self.m_m2s_max = m2s
        #     else:
        #         p = m2s / self.m_m2s_max
        #         if (self.m_num_errors > self.m_min_num_errors) and (p < self.FDDM_OUTCONTROL):
        #             self.in_concept_change = True
        #
        #         elif (self.m_num_errors > self.m_min_num_errors) and (p < self.FDDM_WARNING):
        #             self.in_warning_zone = True
        #
        #         else:
        #             self.in_warning_zone = False
