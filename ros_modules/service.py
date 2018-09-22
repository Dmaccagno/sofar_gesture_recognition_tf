from ros_modules.file_utils import FilesUtil as F
from ros_modules.config import CONFIG as C
import ros_modules.rnn_module as R
import numpy as np
import os


# todo add some comments
def evaluate_online(stream_path, gesture_id):
    results_folder_name = os.path.basename(os.path.splitext(stream_path)[0])
    full_path = C.base_path + '/online_results/' + results_folder_name

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    error_stream = list()
    stream = F.generate_data_set_from_file(stream_path)
    start_index = 0
    end_index = C.get_size_dim(gesture_id)
    stream_len = len(stream)
    v_prediction_old = np.full((3,), np.nan)
    for i in range(stream_len):
        if end_index < stream_len:
            stream_batch = stream[start_index:end_index]
            v_current, v_prediction = R.predict_next_values(stream_batch.values, str(gesture_id),
                                                            C.get_neurons_dim(gesture_id))
            stream_batch, start_index, end_index = \
                F.get_next_window(stream_batch, dimension=150, start=start_index, end=end_index)
            if not all(np.isnan(i) for i in v_prediction_old):
                x_error = v_prediction_old[0] - v_current[0]
                y_error = v_prediction_old[1] - v_current[1]
                z_error = v_prediction_old[2] - v_current[2]
                current_error = np.sqrt(x_error * x_error + y_error * y_error + z_error * z_error)
                error_stream.append(current_error)
            v_prediction_old = v_prediction
            print("errors has len.." + len(error_stream).__str__())
    F.save_results_to_file(error_stream, full_path, gesture_id)
    return error_stream
