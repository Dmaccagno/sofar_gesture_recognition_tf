# test script
import os
import ros_modules.service as S
from ros_modules.config import CONFIG as C
from ros_modules.file_utils import FilesUtil as F

current_stream_path = os.path.join(C.base_path, 'online_data\\34563456_1.txt')

# if os.path.exists(current_stream_path):
#     S.evaluate_online(current_stream_path, 1)
#     S.evaluate_online(current_stream_path, 2)
#     S.evaluate_online(current_stream_path, 5)
#     S.evaluate_online(current_stream_path, 6)
#     S.evaluate_online(current_stream_path, 7)
#     S.evaluate_online(current_stream_path, 8)
# else:
#     print("path does not exist...")

# e = F.load_result_file('online_results/34563456_1', 'err1.p')
# e = F.load_result_file('online_results/34563456_1', 'err2.p')
# e = F.load_result_file('online_results/34563456_1', 'err5.p')
# e = F.load_result_file('online_results/34563456_1', 'err6.p')
# e = F.load_result_file('online_results/34563456_1', 'err7.p')
e = F.load_result_file('online_results/34563456_1', 'err8.p')
peaks = S.detect_gesture(e, 8)
print(peaks)
