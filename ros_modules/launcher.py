# test script
import os
import ros_modules.service as S
from ros_modules.config import CONFIG as C

current_stream_path = os.path.join(C.base_path, 'online_data\\34563456_1.txt')

if os.path.exists(current_stream_path):
    S.evaluate_online(current_stream_path, 1)
    S.evaluate_online(current_stream_path, 2)
    S.evaluate_online(current_stream_path, 5)
    S.evaluate_online(current_stream_path, 6)
    S.evaluate_online(current_stream_path, 7)
    S.evaluate_online(current_stream_path, 8)
else:
    print("path does not exist...")

