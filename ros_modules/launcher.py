# test script
import os
import ros_modules.service as S
from ros_modules.config import CONFIG as C

current_stream_path = os.path.join(C.base_path, 'online_data/34563456_1.txt')

e = S.evaluate_online(current_stream_path, 1)

