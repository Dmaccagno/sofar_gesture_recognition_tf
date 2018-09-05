# FilesUtil.save_results_to_file(err1, 1)
# FilesUtil.save_results_to_file(err2, 2)
# FilesUtil.save_results_to_file(err5, 5)
# FilesUtil.save_results_to_file(err6, 6)
# FilesUtil.save_results_to_file(err7, 7)
# FilesUtil.save_results_to_file(err8, 8)

# _len = len(err1)

# plt.figure(1)
# a1 = plt.subplot(611)
# a1.set_title("x-data")
# a1.plot(np.arange(len(err1)), err1)
# a2 = plt.subplot(612)
# a2.plot(np.arange(len(err2)), err2)
# a2.set_title("y-data")
# a5 = plt.subplot(613)
# a5.plot(np.arange(len(err3)), err3)
# a5.set_title("z-data")
# a6 = plt.subplot(614)
# a6.set_title("x-data")
# a6.plot(np.arange(len(err4)), err4)
# a7 = plt.subplot(615)
# a7.plot(np.arange(len(err5)), err5)
# a7.set_title("y-data")
# a8 = plt.subplot(616)
# a8.plot(np.arange(len(err6)), err6)
# a8.set_title("z-data")
# plt.tight_layout()
# plt.show()

# res1 = dict()
# res2 = dict()
# res3 = dict()
# res4 = dict()
# res5 = dict()
# res6 = dict()


# check for the proper mse in the proper interval
# for i, v in enumerate(err1):
#     res1['1'] = v[53 * 4:111 * 4]
#     res1['2'] = v[154 * 4:179 * 4]
#     res1['3'] = v[232 * 4:254 * 4]
#     res1['4'] = v[284 * 4:326 * 4]
#     res1['5'] = v[342 * 4:359 * 4]
#     res1['6'] = v[366 * 4:377 * 4]
#     res1['7'] = v[378 * 4:386 * 4]
#     res1['8'] = v[396 * 4:421 * 4]
#     res1['9'] = v[439 * 4:476 * 4]
#     res1['10'] = v[501 * 4:542 * 4]

# for i, v in enumerate(err2):
#     res2['1'] = v[53 * 4:111 * 4]
#     res2['2'] = v[154 * 4:179 * 4]
#     res2['3'] = v[232 * 4:254 * 4]
#     res2['4'] = v[284 * 4:326 * 4]
#     res2['5'] = v[342 * 4:359 * 4]
#     res2['6'] = v[366 * 4:377 * 4]
#     res2['7'] = v[378 * 4:386 * 4]
#     res2['8'] = v[396 * 4:421 * 4]
#     res2['9'] = v[439 * 4:476 * 4]
#     res2['10'] = v[501 * 4:542 * 4]

# for i, v in enumerate(err3):
#     res3['1'] = v[53 * 4:111 * 4]
#     res3['2'] = v[154 * 4:179 * 4]
#     res3['3'] = v[232 * 4:254 * 4]
#     res3['4'] = v[284 * 4:326 * 4]
#     res3['5'] = v[342 * 4:359 * 4]
#     res3['6'] = v[366 * 4:377 * 4]
#     res3['7'] = v[378 * 4:386 * 4]
#     res3['8'] = v[396 * 4:421 * 4]
#     res3['9'] = v[439 * 4:476 * 4]
#     res3['10'] = v[501 * 4:542 * 4]

# for i, v in enumerate(err4):
#     res4['1'] = v[53 * 4:111 * 4]
#     res4['2'] = v[154 * 4:179 * 4]
#     res4['3'] = v[232 * 4:254 * 4]
#     res4['4'] = v[284 * 4:326 * 4]
#     res4['5'] = v[342 * 4:359 * 4]
#     res4['6'] = v[366 * 4:377 * 4]
#     res4['7'] = v[378 * 4:386 * 4]
#     res4['8'] = v[396 * 4:421 * 4]
#     res4['9'] = v[439 * 4:476 * 4]
#     res4['10'] = v[501 * 4:542 * 4]

# for i, v in enumerate(err5):
#     res5['1'] = v[53 * 4:111 * 4]
#     res5['2'] = v[154 * 4:179 * 4]
#     res5['3'] = v[232 * 4:254 * 4]
#     res5['4'] = v[284 * 4:326 * 4]
#     res5['5'] = v[342 * 4:359 * 4]
#     res5['6'] = v[366 * 4:377 * 4]
#     res5['7'] = v[378 * 4:386 * 4]
#     res5['8'] = v[396 * 4:421 * 4]
#     res5['9'] = v[439 * 4:476 * 4]
#     res5['10'] = v[501 * 4:542 * 4]

# for i, v in enumerate(err6):
#     res6['1'] = v[53 * 4:111 * 4]
#     res6['2'] = v[154 * 4:179 * 4]
#     res6['3'] = v[232 * 4:254 * 4]
#     res6['4'] = v[284 * 4:326 * 4]
#     res6['5'] = v[342 * 4:359 * 4]
#     res6['6'] = v[366 * 4:377 * 4]
#     res6['7'] = v[378 * 4:386 * 4]
#     res6['8'] = v[396 * 4:421 * 4]
#     res6['9'] = v[439 * 4:476 * 4]
#     res6['10'] = v[501 * 4:542 * 4]

# print(res2)

# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      1)
# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      2)
# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      5)
# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      6)
# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      7)
# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      8)
# 'C:\\Users\\Davide\\Documents\\Projects\\sofar_gesture_recognition_tf\\online_data/1115556662.csv'
# Launcher.launch_evaluation()
# Launcher.test()
# test1 = FilesUtil.generate_data_set(os.path.join(CONFIG.G1_PATH))
# test2 = FilesUtil.generate_data_set(os.path.join(CONFIG.G5_PATH))
#
# for list1, list2 in zip(test1, test2):
#     for v1, v2 in zip(list1.values, list2.values):
#         print(v1[0] - v2[0])
#         print(v1[1] - v2[1])
#         print(v1[2] - v2[2])
#         print("----------------------------------------------")
#     print("--------------------GESTURE-----------------------")
# # for j in enumerate(val):
# # print(j)

# print(series[212:444])
# print(v[154 * 4:179 * 4])
# print(v[232 * 4:254 * 4])
# print(v[284 * 4:326 * 4])
# print(v[342 * 4:359 * 4])
# print(v[366 * 4:377 * 4])
# print(v[378 * 4:386 * 4])
# print(v[396 * 4:421 * 4])
# print(v[439 * 4:476 * 4])
# print(v[501 * 4:542 * 4])