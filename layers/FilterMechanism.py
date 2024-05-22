# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : Solarformer 
@File    : filtering.py
@IDE     : PyCharm 
@Author  : Lifeng
@Date    : 2024/1/11 9:37 
@Software: PyCharm
'''
import numpy as np
import pandas as pd
import torch
import datetime

from torch import nn

# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : Solarformer 
@File    : filtering.py
@IDE     : PyCharm 
@Author  : Lifeng
@Date    : 2024/1/11 9:37 
@Software: PyCharm
'''
import torch
import datetime


class FilterMechanism(torch.nn.Module):
    def __init__(self, sunrise_time="06:00:00", sunset_time="18:00:00"):
        super().__init__()
        self.sunrise_time = datetime.datetime.strptime(sunrise_time, "%H:%M:%S").time()
        self.sunset_time = datetime.datetime.strptime(sunset_time, "%H:%M:%S").time()

        self.sunrise = self.sunrise_time.hour * 3600 + self.sunrise_time.minute * 60 + self.sunrise_time.second
        self.sunset = self.sunset_time.hour * 3600 + self.sunset_time.minute * 60 + self.sunset_time.second

    def forward(self, timestamp_tensor, feature_tensor):
        # 将时间戳转换为 Unix 时间戳

        unix_timestamp = timestamp_tensor

        # 计算每个时间戳对应的时间
        time_in_seconds = torch.tensor(
            ((unix_timestamp - unix_timestamp // 86400 * 86400) // 3600) * 3600,
            device=timestamp_tensor.device)
        # 判断是否处于夜间时刻
        is_night = torch.logical_or(time_in_seconds < self.sunrise,
                                    time_in_seconds > self.sunset)

        is_day = torch.logical_not(is_night).cuda()

        feature_tensor = torch.where(is_day.unsqueeze(-1), feature_tensor.cuda(),
                                     torch.zeros_like(feature_tensor).cuda())
        return feature_tensor.cuda()

# class FilterMechanism(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 设置日出日落时间为可学习参数
#         self.sunrise_time = 0
#         self.sunset_time = 0
#
#     def get_season_sunrise_sunset(self, date):
#
#         # if date.month in [12, 1, 2]:  # 夏季
#         #     sunrise_time = datetime.time(hour=5, minute=30, second=0)
#         #     sunset_time = datetime.time(hour=19, minute=30, second=0)
#         # elif date.month in [9, 11]:  # 春季
#         #     sunrise_time = datetime.time(hour=6, minute=0, second=0)
#         #     sunset_time = datetime.time(hour=18, minute=0, second=0)
#         # elif date.month in [3, 4, 5]:  # 秋季
#         #     sunrise_time = datetime.time(hour=6, minute=0, second=0)
#         #     sunset_time = datetime.time(hour=18, minute=0, second=0)
#         # else:  # 冬季
#         #     sunrise_time = datetime.time(hour=7, minute=30, second=0)
#         #     sunset_time = datetime.time(hour=17, minute=30, second=0)
#         # return sunrise_time, sunset_time
#
#         if date.month in [12, 1, 2]:  # 夏季
#             sunrise_time = 6
#             sunset_time = 19
#         elif date.month in [9, 10, 11]:  # 春季
#             sunrise_time = 6
#             sunset_time = 18
#         elif date.month in [3, 4, 5]:  # 秋季
#             sunrise_time = 6
#             sunset_time = 18
#         else:  # 冬季
#             sunrise_time = 7
#             sunset_time = 17
#         return sunrise_time, sunset_time
#
#     def update_sunrise_sunset(self, date):
#
#         # 根据日期获取对应季节的日出时间和日落时间
#         self.sunrise_time, self.sunset_time = self.get_season_sunrise_sunset(date)
#
#     def forward(self, unix_timestamp, feature_tensor):
#
#         # 将PyTorch张量转换为NumPy数组
#         numpy_data = unix_timestamp.numpy()
#
#         # 转换为日期时间数据
#         datetime_data = np.array([pd.to_datetime(date, unit='s') for date in numpy_data.flatten()])
#         datetime_data = datetime_data.reshape(numpy_data.shape)
#         # 将二维NumPy数组转换为Pandas DataFrame
#
#         data = pd.DataFrame(datetime_data)
#
#         # 逐行读取数据
#         for row_index, row in data.iterrows():
#             # row[0].month
#             # 遍历该行的每一列
#             for col_index, value in row.iteritems():
#                 # 将特定位置置为0
#                 self.update_sunrise_sunset(value)
#                 if value.hour > self.sunset_time & value.hour < self.sunrise_time:
#                     feature_tensor[row_index, col_index, :] = 0
#
#         return feature_tensor
