#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: main.py
@time: 2018/7/25 0025 10:01
@desc: 
"""

import os
import copy
import numpy as np
# import sys 
# sys.path.append('../../')
from agent.fix_rule_no_att.agent import Agent
from interface import Environment
from train.simple import dqn
from train.simple import qmix
import time
MAP_PATH = 'maps/1000_1000_fighter10v10.map'

RENDER = False
MAX_EPOCH = 1000
BATCH_SIZE = 200
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 10         # 不同角度的个数
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1 # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM      # 总的动作数

LEARN_INTERVAL = 100

def replace_rowvec(matrix, ind):
    "替换输入matrix中的第ind行并返回"
    tmp_mat = copy.deepcopy(matrix)
    curr_row = np.zeros_like(tmp_mat[ind, :])
    curr_row[2], curr_row[3] = tmp_mat[ind, 2], tmp_mat[ind, 3]     # x, y 坐标的位置
    
    # 将全局观测的前一半，也就是我方态势的坐标减去相应坐标得到相对坐标
    tmp_num = int(tmp_mat.shape[0]/2)
    tmp_mat = np.concatenate([tmp_mat[0: tmp_num, :] - curr_row, tmp_mat[tmp_num:, :]], axis=0)
    num_column = tmp_mat.shape[1]
    tmp_mat[ind, :] = np.full([num_column], -1.)
    
    return tmp_mat

if __name__ == "__main__":
    # create blue agent
    blue_agent = Agent()
    # get agent obs type
    red_agent_obs_ind = 'simple'
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    red_detector_action = []
    # fighter_model = dqn.RLFighter(ACTION_NUM)
    # 
    fighter_model = qmix.RLFighter(num_act=10, agent_num=10, num_units=20)
    has_initialized = False

    start_time  = None

    # execution
    for x in range(MAX_EPOCH):
        step_cnt = 0
        num_agents = 10
        num_units = 20
        # print('step count', step_cnt)
        env.reset()
        while True:
            obs_list = []
            action_list = []
            red_fighter_action = []
            # get obs
            if step_cnt == 0:
                red_obs_dict, blue_obs_dict = env.get_obs()     # 此处得到 obs
            # get action
            # get blue action
            blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step_cnt)
            # get red action
            obs_got_ind = [False] * red_fighter_num
            # obs_got_ind = [True]

            # 全局态势和每个智能体观测到的态势
            red_obs = np.concatenate([red_obs_dict['fighter'], red_obs_dict['enemy']], axis=0)
            agent_obs = np.zeros([num_agents] + list(red_obs.shape))

            embedding_size = red_obs.shape[1]
            if not has_initialized:
                fighter_model.init(embedding_size)
                has_initialized = True

            # 拼接多个智能体的观测态势得到一个三维张量
            for y in range(num_agents):
                tmp_obs = replace_rowvec(red_obs, y)
                agent_obs[y, :, :] = tmp_obs

            # 20 维的动作空间，10 个方向移动，10 个指定发弹的对象，试试看行不行；有 20 个值，每个值的范围为 [0, 9] 区间的整数
            agent_obs, red_obs = np.expand_dims(agent_obs, axis=0), np.expand_dims(red_obs, axis=0)
            tmp_action = fighter_model.choose_action(agent_obs, red_obs)
            # print('tmp action', tmp_action)
            tmp_action = np.squeeze(tmp_action)
            agent_obs, red_obs = np.squeeze(agent_obs, axis=0), np.squeeze(red_obs, axis=0)

            # tmp_img_obs = red_obs_dict['fighter'][y]['screen']
            # obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
            action_list.append(tmp_action)

            # 此处转换为仿真平台接受的指令
            for y in range(red_fighter_num):
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)

                if red_obs_dict['alive'][y]:
                    obs_got_ind[y] = True
                    if tmp_action[y] < 10:
                        true_action[0] = int(360 / COURSE_NUM * int(tmp_action[y]))
                    else:
                        true_action[3] = int(tmp_action[y]) - 10

                red_fighter_action.append(true_action)

            # step and get reward
            red_fighter_action = np.array(red_fighter_action)
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

            # tmp_fighter_action = np.array([np.array([0, 1, 0, 0], dtype=np.int32)] * 10)
            # env.step([], tmp_fighter_action, blue_detector_action, blue_fighter_action)
            
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
            detector_reward = red_detector_reward + red_game_reward
            fighter_reward = red_fighter_reward + red_game_reward
            
            # save replay
            red_obs_dict, blue_obs_dict = env.get_obs()
            next_red_obs = np.concatenate([red_obs_dict['fighter'], red_obs_dict['enemy']], axis=0)
            next_agent_obs = np.zeros([num_agents] + list(next_red_obs.shape))

            # 组装下一步的态势 s, global_s, a, r, s_, global_next_s
            for y in range(num_agents):
                tmp_obs = replace_rowvec(next_red_obs, y)
                next_agent_obs[y, :, :] = tmp_obs

            fighter_model.store_transition(agent_obs, red_obs, action_list, sum(fighter_reward), next_agent_obs, next_red_obs)
            
            # if done, perform a learn
            if env.get_done():
                # detector_model.learn()
                # fighter_model.learn()
                break
            # if not done learn when learn interval
            if (step_cnt > 0) and (step_cnt % LEARN_INTERVAL == 0):
                now_time = time.time()
                if start_time is not None:
                    duration_time = now_time - start_time
                    # print('time cost per 100 steps: ', duration_time)

                start_time = now_time
                print('step_cnt: ', step_cnt)
                # detector_model.learn()
                fighter_model.learn()
            step_cnt += 1

