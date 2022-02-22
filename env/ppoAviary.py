from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
from gym import spaces
import numpy as np
import  pybullet as p
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from ray.rllib.env.env_context import EnvContext
import  os
class ppoAviary(BaseMultiagentAviary):

    def __init__(self,

                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 3,
                 neighbourhood_radius: float = 0.25,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 gui=True,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.PID):



        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.fly_reward = {}
        self.target_reward = {}
        self.obstacle_reward = {}
        self.neighbor_reward = {}
        self.boundry_reward={}
        self.n = self.NUM_DRONES
        self.boundry = np.array([6,6,2])
        self.target = np.array([2,1])
        self.agent_R_done=np.zeros(self.NUM_DRONES,dtype=bool)

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.KIN:
            # lowarr = np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1])
            # higharr = np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1,1,1,1])
            lowarr = np.array([-1, -1, 0])
            higharr = np.array([1, 1,  1])
            return spaces.Dict({i: spaces.Box(low=lowarr,
                                              high=higharr,
                                              dtype=np.float32
                                              ) for i in range(self.NUM_DRONES)})
        else:
            print("[ERROR] in BaseMultiagentAviary._observationSpace()")

    # def _actionSpace(self):
    #     a = []
    #     if self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
    #         size = 4
    #     elif self.ACT_TYPE == ActionType.PID:
    #         size = 3
    #     elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
    #         size = 1
    #     else:
    #         print("[ERROR] in BaseMultiagentAviary._actionSpace()")
    #         exit()
    #     a.append(spaces.Box(low=-1*np.ones(size), high=np.ones(size),dtype=np.float32))
    #     return a


    def _computeObs(self):
        # obs_18={}
        obs_3 = {}
        #neighborIndex = self._getNeighborList()
        for i in range(self.NUM_DRONES):
            obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
            # neighbor_index0 = neighborIndex[i][0]
            # neighbor_index1 = neighborIndex[i][1]
            # neighbor_0 = self._clipAndNormalizeState(self._getDroneStateVector(neighbor_index0))
            # neighbor_0 = neighbor_0[0:3]
            # neighbor_1 = self._clipAndNormalizeState(self._getDroneStateVector(neighbor_index1))
            # neighbor_1 = neighbor_1[0:3]
            #
            # neighbor_0 = self.pos[neighbor_index1,:]
            # neighbor_1 = self.pos[neighbor_index2,:]
            #obs_18[i] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], neighbor_0,neighbor_1]).reshape(18,)
            obs_3[i] = obs[0:3]
        return obs_3

    def _detectCollision(self):
        collision_list=[]

        for i in range(self.NUM_DRONES):
            p_min, p_max = p.getAABB(self.DRONE_IDS[i])
            collision_tuple = p.getOverlappingObjects(p_min, p_max)
            l = len(collision_tuple)
            if l>6:
                collision_list.append(True)
            else:
                collision_list.append(False)
        return collision_list
    def _getNeighborList(self):
        neighbor_info = np.zeros((self.NUM_DRONES,self.NUM_DRONES))
        for i in range(self.NUM_DRONES - 1):
            for j in range(self.NUM_DRONES - i - 1):
                neighbor_info[i, j + i + 1] = neighbor_info[j + i + 1, i] = np.linalg.norm(
                    self.pos[i, :] - self.pos[j + i + 1, :])
        neighbor_index = {}
        for i in range(self.NUM_DRONES):
            agent_i = neighbor_info[i]
            sort_agent_i = np.argsort(agent_i)
            neighbor_index[i] = sort_agent_i[1:3]

        return neighbor_index


    def _computeReward(self):
        rewards = {}
        neighborIndex = self._getNeighborList()

       # collision_occur = self._detectCollision()
        # for i in range(self.NUM_DRONES):
        #     if collision_occur[i]==True:
        #         self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i],
        #                                                                     physicsClientId=self.CLIENT)
        #         quat = p.getQuaternionFromEuler([0,0,0])
        #         p.resetBasePositionAndOrientation(self.DRONE_IDS[i],self.pos[i],self.quat[i])
        #print(f"step--------------{self.step_counter}")
        for i in range(self.NUM_DRONES):
            fly_reward = self._flyReward(i)
            target_reward = self._targetReward(i)
            boundry_reward = self._boundryReward(i)
            neighbor_reward = self._neighborReward(i)
            obstacle_reward = self._obstacleReward(i)
            self.fly_reward[i]=fly_reward
            self.target_reward[i]=target_reward
            self.boundry_reward[i]=boundry_reward
            self.neighbor_reward[i]=neighbor_reward
            self.obstacle_reward[i]=obstacle_reward
            self.target_reward[i] = target_reward

            #rewards[i] =  -1000*fly_reward-10*target_reward-1*height_reward-0*neighbor_reward
            if self.agent_R_done[i]==True:
                rewards[i]=0
            else:
                rewards[i] = target_reward+boundry_reward-1
        return rewards

    def _boundryReward(self,nth):
        pos = self.pos[nth, :]
        if   (abs(pos[0])>self.boundry[0] or abs(pos[1])>self.boundry[1] or abs(pos[2])>self.boundry[2]):
            return -100
        else:
            return 0


    def _flyReward(self,nth):
        state = self._getDroneStateVector(nth)
        return np.linalg.norm(state[9])+0.15*np.linalg.norm(state[7:9])+0.25*np.linalg.norm(state[13:16])
    def _targetReward(self,nth):
       # vel_vec = state[10:12]
        #target_vec = self.target[0:2]-state[0:2]
        distance = np.linalg.norm(self.pos[nth, :][0:2]-self.target)
        distance_reward = max(0,distance-0.3)
   #     l = np.linalg.norm(state[0:2]-self.target)
   #      if np.linalg.norm(vel_vec)==0:
   #          return 0
       # cos_pi = vel_vec.dot(target_vec)/(np.linalg.norm(vel_vec)*np.linalg.norm(target_vec))
        #cos_hudu =np.arccos(cos_pi)*180/np.pi
        #reward = 1/(cos_hudu+1) if cos_hudu<65 else -0.5*cos_hudu
        return -distance_reward#-0.01*cos_hudu

    def _neighborReward(self,nth):
        state = self._getDroneStateVector(nth)
        agent_pos = state[0:3]
        neighbor_list = self._getNeighborList()
        neighbor_0 = neighbor_list[nth][0]
        neighbor_1 = neighbor_list[nth][1]
        neighbor_0_pos = self.pos[neighbor_0,:]
        neighbor_1_pos = self.pos[neighbor_1,:]
        N_0 = 0 if np.linalg.norm(agent_pos-neighbor_0_pos)>2*self.L else 1/(np.linalg.norm(agent_pos-neighbor_0_pos))
        N_1 = 0 if np.linalg.norm(agent_pos-neighbor_1_pos)>2*self.L else 1/(np.linalg.norm(agent_pos-neighbor_1_pos))
        return N_0**2+N_1**2

    def _obstacleReward(self,nth):
        state = self._getDroneStateVector(nth)
        obstacle_distance_min = self._obstacleDistance(nth,state)
        obstacle_distance = obstacle_distance_min-0.5-self.L
        return 1/obstacle_distance


    def _obstacleDistance(self,nth,state):
        obstacle_agent=[]
        for i in range(1,9):
            obstacle_pos = np.array([(i // 3) * 2 , (i % 3) * 2 ])
            obstacle_agent.append(np.linalg.norm(obstacle_pos-state[0:2]))
        return min(obstacle_agent)

    def _computeInfo(self):
        return {i: {} for i in range(self.NUM_DRONES)}

    def _computeDone(self):
                #p.resetBasePositionAndOrientation(self.DRONE_IDS[i], [0,0,0.5*i+0.1], quat)
        # for i in range(self.NUM_DRONES):
        #     if self.pos[i][2]>1:
        #         qua = p.getQuaternionFromEuler([0,0,0])
        #         p.resetBasePositionAndOrientation(0,[0,0,.5],qua)
        bool_val = True if self.step_counter >6000 else False

        done = {i:bool_val for i in range(self.NUM_DRONES)}
        # for i in range(self.NUM_DRONES):
        #     pos=self.pos[i, :]
        #     target_distance = np.linalg.norm(pos[0:2]-self.target)
        #     done_i = (target_distance<0.3)
        #     self.agent_R_done[i] =done_i
        #     # done_i = (target_distance==0)
        #     done[i]=done_i

        done["__all__"] = bool_val  # True if True in done.values() else False
        #self._freeze(done)

        return done

    def _freeze(self,done):
        for i in range(self.NUM_DRONES):
            quat = p.getQuaternionFromEuler([0, 0, 0])
            if done[i]==True:
                pos = self.pos[i]
                p.resetBasePositionAndOrientation(self.DRONE_IDS[i], pos, quat)

    def _addObstacles(self):
        pass
    '''
        radius = .5
        height = 2
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[1, .3, .94, 1]
        )

        collison_cylinder_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=height
        )
        for i in range(1,9):
            wall_id_0 = p.createMultiBody(
                baseMass=100000,
                baseCollisionShapeIndex=collison_cylinder_id,
                baseVisualShapeIndex=visual_shape_id,

                basePosition=[(i // 3) * 2 , (i % 3) * 2 , height / 2]
            )
    '''

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        #MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        # MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC
        MAX_XY = 6
        MAX_Z =2
        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        # if self.GUI:
        #     self._clipAndNormalizeStateWarning(state,
        #                                        clipped_pos_xy,
        #                                        clipped_pos_z,
        #                                        clipped_rp,
        #                                        clipped_vel_xy,
        #                                        clipped_vel_z
        #                                        )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20, )

        return norm_and_clipped

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter,
                  "in xyzMulti._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0],
                                                                                                        state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in xyzMulti._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in xyzMulti._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7],
                                                                                                       state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in xyzMulti._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10],
                                                                                                        state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in xyzMulti._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))


'''
            neighbor_index1 = neighborIndex[i][0]
            neighbor_index2 = neighborIndex[i][1]
            neighbor_0 = self.pos[neighbor_index1, :]
            neighbor_1 = self.pos[neighbor_index2, :]
            clip_neighbor_0 = np.linalg.norm(states[i][0:3] - neighbor_0)-self.NEIGHBOURHOOD_RADIUS
            clip_neighbor_1 = np.linalg.norm(states[i][0:3] - neighbor_1)-self.NEIGHBOURHOOD_RADIUS
            clip_neighbor_0 = clip_neighbor_0 if clip_neighbor_0<0 else 0
            clip_neighbor_1 = clip_neighbor_1 if clip_neighbor_1<0 else 0
            neighbor_error  = 1/np.exp(clip_neighbor_0)+1/np.exp(clip_neighbor_1)-2
            self.neighbor_error[i] = neighbor_error

            fly_error = np.linalg.norm(states[i][9])+0.15*np.linalg.norm(states[i][7:9])+0.25*np.linalg.norm(states[i][13:16])#-0.707*np.linalg.norm(states[i][16:20])
            self.fly_error[i] = fly_error

            height_exceed = states[i][2]-2 if states[i][2]-2>=0 else 0
            height_error = np.exp(height_exceed)-1
            self.height_error[i]=height_error
            target_error = np.linalg.norm(self.pos[i][0:2]-self.target)**2#让飞机悬停在target处，高度有区别
            # target_error_x = np.clip(self.pos[i][0]-target_x,-np.inf,0)
            # target_error_y = np.clip(self.pos[i][1]-target_y,-np.inf,0)
            #target_error = np.linalg.norm([target_error_x,target_error_y])
            #target_error = 1/np.exp(target_error_x) +1/np.exp(target_error_y) -2
            self.target_error[i] = target_error

            pos = self.pos[i][0:2]
            obstacle_pos = {}

            for pp in range(1,9):
                obstacle_pos[pp] =np.array([(pp // 3) * 2 , (pp % 3) * 2 ])
            obstacle_distance = np.array([np.linalg.norm(obstacle_pos[i]-pos) for i in range(1,9)])#find out the closest obstacle to agent_i
            #obstacle_error = obstacle_distance.min()-0.5-self.L
            # obstacle_error = (1/self.L)*(obstacle_distance.min()-0.5-self.L) if obstacle_distance.min()<(0.5+2*self.L) else 1
            # obstacle_error = np.log(obstacle_error)
            obstacle_min = obstacle_distance.min()
            obstacle_min_minus_R = obstacle_distance.min()-0.5-self.L
            obstacle_min_minus_R = obstacle_min_minus_R if obstacle_min_minus_R<2*self.L else 0
            obstacle_error = np.exp(obstacle_min_minus_R)-1
            self.obstacle_error[i] = obstacle_error
            # if (obstacle_min <=0.5+2*self.L):
            #     self.obstacle_done[i] = True
            # else:
            #     self.obstacle_done[i]=False
            '''