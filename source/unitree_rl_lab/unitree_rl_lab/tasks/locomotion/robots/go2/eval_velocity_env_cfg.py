# import math

# import isaaclab.sim as sim_utils
# import isaaclab.terrains as terrain_gen
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# from isaaclab.envs import ManagerBasedRLEnvCfg
# from isaaclab.managers import CurriculumTermCfg as CurrTerm
# from isaaclab.managers import EventTermCfg as EventTerm
# from isaaclab.managers import ObservationGroupCfg as ObsGroup
# from isaaclab.managers import ObservationTermCfg as ObsTerm
# from isaaclab.managers import RewardTermCfg as RewTerm
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.managers import TerminationTermCfg as DoneTerm
# from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
# from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.utils import configclass
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
# from unitree_rl_lab.tasks.locomotion import mdp

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg

@configclass 
class RobotEvalEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        '''
        DONE:

        disable push

        disable curriculum commands (terrain, command)
            but have to enable curriculum for terrain generation (because of how coded)

        set max_init_terrain_level = None

        resampling_time_range large to each env is strictly single command

        make episode length very long, reset manually

        TODO:

        

        make episode length less than total env time; maybe not since can just graph data and look at it really hard

        '''

        self.events.push_robot = None

        self.curriculum = None
        self.scene.terrain.terrain_generator.curriculum = True # enable terrain curriculum

        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.resampling_time_range = (1e9, 1e9)

        
        self.episode_length_s = 1000.0