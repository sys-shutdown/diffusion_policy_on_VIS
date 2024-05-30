from gym.envs.registration import register
import diffusion_policy.env.vascular

register(
    id='vis-v0',
    entry_point='envs.vascular.vis_env:VISEnv',
)

register(
    id='branches-v0',
    entry_point='envs.vascular.branch_env:branchEnv',
)