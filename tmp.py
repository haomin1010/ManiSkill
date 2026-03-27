import inspect
import mani_skill.envs.tasks.tabletop.stack_cube as m
print(m.__file__)

# 或者用 inspect
from mani_skill.envs.tasks.tabletop import StackCubeEnv
import inspect
print(inspect.getfile(StackCubeEnv))
