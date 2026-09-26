"""Research-harness alias: breath detection now lives in FinalPass itself."""
from finalpass.breath_detect import *  # noqa: F401,F403
from finalpass.breath_detect import Breath, BreathParams, _body, _runs, detect, narration_level  # noqa: F401
