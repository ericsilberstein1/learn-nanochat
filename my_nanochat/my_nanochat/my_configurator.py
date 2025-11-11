import os
import sys
from ast import literal_eval

def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

for arg in sys.argv[1:]:
    if '=' not in arg:
        assert False, "TODO implement config file support"
    else:
        # assume it's a --key=value argument
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            if globals()[key] is not None:
                attempt_type = type(attempt)
                default_type = type(globals()[key])
                assert attempt_type == default_type, f"type mismatch: {attempt_type} != {default_type}"
            print0(f"overriding {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"unknown config key: {key}")
