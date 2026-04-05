import json
import sys

from strict_policy_agent import get_action


if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read())
    result = get_action(input_data)
    print(json.dumps(result))
