import inspect, openai
import os
print(openai.__file__)
compat_path = os.path.join(os.path.dirname(openai.__file__), "_compat.py")
print("compat.py path:", compat_path)

with open(compat_path, "r", encoding="utf-8") as f:
    lines = f.readlines()[:30]
    print("前30行:\n", "".join(lines))
