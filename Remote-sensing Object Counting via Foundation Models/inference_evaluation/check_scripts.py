import sys
import os

print("\n" + "="*40)
print("🔍 finalextremeenvironmentarrangecheckstart")
print("="*40)

# 1. looklooktobottomiswhich Python inrun
print(f"👉 1. currentrealjustrun  Python path:\n   {sys.executable}\n")

# 2. looklookthis Python gowhichfind inthirdsidelibrary
print("👉 2. current Python checkfindlibrary pathlist (sys.path):")
for p in sys.path:
    print(f"   - {p}")
print("\n")

# 3. trydirectimport transformers looklookittobottomsaywhat
try:
    import transformers
    print(f"✅ 3. successfindto transformers!\n   it installpositionis: {transformers.__file__}")
except Exception as e:
    print(f"❌ 3. accordingthenfindnotto transformers! real reportwrongreasonis:\n   {repr(e)}")

print("="*40)
print("arrangecheckend, forcestopstopprogramrun...\n")
sys.exit(0) # letprogramstopinhere, notneedtowardbelowrunoriginalcome reportwrongcode 

