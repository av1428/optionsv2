import sys
import os

# Add current directory to sys.path
sys.path.append(os.getcwd())

try:
    print("Attempting to import...")
    from backend.app.strategies.scanner import options_scanner
    print("Import SUCCESS!")
except Exception as e:
    print(f"Import FAILED: {e}")
    import traceback
    traceback.print_exc()

# Check directory structure
print("\nDirectory check:")
print(f"backend exists: {os.path.exists('backend')}")
print(f"backend/__init__.py exists: {os.path.exists('backend/__init__.py')}")
print(f"backend/app exists: {os.path.exists('backend/app')}")
print(f"backend/app/__init__.py exists: {os.path.exists('backend/app/__init__.py')}")
print(f"backend/app/strategies exists: {os.path.exists('backend/app/strategies')}")
print(f"backend/app/strategies/__init__.py exists: {os.path.exists('backend/app/strategies/__init__.py')}")
