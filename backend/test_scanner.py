import sys
import os
import json

# Add current directory to path so 'app' imports work
sys.path.append(os.getcwd())

from app.strategies.scanner import options_scanner

print("Running Scanner...")
try:
    results = options_scanner.scan_opportunities()
    print(json.dumps(results, indent=2, default=str))
except Exception as e:
    print(f"Error: {e}")
