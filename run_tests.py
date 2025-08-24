import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd):
    """Run command and return success status with output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def test_imports():
    """Test core imports."""
    import_tests = [
        "torch",
        "transformers", 
        "datasets",
        "active_learning.models",
        "active_learning.acquisition",
        "active_learning.engine",
        "active_learning.utils"
    ]
    
    failed_imports = []
    for module in import_tests:
        success, stdout, stderr = run_command(f"uv run python -c 'import {module}'")
        if not success:
            print(f"FAIL: {module}")
            if stderr:
                print(f"  Error: {stderr.strip()}")
            failed_imports.append(module)
        else:
            print(f"PASS: {module}")
    
    if failed_imports:
        print(f"\nFailed imports: {', '.join(failed_imports)}")
        return False
    return True


def run_unit_tests():
    """Run pytest test suite."""
    test_files = [
        "src/tests/test_models.py",
        "src/tests/test_acquisition.py", 
        "src/tests/test_data_utils.py",
        "src/tests/test_engine_and_collate.py"
    ]
    
    results = []
    for test_file in test_files:
        cmd = f"uv run pytest {test_file} -v --tb=short"
        success, stdout, stderr = run_command(cmd)
        results.append((test_file, success))
        
        if success:
            print(f"PASS: {test_file}")
        else:
            print(f"FAIL: {test_file}")
            print("--- Test Output ---")
            if stdout:
                print(stdout)
            if stderr:
                print("STDERR:")
                print(stderr)
            print("--- End Output ---\n")
    
    return all(success for _, success in results)


def test_basic_functionality():
    """Test basic framework functionality."""
    tests = [
        # Collate function
        ("collate_function", """
from active_learning.utils.collate import collate_fn
batch = [{'a': 1, 'b': 'hello'}, {'a': 2, 'b': 'world'}]
result = collate_fn(batch)
assert result == {'a': [1, 2], 'b': ['hello', 'world']}
        """),
        
        # Enums
        ("enums", """
from active_learning.models._enums import ApplyTo
assert ApplyTo.TEXT == 'text'
assert ApplyTo.IMAGE == 'image'
assert ApplyTo.BOTH == 'both'
        """),
        
        # Acquisition imports
        ("acquisition_imports", """
from active_learning.acquisition import BALD, Random, CoreSet
        """)
    ]
    
    failed_tests = []
    for test_name, test_code in tests:
        cmd = f"uv run python -c \"{test_code.strip()}\""
        success, stdout, stderr = run_command(cmd)
        
        if success:
            print(f"PASS: {test_name}")
        else:
            print(f"FAIL: {test_name}")
            if stderr:
                print(f"  Error: {stderr.strip()}")
            if stdout:
                print(f"  Output: {stdout.strip()}")
            failed_tests.append(test_name)
    
    if failed_tests:
        print(f"\nFailed basic tests: {', '.join(failed_tests)}")
        return False
    return True


def main():
    """Run test suite."""
    os.chdir(Path(__file__).parent)
    
    print("Testing imports...")
    imports_ok = test_imports()
    
    print("\nTesting basic functionality...")
    basic_ok = test_basic_functionality()
    
    print("\nRunning unit tests...")
    tests_ok = run_unit_tests()
    
    # Summary
    all_passed = imports_ok and basic_ok and tests_ok
    print(f"\nResult: {'PASS' if all_passed else 'FAIL'}")
    
    if not all_passed:
        print("\nTo debug specific failures:")
        print("  uv run pytest tests/test_models.py -v -s")
        print("  uv run pytest tests/ --tb=long")
        print("  uv run python -c 'import active_learning'")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())