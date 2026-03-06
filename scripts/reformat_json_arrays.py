#!/usr/bin/env python3
"""
Script to reformat JSON config files so that arrays are on the same line.
This makes the JSON more compact while preserving readability.
"""
import json
import os
import glob
import argparse


def compact_json_arrays(obj, indent=4):
    """
    Recursively process JSON object to make arrays compact (single line).
    """
    if isinstance(obj, dict):
        return {key: compact_json_arrays(value, indent) for key, value in obj.items()}
    elif isinstance(obj, list):
        # If it's a list/array, return it as-is (will be formatted compactly by json.dumps)
        return obj
    else:
        return obj


def reformat_json_file(filepath, max_line_length=120):
    """
    Reformat a single JSON file to have compact arrays and objects.
    """
    try:
        # Read the JSON file
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Custom JSON encoder to make arrays and simple objects compact
        class CompactEncoder(json.JSONEncoder):
            def __init__(self, max_line_length=130, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.max_line_length = max_line_length
                self.indentation_level = 0

            def encode(self, o):
                if isinstance(o, (list, tuple)):
                    if self._is_simple_array(o):
                        return '[' + ', '.join(json.dumps(item, separators=(',', ':')) for item in o) + ']'
                    else:
                        self.indentation_level += 1
                        output = [self.indent_str + self.encode(el) for el in o]
                        self.indentation_level -= 1
                        return '[\n' + ',\n'.join(output) + '\n' + self.indent_str + ']'
                elif isinstance(o, dict):
                    if self._is_simple_object(o) and self._would_fit_on_line(o):
                        items = [f"{json.dumps(k, separators=(',', ':'))}: {self.encode(v)}" for k, v in o.items()]
                        return '{' + ', '.join(items) + '}'
                    else:
                        if not o:
                            return '{}'
                        else:
                            self.indentation_level += 1
                            items = []
                            for key, value in o.items():
                                items.append(self.indent_str + json.dumps(key, separators=(',', ':')) + ': ' + self.encode(value))
                            self.indentation_level -= 1
                            return '{\n' + ',\n'.join(items) + '\n' + self.indent_str + '}'
                else:
                    return json.dumps(o, separators=(',', ':'))

            @property
            def indent_str(self):
                return '    ' * self.indentation_level

            def _is_simple_array(self, arr):
                """Check if array contains only simple values (no nested objects/arrays)"""
                for item in arr:
                    if isinstance(item, (dict, list)):
                        return False
                return True

            def _is_simple_object(self, obj):
                """Check if object contains only simple values (no nested objects, but arrays are OK)"""
                for value in obj.values():
                    if isinstance(value, dict):
                        return False
                    # Allow arrays as long as they are simple arrays
                    elif isinstance(value, (list, tuple)):
                        if not self._is_simple_array(value):
                            return False
                return True

            def _would_fit_on_line(self, obj):
                """Check if the compact version of this object would fit within max_line_length"""
                items = [f"{json.dumps(k, separators=(',', ':'))}: {self.encode(v)}" for k, v in obj.items()]
                compact_version = '{' + ', '.join(items) + '}'
                return len(compact_version) <= self.max_line_length

        # Write back with compact array and object formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(CompactEncoder(max_line_length=max_line_length).encode(data))

        print(f"Reformatted: {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Reformat JSON files to have compact arrays and objects')
    parser.add_argument('-d', '--directory', default='./config',
                       help='Directory to search for JSON files (default: ./config)')
    parser.add_argument('-p', '--pattern', default='**/*.json',
                       help='Glob pattern for JSON files (default: **/*.json)')
    parser.add_argument('--max-line-length', type=int, default=120,
                       help='Maximum line length before keeping multi-line format (default: 120)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show which files would be processed without modifying them')

    args = parser.parse_args()

    # Find all JSON files
    pattern = os.path.join(args.directory, args.pattern)
    json_files = glob.glob(pattern, recursive=True)

    if not json_files:
        print(f"No JSON files found in {args.directory} with pattern {args.pattern}")
        return

    print(f"Found {len(json_files)} JSON files to process:")
    for filepath in json_files:
        print(f"  {filepath}")

    if args.dry_run:
        print("\nDry run - no files modified.")
        return

    print("\nProcessing files...")
    for filepath in json_files:
        reformat_json_file(filepath, args.max_line_length)

    print(f"\nCompleted! Processed {len(json_files)} files.")


if __name__ == '__main__':
    main()