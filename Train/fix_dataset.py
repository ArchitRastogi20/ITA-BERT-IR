import json

def validate_and_clean_jsonl(input_file, output_file, error_file):
    valid_lines = 0
    invalid_lines = 0
    total_lines = 0
    required_keys = ['query', 'positive', 'hard_negative']
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         open(error_file, 'w', encoding='utf-8') as errfile:
        for line_number, line in enumerate(infile, 1):
            total_lines += 1
            try:
                line_clean = line.strip()
                json_obj = json.loads(line_clean)
                # Normalize keys: strip whitespace and convert to lowercase
                json_obj_normalized = {key.strip().lower(): value for key, value in json_obj.items()}
                # Check for required keys
                all_keys_present = all(key in json_obj_normalized for key in required_keys)
                # Check if values are strings and not empty
                all_values_are_strings = all(isinstance(json_obj_normalized[key], str) for key in required_keys if key in json_obj_normalized)
                all_values_non_empty = all(json_obj_normalized[key].strip() != '' for key in required_keys if key in json_obj_normalized)
                if all_keys_present and all_values_are_strings and all_values_non_empty:
                    # Reconstruct the JSON object with normalized keys
                    valid_json_obj = {key: json_obj_normalized[key] for key in required_keys}
                    outfile.write(json.dumps(valid_json_obj, ensure_ascii=False) + '\n')
                    valid_lines += 1
                else:
                    print(f"Missing or invalid keys in line {line_number}")
                    print(f"all_keys_present: {all_keys_present}")
                    print(f"all_values_are_strings: {all_values_are_strings}")
                    print(f"all_values_non_empty: {all_values_non_empty}")
                    errfile.write(f"Line {line_number}: Missing or invalid keys\n")
                    invalid_lines += 1
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in line {line_number}: {e}")
                errfile.write(f"Line {line_number}: {e}\n")
                invalid_lines += 1
    print(f"Total lines: {total_lines}")
    print(f"Valid lines: {valid_lines}")
    print(f"Invalid lines: {invalid_lines}")


def main():
    validate_and_clean_jsonl("train.jsonl", "train_fixed.jsonl", "fixed_train_error.log")

if __name__ == "__main__":
    main()