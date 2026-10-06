"""Convert PLLM generation CSVs to the historical PLL decoder's input format."""
import argparse
import csv
from pathlib import Path


def convert(input_path: str, output_path: str) -> int:
    output = Path(output_path)
    if Path(input_path).resolve() == output.resolve():
        raise ValueError('Input and output must be different files.')
    rows = []
    with open(input_path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'pll_sequence' not in reader.fieldnames:
            raise ValueError('Expected a pll_sequence column from PLLM generation.')
        for row_number, row in enumerate(reader, start=2):
            indices = [int(token) for token in row['pll_sequence'].split()]
            if not indices or any(index < 0 or index >= 4096 for index in indices):
                raise ValueError(f'Row {row_number} must contain nonempty PLL content codes in [0, 4095].')
            rows.append({'sequence': '', 'sequence_length': len(indices),
                         'indices': ' '.join(map(str, indices))})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sequence', 'sequence_length', 'indices'])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='PLLM generated_sequences.csv')
    parser.add_argument('--output', required=True, help='PLL decoder input CSV')
    args = parser.parse_args()
    print(f'Converted {convert(args.input, args.output)} PLL samples.')
