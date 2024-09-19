import numpy as np

def save_modified_obj(original_obj_path, modified_vertices, output_obj_path):
    print('before:')
    print('original_obj_path: {}'.format(original_obj_path))
    print('output_obj_path: {}'.format(output_obj_path))
    with open(original_obj_path, 'r') as original_obj_file:
        lines = original_obj_file.readlines()

    modified_lines = []

    for line in lines:
        tokens = line.strip().split()

        if not tokens:
            modified_lines.append(line)
            continue

        if tokens[0] == 'v':
            # Replace vertex coordinates with modified values
            vertex = modified_vertices.pop(0)
            modified_line = f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    # Write the modified lines to the new OBJ file
    with open(output_obj_path, 'w') as output_obj_file:
        output_obj_file.writelines(modified_lines)
    print('after')
    print('save to {}'.format(output_obj_path))
    print('###')
