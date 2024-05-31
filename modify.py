import sys, subprocess

def transform_script(script_path):
    with open(script_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    preamble = [
        "#!/bin/bash\n",
        "#SBATCH -p gpu --gres=gpu:1\n",
        "module load container_env pytorch-gpu\n",
    ]

    # Check if the script already contains the preamble
    has_preamble = lines[0].startswith("#!")

    # Remove the existing preamble and prefix if present
    if has_preamble:
        for line in lines:
            if line in preamble:
                pass
            elif line.strip().startswith('crun -p ~/envs/NVFlarev2.4.0rc8'):
                modified_lines.append(line.strip()[32:])
            elif line.strip().startswith('command="crun -p ~/envs/NVFlarev2.4.0rc8'):
                modified_lines.append('command="'+line.strip()[43:])
            else:
                modified_lines.append(line)        
    else:
        # Add the preamble
        modified_lines.extend(preamble)    
        # Add prefix to lines starting with 'nvflare'
        for line in lines:
            if line.strip().startswith("nvflare"):
                modified_lines.append(f"crun -p ~/envs/NVFlarev2.4.0rc8 {line.strip()}\n")
            elif line.strip().startswith('command="nvflare'):
                modified_lines.append(f'command="crun -p ~/envs/NVFlarev2.4.0rc8 {line.strip()[9:]}\n')
            else:
                modified_lines.append(line)

   
    # Write the transformed script back to the input file with Unix line endings
    with open(script_path, 'w', newline='\n') as f:
        f.writelines(modified_lines)


    if has_preamble:
        print("Script contained special modifications for HPC, removed the modifications.")
    else:
        print("Script special modifications for HPC added.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python modify.py <input_script_path>")
        sys.exit(1)

    input_script_path = sys.argv[1]

    transform_script(input_script_path)

if __name__ == "__main__":
    main()
