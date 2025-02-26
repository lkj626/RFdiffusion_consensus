import numpy as np
from pathlib import Path
import gradio as gr
import pandas as pd
import os
import glob
import datetime
import requests
import shutil
import re
from gradio_molecule3d import Molecule3D
import subprocess
from Bio import PDB, SeqIO
from consensus import ConsensusSequenceGenerator
from ColabDesign.colabdesign.af.model import mk_af_model

env = os.environ.copy()
env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_free_gpu():
    """Find free memory gpu"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )

        gpus = result.stdout.strip().split("\n")
        gpus = [line.split(",") for line in gpus]
        gpus = [(gpu[0], int(gpu[1])) for gpu in gpus]  # (GPU 번호, 메모리)
        print(gpus)
        
        best_gpu = sorted(gpus, key=lambda x: x[1], reverse=True)[0][0]

        return best_gpu
    
    except Exception as e:
        print(f"GPU 확인 중 오류 발생: {e}")
        return None

free_gpu = get_free_gpu()
if free_gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu
    print(f"사용할 GPU: {free_gpu}")
else:
    print("사용 가능한 GPU가 없습니다.")


reps =  [
    {
      "model": 0,
      "chain": "",
      "resname": "",
      "style": "cartoon",
      "color": "greenCarbon",
      "residue_range": "",
    #   "around": 0,
    #   "byres": False,
    }
]

# Upload Fasta File case
def get_pdb_file(pdb_id):
    """
    Download pdb file using rcsb api with input PDB ID.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "get_pdb")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)

    if response.status_code == 200:
        file_path = f"{save_path}/{pdb_id}.pdb"
        with open(file_path, "w") as f:
            f.write(response.text)
        print(f"PDB file {pdb_id} downloaded successfully to {file_path}")
    else:
        print(f"Failed to download PDB file {pdb_id}. Check if the PDB ID is correct.")

    return file_path

def read_pdb(file):
    """Extract sequence by reading pdb file"""
    results = []
    pdb_id = []
    uniprot_id = []
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(file.name, file.name)

    # 서열 추출
    ppb = PDB.PPBuilder()
    sequences = [pp.get_sequence() for pp in ppb.build_peptides(structure)]
        
    # 파일 이름과 함께 출력할 서열 생성
    if sequences:
        seq_text = "".join(map(str, sequences))
    else:
        seq_text = "No sequence found in this PDB file."
        
    results.append(f"{seq_text}")
    
    return results, pdb_id, uniprot_id

def read_fasta(file):
    """
    Extract sequence by reading fasta file and extract uniprot Id or Pdbid if present in file
    """
    sequences = []
    pdb_id = []
    uniprot_id = []
    entry_cnt = 0

    with open(file.name, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            sequences.append(f"{record.seq}")
            entry_cnt += 1

            # fasta file has one sequence information
            if entry_cnt < 2:
                # PDB ID 패턴 (예: ">4HHB:A" -> "4HHB", "4HBB_1" -> "4HHB")
                pdb_match = re.search(r"^(\d[A-Z0-9]{3}(?:[:_][A-Za-z0-9])?)", record.description)

                if pdb_match:
                    pdb_id_data = pdb_match.group(1).upper()
                    pdb_id_data = re.sub(r"[:_].*$", "", pdb_id_data)  # `:문자` 또는 `_숫자` 제거
                    pdb_id.append(pdb_id_data)

                # UniProt ID 패턴 (예: ">sp|P12345|Protein_Name")
                uniprot_match = re.search(r">\S*\|([A-Z0-9]+)\|", record.description)

                if uniprot_match:
                    uniprot_id.append(uniprot_match.group(1).upper())    
              
    if sequences:
        return sequences, pdb_id, uniprot_id

    else: 
        return ["No sequence found in this FASTA file."], pdb_id, uniprot_id

def diffusion_score(pdb_path, output_path, consen_seq_len):
    """
    Make a sequence through proteinMPNN with the pdb file you received,
    create a pdb again through alphafold, and calculate the mpnn, plddt, ptm, pae, rmsd score
    """    
    opts = [
        f"--pdb={pdb_path}",
        f"--loc={output_path}",
        f"--contig={str(consen_seq_len)}-{str(consen_seq_len)}"
        # f"--copies={copies}",
        # f"--num_seqs={num_seqs}",
        # f"--num_recycles={num_recycles}",
        # f"--rm_aa={rm_aa}",
        # f"--mpnn_sampling_temp={mpnn_sampling_temp}",
        # f"--num_designs={num_designs}"
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = [
        "bash", "-c",
        f"source {script_dir}/anaconda3/etc/profile.d/conda.sh && conda activate {script_dir}/anaconda3/envs/SE3nv && python "
        f"{script_dir}/ColabDesign/designability_test.py " + " ".join(opts)
    ]

    try:
        result = subprocess.run(cmd,text=True, check=True, env=env)

        # RFdiffusion score output pdb file or path return 
        pdb_file = f"{output_path}/best.pdb"
        with open(pdb_file, "r") as f:
            first_line = f.readline().strip()

        # Find 'N'
        tokens = first_line.split()
        n_value = None
        for i, token in enumerate(tokens):
            if token == "N":
                n_value = int(tokens[i + 1])  # Get the number next to 'N'

                break

        if n_value is None:
            raise ValueError("PDB 파일에서 'N' 값을 찾을 수 없습니다.")

        print(n_value)
        # Get rows with n columns matching that value from CSV file
        score_data = pd.read_csv(f"{output_path}/mpnn_results.csv", index_col=0)

        matching_rows = score_data[score_data['n'] == n_value]
        print(matching_rows)

        return matching_rows

    except subprocess.CalledProcessError as e:
        return f"오류 발생: {e}" 


def run_rfdiffusion(input, output_path, provide_seq, consen_seq_len, num_designs=1):
    """
    After receiving the Pdb file, protein length, and parts to be fixed, run RFdiffusion
    """
    # make directory RFdiffusion output_folder
    print(output_path)
    output_path = f"{output_path}/diffusion_pdb"

    os.mkdir(output_path)

    consen_seq_len = str(consen_seq_len)
    provide_seq = str(provide_seq)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "RFdiffusion/scripts/run_inference.py")

    # run RFdiffusion command

    cmd = [
        "bash", "-c",
        f"source {script_dir}/anaconda3/etc/profile.d/conda.sh && "
        f"conda activate {script_dir}/anaconda3/envs/SE3nv && "
        f"python {script_path} "
        f"'contigmap.contigs=[{consen_seq_len}-{consen_seq_len}]' "
        f"'contigmap.provide_seq={provide_seq}' "
        f"diffuser.partial_T=30 "
        f"inference.input_pdb={input} "
        f"inference.output_prefix={output_path} "
        f"inference.num_designs={num_designs}"
    ]

    try:
        result = subprocess.run(cmd,text=True, check=True)
        # RFdiffusion output pdb file or path return 
        return timestamp
    
    except subprocess.CalledProcessError as e:
        return f"오류 발생: {e}"

def run_alphafold(consen_seq, output_path):
    """
    Enter consensus sequence and run alphafold based on this to create a new PDB file
    """
    # consen_seq to pdb File
    output_path = f"{output_path}/alphafold_pdb"

    os.mkdir(output_path)

    print("model load start")
    af_model = mk_af_model(protocol="hallucination")
    print("model loaded")
    af_model.prep_inputs(length=len(consen_seq))
    af_model.set_seq(consen_seq)

    seq = af_model._params["seq"].copy()
    print("modeling start")
  
    af_model.predict(verbose=False, hard=False, num_recycles=1)
    af_model._save_results(verbose=False)
    af_model.save_current_pdb(f"{output_path}/consen2alpha.pdb")

    pdb_path = f"{output_path}/consen2alpha.pdb"

    print("modeling done")

    return pdb_path

def parse_ranges(input_data, total_range=(1, 100)):
    """
    Amino acid list to be fixed is received from the consensus sequence and set to residue_range of reps.
    """
    if isinstance(input_data, str):
        input_data = re.findall(r'\d+-\d+|\d+', input_data)
    
    selected_ranges = []
    for part in input_data:
        if '-' in part:
            start, end = map(int, part.split('-'))
            selected_ranges.extend(range(start, end + 1))
        else:
            selected_ranges.append(int(part))
    
    selected_set = set(selected_ranges)
    full_set = set(range(total_range[0], total_range[1] + 1))
    unselected_set = full_set - selected_set
    
    def group_consecutive_numbers(numbers, color):
        reps = []
        sorted_numbers = sorted(numbers)
        if not sorted_numbers:
            return reps
        
        start = sorted_numbers[0]
        prev = start
        
        for num in sorted_numbers[1:]:
            if num != prev + 1:
                reps.append({
                    "model": 0,
                    "chain": "",
                    "resname": "",
                    "style": "cartoon",
                    "color": color,
                    "residue_range": f"{start}-{prev}" if start != prev else str(start)
                })
                start = num
            prev = num
        
        reps.append({
            "model": 0,
            "chain": "",
            "resname": "",
            "style": "cartoon",
            "color": color,
            "residue_range": f"{start}-{prev}" if start != prev else str(start)
        })
        
        return reps
    
    reps = group_consecutive_numbers(selected_set, "blueCarbon") + group_consecutive_numbers(unselected_set, "greenCarbon")
    
    return reps


def extract_seq(ext, file):
    """
    Run the appropriate function according to the format of the uploaded file.
    """
    if ext == ".pdb":
        return read_pdb(file)
    elif ext in [".fasta", ".fa"]:
        result, pdb_id, uniprot_id = read_fasta(file)
        if len(pdb_id) == 1:
            print(pdb_id)
            pdb_path = get_pdb_file(pdb_id[0])
            print("get pdb file from pdb id and saved in ", pdb_path)
            return result, pdb_path
    
        if len(uniprot_id) == 1:
            pdb_path = get_pdb_file(uniprot_id[0])
            print("get pdb file from uniprot id and saved in ", pdb_path)
            return result, pdb_path
        
        else :
            return result, ""
    else:
        return ["Unsupported file format."]

def process_files(files):
    """
    The file is uploaded, the sequence is extracted, and the consensus sequence is extracted through this.
    After that, a new pdb file is created through alphafold, and the list of amino acids to be fixed with this file
    is put into RFdiffusion as input to create a new pdb file, and then evaluated through scoring. 
    Files for all these processes can be downloaded.
    """
    results = []
    consen_pdb = []
    i = 0

    # Extract Sequence data list from file
    # Uploaded Only One file 
    if len(files) == 1 :
        ext = Path(files[0].name).suffix.lower()  # 확장자 추출 (소문자로 변환)

         # uploaded only one pdb file
        if ext == ".pdb" :
            result = extract_seq(ext, files[0])
            print("Uploaded one PDB file, result: ", result)

            results.append(result[0][0])
            print(results)

            # Generate Consensus Sequence (with similarity search...)
            consen = ConsensusSequenceGenerator(results)
            consen_seq, output_path, provide_seq = consen.run()
            
            consen_seq_len = len(consen_seq)

            consen_pdb = files[0]
            rf_timestamp = run_rfdiffusion(consen_pdb, output_path, provide_seq, consen_seq_len)
        
        # uploaded only one fasta file
        else :
            result, pdb_path = extract_seq(ext, files[0])
            print("Uploaded one FASTA file, result: ", result)

            results.append(result[0])

            # Generate Consensus Sequence (with similarity search...)
            consen = ConsensusSequenceGenerator(results)
            consen_seq, output_path, provide_seq = consen.run()
            
            consen_seq_len = len(consen_seq)
            # fasta has Only one pdb_id or uniprot id -> 
            
            if len(pdb_path) > 1:
                i = 1
                consen_pdb = pdb_path
            else:    
                print("consen2alpha start with one fasta no pdb id data")
                consen_pdb = run_alphafold(consen_seq, output_path)
                print("consen2alpha end with one fasta no pdb id data")

            rf_timestamp = run_rfdiffusion(consen_pdb, output_path, provide_seq, consen_seq_len)


    # Uploaded Multiple files
    else:
        # Extract Sequences
        for file in files:
            ext = Path(file.name).suffix.lower()  # 확장자 추출 (소문자로 변환)
            
            result = extract_seq(ext, file)
            print("uploaded multiple files, result: ", result)

            # fasta file have more than one sequences
            if len(result) > 1:
                if len(result[0])>1: 
                    print("one file multi seq")
                    for i in result[0]:
                        results.append(i)
                else:
                    print("one seq")
                    results.append(result[0])
            else:
                results.append(result[0])
        
        print("results :", results)
        print("length ", len(results))
        # Generate Consensus Sequence
        consen = ConsensusSequenceGenerator(results)
        consen_seq, output_path, provide_seq = consen.run()
        consen_seq_len = len(consen_seq)

        print("consen2alpha start with many files")

        consen_pdb = run_alphafold(consen_seq, output_path)

        print("consen2alpha end with many files")

        rf_timestamp = run_rfdiffusion(consen_pdb, output_path, provide_seq, consen_seq_len)


    print(output_path)
    pdb_files = glob.glob(os.path.join(output_path, "*.pdb"))

    matching_files = [file for file in pdb_files if "diffusion" in os.path.basename(file)]
    print("매칭 파일 개수: ", len(matching_files))

    if not matching_files:
        print(f"'{rf_timestamp}'을 포함하는 PDB 파일이 없습니다.")

    rf_score = diffusion_score(matching_files[0], output_path, consen_seq_len)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    get_pdb_path = os.path.join(script_dir, "get_pdb")

    if not os.path.exists(get_pdb_path):
        print("get_pdb 폴더가 존재하지 않습니다.")
    
    else:
        shutil.move(get_pdb_path, output_path)
        if i == 1 :
            pdb_path = f"{output_path}/get_pdb/"
            pdb_files = glob.glob(os.path.join(pdb_path, "*.pdb"))
            consen_pdb = pdb_files[0]
            
        # shutil.rmtree(get_pdb_path) 

    zip_file_path = output_path + ".zip"
    shutil.make_archive(output_path, 'zip', output_path)

    input_data = str(provide_seq)[1:-1]

    rep = parse_ranges(input_data, (1, consen_seq_len))

    return consen_seq, consen_pdb, Molecule3D(value=matching_files, reps=rep), rf_score, gr.DownloadButton(label="Download all output files", value=zip_file_path, visible=True)

def download_files():
    """Return Download Button"""
    return gr.DownloadButton(visible=False)

with gr.Blocks() as demo:
    gr.Markdown("# RFdiffusion with Consensus Sequence")
    file_input = gr.File(label="Upload PDB or FASTA file", file_types=[".pdb", ".fasta"], file_count='multiple')
    textbox = gr.Textbox(label="Consensus Sequence")
    score = gr.DataFrame(label= "RFdiffusion Score", headers=["Design", "N", "MPNN", "PLDDT", "PTM", "PAE", "RMSD", "SEQUENCE"])
    # score = gr.Textbox(label = "RFdiffusion Score")
    
    with gr.Row():
        with gr.Column():
            output = Molecule3D(label="Upload PDB Structure", reps=reps)
        with gr.Column():
            rf_output = Molecule3D(label="RFdiffusion")
    
    btn = gr.Button("Predict")
    download_btn = gr.DownloadButton("Download files", visible=False)

    btn.click(process_files, inputs= file_input, outputs=[textbox, output, rf_output, score, download_btn])
    download_btn.click(download_files, outputs=download_btn)

if __name__ == "__main__":
    demo.launch()