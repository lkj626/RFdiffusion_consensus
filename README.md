# RFdiffusion with Consensus

## Workflow
0. **Input Files**
   - Upload `.pdb` or `.fasta` format files.

1. **Extract Sequences**
   - Retrieve sequences from the input files.

2. **Determine Consensus Sequence**
   - Determine the **consensus sequence** based on the extracted sequences.

3. **Predict Structure using AlphaFold**
   - Use **AlphaFold** to predict the 3D protein structure from the consensus sequence.

4. **Redesign with RFDiffusion**
   - Use the AlphaFold-predicted structure as input for **RFDiffusion**.
   - Fix amino acids that appeared with high frequency in the consensus sequence.
   - Design the remaining parts to generate new sequences.

5. **Evaluate with ProteinMPNN & AlphaFold**
   - Evaluate the designed sequences using **ProteinMPNN & AlphaFold**.

6. **Output the Best Structure & Provide Downloadable Files**
   - Select and output the highest-scoring designed structure.  
   - All generated files from each step are download available.

  
![input](file_upload.gif)


![output](output.gif)


## Downloading Project from Git


```
git clone --recursive https://github.com/molcube/rfdiffusion-consensus.git
```

## Installation via Anaconda

```
conda env create --name <your_env> -f base-env.yml
```

## Infernece



## Output
