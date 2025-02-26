import subprocess
import os
import random
from datetime import datetime
import json

import numpy as np
from collections import Counter

from Bio import SeqIO
from Bio import AlignIO

def convert_seq_string_to_fasta(seq : str):
    fasta_seq = "\n".join(seq[i:i+60] for i in range(0, len(seq), 60)) + "\n"
    return fasta_seq

class ConsensusSequenceGenerator:
    def __init__(self, input_fasta : list):
        # self.sequences = list(SeqIO.parse(input_fasta, "fasta"))
        self.sequences = input_fasta
        
        self.default_identity_threshold = 0.9
        self.default_mer_size = 5
        self.default_threads = 4

    def _make_folder(self):
        """
        consensus sequence와 관련된 내용들을 저장할 폴더 제작
        """
        date = datetime.now()
        
        folder_index = 1

        base_folder = f"output_{date.year}{date.month:02d}{date.day:02d}_{date.hour:02d}{date.minute:02d}"

        self.folder_name = f"{base_folder}_{folder_index}"

        # 폴더가 이미 존재하는 경우 예외 처리
        while os.path.exists(self.folder_name):
            # 내가 만든 폴더 명이 이미 있다면,
            folder_index += 1 # 뒤 인덱스를 1씩 추가해줌
            self.folder_name = f"{base_folder}_{folder_index}"
        
        os.mkdir(self.folder_name)
        output_path = os.path.join(os.getcwd(), self.folder_name)
        print(f"All Results will be saved in {output_path}")
        print("")
        return output_path

    def _filter_sequence_by_length(self, threshold = 0.3):
        sequence_lengths = np.array([len(sequence) for sequence in self.sequences])
        sequence_median_length = np.median(sequence_lengths)

        # 주어진 임계값을 기준으로, 내가 consensus를 판단할 시퀀스의 길이를 확인
        sequence_min_length = (1 - threshold) * sequence_median_length
        sequence_max_length = (1 + threshold) * sequence_median_length

        # 그 길이를 가지는 시퀀스만 필터링
        sequences_length_filter = np.where((sequence_lengths >= sequence_min_length) & (sequence_lengths <= sequence_max_length))[0]
        # 길이 조건을 만족하는 것들엔 True, 아니면 False
        
        self.sequences_length_filtered = [self.sequences[i] for i in sequences_length_filter]
        
        # self.sequences_length_filtered를 fasta 파일 형식으로 만들어야함.
        with open(os.path.join(self.folder_name, "lengthFiltered.fasta"), "w") as file:
            for i, sequence in enumerate(self.sequences_length_filtered):
                file.write(f">{i}\n{convert_seq_string_to_fasta(sequence)}")

        # SeqIO.write(self.sequences_length_filtered, os.path.join(self.folder_name, "lengthFiltered.fasta"), "fasta")
        
        print("===========================================================================")
        print("Length Filtering Done")
        print(os.path.join(self.folder_name, "lengthFiltered.fasta"))
        print("")
    
    def _run_cdhit(self, identity_threshold = None, mer_size = None, threads = None):
        """
        유사한 서열들을 하나의 군집으로 묶어버림.
        identity_threshold를 기준으로, 그 이상의 유사도를 보이는 서열들을 하나의 군집으로 묶음.
        각 군집에서 대표 서열은, 제일 긴 서열
        default identity_threshold = 0.9
        default mer_size = 5 : k-mer의 k
        default threads  = 4 : 연산 속도 향상을 위한 멀티 쓰레드 사용
        """

        identity_threshold = self.default_identity_threshold if identity_threshold is None else identity_threshold
        mer_size = self.default_mer_size if mer_size is None else mer_size
        threads = self.default_threads if threads is None else threads

        # terminal = ["cd-hit", 
        #             "-i", os.path.join(self.folder_name, "lengthFiltered.fasta"), 
        #             "-o", os.path.join(self.folder_name, "cdhited_lengthFiltered.fasta"),
        #             "-c", str(identity_threshold), 
        #             "-n", str(mer_size), 
        #             "-T", str(threads)]

        terminal = ["/home/manhwi/Consensus/cdhit/cd-hit", 
                    "-i", os.path.join(self.folder_name, "lengthFiltered.fasta"), 
                    "-o", os.path.join(self.folder_name, "cdhited_lengthFiltered.fasta"),
                    "-c", str(identity_threshold), 
                    "-n", str(mer_size), 
                    "-T", str(threads)]
        
        subprocess.run(terminal, 
                       stdout = subprocess.DEVNULL,
                       check = True)
        
        print("===========================================================================")
        print(f"Sequence Clustering(cd-hit) Done, threshold == {identity_threshold}")
        print(os.path.join(self.folder_name, "cdhited_lengthFiltered.fasta"))
        print("")

    def _run_msa_mafft(self, refernce_sequence = None):
        """
        MSA 수행 - MAFFT
        reference 서열을 넣을지 말지 정할 수 있음.
        없다면 fasta 파일에 있는 서열들끼리 정렬됨.

        Arguments
        reference_sequence : 레퍼런스 시퀀스 (fasta 파일)의 위치
        """

        if refernce_sequence is None:
            # terminal = ["mafft", "--auto", 
            #             os.path.join(self.folder_name, "cdhited_lengthFiltered.fasta")]
            terminal = ["/home/manhwi/Consensus/mafft/bin/mafft", "--auto", 
                        os.path.join(self.folder_name, "cdhited_lengthFiltered.fasta")]
        
        else:
            # terminal = ["mafft", "--auto", 
            #             "--keeplength",
            #             "--addfragments", os.path.join(self.folder_name, "cdhited_lengthFiltered.fasta"),
            #             refernce_sequence]
            terminal = ["/home/manhwi/Consensus/mafft/bin/mafft", "--auto", 
                        "--keeplength",
                        "--addfragments", os.path.join(self.folder_name, "cdhited_lengthFiltered.fasta"),
                        refernce_sequence]

        file_path = os.path.join(self.folder_name, "mafft_cdhited_lengthFiltered.fasta")
        
        try:
            with open(file_path, "w") as file:
                result = subprocess.run(terminal, 
                                        stdout = file,
                                        stderr = subprocess.PIPE,
                                        text = True,
                                        check = True)
        
        except subprocess.SubprocessError as e:
            print(e.stderr)
            
        print("===========================================================================")
        print(f"MSA(mafft) Done")
        print(os.path.join(self.folder_name, "mafft_cdhited_lengthFiltered.fasta"))
        print("")
            
    def _make_consensus(self):
        """
        msa 까지 수행한 fasta 파일을 입력 받아,
        해당 fasta 파일에 들어있는 단백질 서열들의 Consensus를 찾아주는 함수
        """
        alignment = AlignIO.read(os.path.join(self.folder_name, "mafft_cdhited_lengthFiltered.fasta"), "fasta")
        sequences = [str(record.seq) for record in alignment] # 단백질 서열 리스트

        seq_length = len(sequences[0])  # 정렬된 서열의 길이
        consensus_sequence = [] # Consensus Sequence를 저장할 리스트

        amino_counts_in_consensus = []

        for i in range(seq_length):
            amino_counts_in_column = {aa : 0 for aa in "ACDEFGHIKLMNPQRSTVWY-"}  # '-'는 gap
            # 한 column 내에서 아미노산의 등장 횟수를 세주는 딕셔너리

            column = [seq[i] for seq in sequences] # 특정 위치의 모든 서열 문자 추출
            counts = Counter(column).most_common() # 가장 많이 등장한게 맨 앞에 오도록 정렬

            highest_count = counts[0][1] # 가장 많이 나온 횟수 확인
            # frequent_amino_acids = [counts[0][0]] # 가장 많이 나온 아미노산 넣어주기
            # (20250211 수정) 가장 많이 나온 아미노산을 넣을 필요가 없음.
            # 밑에서 넣을 것이기 때문
            frequent_amino_acids = [] # 가장 많이 나온 아미노산 넣어주기

            if counts[0][0] == "-" and highest_count/len(column) >= 0.5:
                # gap 비율이 0.5를 넘어간다면, 굳이 아래 코드 안 돌려도 됨
                continue

            if counts[0][0] == "-" and highest_count/len(column) < 0.5:
                # gap이 제일 많이 나왔지만, 그 비율이 0.5가 안된다면
                highest_count = counts[1][1]
                # frequent_amino_acids = [counts[1][0]]

            for amino_acid, count in counts:
                amino_counts_in_column[amino_acid] = count
                # 아미노산의 등장 횟수 추출

                # 가장 많이 나온게 몇 개 더 있다면, 추가해주기
                # (20250211 수정) gap이 추가되는 경우가 있어서 제거
                if count == highest_count and amino_acid != "-":
                    frequent_amino_acids.append(amino_acid)

            # 여기까지 오면, 각 위치에서 아미노산 들이 얼마나 나왔는지 확인이 됐고,
            # 그리고 어떤 아미노산(들)이 제일 많이 나왔는지 확인도 됐음.
            # 가장 많이 나온 아미노산 중 랜덤하게 하나를 뽑아서 이를 Consensus Sequence에 입력
            # print(frequent_amino_acids)

            consensus_amino = random.choice(frequent_amino_acids)
            consensus_sequence.append(consensus_amino)
            # print(consensus_amino)
            # 각 위치에서 아미노산의 등장 횟수를 확인
            amino_counts_in_consensus.append(amino_counts_in_column)

        consensus_sequence = "".join(consensus_sequence)

        with open(os.path.join(self.folder_name, "consensus_sequence.fasta"), "w") as file:
            file.write(">Consensus Sequence\n")
            file.write(f"{convert_seq_string_to_fasta(consensus_sequence)}")
            # consensus sequence 저장

        with open(os.path.join(self.folder_name, "amino_acids_counts_in_consensus_sequence.json"), "w", encoding="utf-8") as file:
            json.dump(amino_counts_in_consensus, file, indent = 4)
            # amino acid 등장 비율 저장

        return consensus_sequence
    
    def run(self):
        """
        consensus 서열 제작 일련의 과정 수행
        """
        output_path = self._make_folder()
        self._filter_sequence_by_length()
        self._run_cdhit()
        self._run_msa_mafft()
        consen_seq = self._make_consensus()

        return consen_seq, output_path
        
if __name__ == "__main__":

    seqs = list(SeqIO.parse("test_sequences.fasta", "fasta"))
    seqs = [str(rec.seq) for rec in seqs]

    csg = ConsensusSequenceGenerator(seqs)
    csg.run()