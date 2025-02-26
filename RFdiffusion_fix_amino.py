def calculate_consensus_frequencies(aa_counts_in_consensus, consensus_sequence):
    """
    Consensus에서 각 아미노산의 등장 비율을 확인하여,
    비율 기준으로 내림차순 정렬된 dictionary를 return
    """
    aa_common_freqs = {}
    for aa_index, amino_acid in enumerate(consensus_sequence):
        if aa_index == 0:
            # 맨 처음에 총 서열 개수 확인
            number_of_seqs = sum(aa_counts_in_consensus[aa_index].values())
            print(number_of_seqs)

        aa_count = aa_counts_in_consensus[aa_index][amino_acid]
        aa_common_freqs[aa_index] = [amino_acid, aa_count/number_of_seqs]
    
    # aa_common_freqs = dict(sorted(aa_common_freqs.items(), key = lambda item: item[1][1], reverse=True))

    return aa_common_freqs

def get_fixed_amino_acids(aa_common_freqs, threshold):
    aa_fixed = {key: value for key, value in aa_common_freqs.items() if value[1] > threshold}
    aa_fixed = list(aa_fixed.keys())
    return aa_fixed

def merge_continuous_amino_acids(aa_fixed):
    if not aa_fixed:
        return []

    aa_fixed.sort()  # 정렬 (이미 정렬된 상태여도 확실히 하기 위해)
    aa_fixed_result = []
    start = aa_fixed[0]
    prev = aa_fixed[0]

    for aa_index in aa_fixed[1:]:
        if aa_index == prev + 1:  # 연속된 경우
            prev = aa_index
        else:  # 연속이 끊긴 경우
            aa_fixed_result.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = aa_index

    # 마지막 그룹 추가
    aa_fixed_result.append(f"{start}-{prev}" if start != prev else f"{start}")

    provide_seq = f"[{",".join(aa_fixed_result)}]"

    return provide_seq