import re

def parse_ranges(input_data, total_range=(1, 100)):
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
    
    reps = group_consecutive_numbers(selected_set, "blue") + group_consecutive_numbers(unselected_set, "green")
    
    return reps

# Example usage
input_data = "32, 33-37, 54, 56-58"
total_residue_range = (1, 100)  # Adjust the total range as needed
result = parse_ranges(input_data, total_residue_range)
print(result)



# input_data = [32, "33-37", 54, "56-58", ""]
# # Example usage
input_data = [33,37-38,40,44,69,76,81,84,95-96,98-99,106-107,125,130,132,134,136,141,148,152,172,184,187-188,195-196,200-201,203,206,208,222,230,235-236,245-246,248,263,266,269,272,288,316,319,323,335,341,353,369]
input_data = str(input_data)

print(input_data[1:-1])

# total_residue_range = (1, 100)  # Adjust the total range as needed
# result = parse_ranges(input_data, total_residue_range)
# print(result)
