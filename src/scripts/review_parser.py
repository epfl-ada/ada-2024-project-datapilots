import json
import math
from pathlib import Path

# Paths and configuration variables
base_dir = Path(__file__).resolve().parent.parent.parent
BA_RATINGS = base_dir / 'data' / 'BeerAdvocate' / 'ratings.txt'
RB_RATINGS = base_dir / 'data' / 'RateBeer' / 'ratings.txt'
OUTPUT_TEMPLATE = 'reviews_chunk.json'
# we are not interested in the field called "text" as we will not use textual review in our analysis
FIELDS_OF_INTEREST = ['beer_name', 'beer_id', 'brewery_name', 'brewery_id', 'style', 'abv', 'date', 'user_name', 'user_id', 'appearance', 'aroma', 'palate', 'taste', 'overall', 'rating']
NUM_PARTS = 3

def count_lines_in_file(file_path):
    """
    Counts the total number of lines in a file.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        int: Total number of lines in the file.
    """
    with open(file_path, "r", encoding="utf-8", errors='ignore') as file:
        return sum(1 for _ in file)

def divide_lines_into_ranges(total_lines, num_parts):
    """
    Divides the total lines in the file into approximately equal parts.
    
    Args:
        total_lines (int): Total number of lines in the file.
        num_parts (int): Number of parts to divide the lines into.
    
    Returns:
        list: A list of [start, end] pairs representing line ranges for each part.
    """
    chunk_size = math.ceil(total_lines / num_parts)
    return [[i * chunk_size, min((i + 1) * chunk_size - 1, total_lines - 1)] for i in range(num_parts)]

def parse_reviews_in_range(file_path, requested_fields, line_range):
    """
    Parses reviews within a given line range and extracts specified fields.
    
    Args:
        file_path (str): Path to the reviews file.
        requested_fields (list): List of fields to extract from each review.
        line_range (list): [start, end] specifying the line range to process.
    
    Returns:
        list: List of dictionaries, each representing a parsed review with requested fields.
    """
    start, end = line_range
    reviews = []
    current_review = {}
    line_index = 0
    
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            if start <= line_index <= end:
                line = line.strip()
                if line == '':
                    if current_review:
                        reviews.append(current_review)
                        current_review = {}
                else:
                    field, _, value = line.partition(':')
                    field, value = field.strip(), value.strip()
                    if field in requested_fields:
                        current_review[field] = value
            elif line_index > end:
                break
            line_index += 1
    
    if current_review:
        reviews.append(current_review)  # Add the last review if it doesn't end with a newline
    
    return reviews

def process_and_save_reviews(input_file, output_template, requested_fields, num_parts):
    """
    Processes the reviews file by dividing it into parts, parsing each part, and saving as JSON.
    
    Args:
        input_file (str): Path to the input reviews file.
        output_template (str): Template for the output JSON filename (e.g., "reviews_chunk.json").
        requested_fields (list): Fields to extract from each review.
        num_parts (int): Number of parts to divide the file into.
    """
    # Count the total number of lines in the file
    total_lines = count_lines_in_file(input_file)
    
    # Divide lines into ranges for each part
    line_ranges = divide_lines_into_ranges(total_lines, num_parts)
    
    # Determine the directory for the output files (same as input file's directory)
    output_dir = Path(input_file).parent

    # Process each range and save the parsed data to a separate JSON file
    for part_number, line_range in enumerate(line_ranges):
        reviews_data = parse_reviews_in_range(input_file, requested_fields, line_range)
        
        # Create the output file path within the input file’s directory
        output_file = output_dir / f"{part_number}_{output_template}"
        
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(reviews_data, file, indent=4)
        
        print(f"Saved part {part_number + 1}/{num_parts} to {output_file} with {len(reviews_data)} entries.")


if __name__ == "__main__":
    # Run the processing and saving function for the 2 review files
    process_and_save_reviews(BA_RATINGS, OUTPUT_TEMPLATE, FIELDS_OF_INTEREST, NUM_PARTS)
    process_and_save_reviews(RB_RATINGS, OUTPUT_TEMPLATE, FIELDS_OF_INTEREST, NUM_PARTS)

