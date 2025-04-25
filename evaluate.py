import pandas as pd
import requests
import time
import argparse
from typing import Literal

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate RAG system with different retrieval methods')
    parser.add_argument(
        '--method',
        type=str,
        choices=['native', 'hybrid'],
        default='native',
        help='Retrieval method to use (native or hybrid)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='evaluasi_data.xlsx',
        help='Input Excel file path'
    )
    parser.add_argument(
        '--delay',
        type=int,
        default=3,
        help='Delay between requests in seconds'
    )
    return parser

def evaluate_questions(
    file_path: str,
    method: Literal['native', 'hybrid'],
    delay: int = 3
) -> None:
    """
    Evaluate questions using specified retrieval method
    
    Args:
        file_path (str): Path to Excel file containing questions
        method (str): Retrieval method ('native' or 'hybrid')
        delay (int): Delay between requests in seconds
    """
    # Baca file Excel
    df = pd.read_excel(file_path)
    api_url = "http://localhost:8000/api/v1/ask"
    
    output_column = f'output_{method}'  # Separate column for each method
    
    # Add output column if it doesn't exist
    if output_column not in df.columns:
        df[output_column] = None

    print(f"\nStarting evaluation using {method.upper()} method...")
    print(f"Total questions: {len(df)}\n")

    for index, row in df.iterrows():
        try:
            question = row['pertanyaan']
            print(f"Processing question {index + 1}: {question}")
            
            # Prepare request payload
            payload = {
                "session_id": f"eval_{index}",
                "query": question,
                "method": method
            }
            
            # Send request
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                # Store answer and metadata
                result = response.json()
                df.at[index, output_column] = result['answer']
                
                # Save after each successful response
                df.to_excel(file_path, index=False)
                print(f"✓ Answer saved for question {index + 1}")
                print(f"✓ Response: {result['answer']}")
            else:
                print(f"✗ Error for question {index + 1}: {response.status_code}")
            
            # Add delay between requests
            time.sleep(delay)
            
        except Exception as e:
            print(f"✗ Skip question {index + 1}: {str(e)}")
            continue

    print(f"\nEvaluation completed for {method} method!")
    print(f"Results saved to {file_path}")

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        evaluate_questions(
            file_path=args.input,
            method=args.method,
            delay=args.delay
        )
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()
