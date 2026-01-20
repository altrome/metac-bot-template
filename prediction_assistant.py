"""
Prediction Assistant Script

This script helps humans make manual predictions for tournaments.
It generates predictions without posting them to Metaculus, storing them locally instead.

Usage:
    python prediction_assistant.py <tournament_code>

Example:
    python prediction_assistant.py 32916
    python prediction_assistant.py q1-2025-cup

The predictions are saved to a local file: <code>_<date>_<time>.log
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Import functions from main_with_no_framework.py
from main_with_no_framework import (
    get_open_question_ids_from_tournament,
    forecast_individual_question,
    get_post_details,
    run_research,
)
from binary_questions import get_binary_gpt_prediction
from numeric_questions import get_numeric_gpt_prediction
from multiple_choice_questions import get_multiple_choice_gpt_prediction


def get_tournament_id(tournament_code: str):
    """
    Convert tournament code to ID if it's a string code, or return as int if numeric.
    
    Args:
        tournament_code: Tournament code or ID (e.g., "32916" or "spring-aib-2026")
    
    Returns:
        int or str: Tournament ID (can be numeric or string)
    """
    # Try to convert to int first
    try:
        return int(tournament_code)
    except ValueError:
        # If it's not numeric, return as string (Metaculus API supports both)
        return tournament_code


def create_log_filename(tournament_code: str) -> str:
    """
    Create log filename with format: predictions/<code>_<date>_<time>.log
    
    Args:
        tournament_code: Tournament code or ID
    
    Returns:
        str: Log filename path
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    
    # Ensure predictions directory exists
    predictions_dir = Path("predictions")
    predictions_dir.mkdir(exist_ok=True)
    
    return f"predictions/{tournament_code}_{date_str}_{time_str}.log"


async def generate_prediction_for_question(
    question_id: int,
    post_id: int,
    num_runs: int = 5,
) -> dict:
    """
    Generate prediction for a single question without posting it.
    
    Args:
        question_id: Question ID
        post_id: Post ID
        num_runs: Number of prediction runs to perform (median is taken)
    
    Returns:
        dict: Prediction details including forecast and comment
    """
    post_details = get_post_details(post_id)
    question_details = post_details["question"]
    title = question_details["title"]
    question_type = question_details["type"]
    
    print(f"\n{'='*80}")
    print(f"Processing Question {question_id} (Post {post_id})")
    print(f"Title: {title}")
    print(f"Type: {question_type}")
    print(f"URL: https://www.metaculus.com/questions/{post_id}/")
    print(f"{'='*80}\n")
    
    # Generate prediction based on question type
    if question_type == "binary":
        forecast, comment = await get_binary_gpt_prediction(
            question_details, num_runs, run_research
        )
    elif question_type == "numeric":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs, run_research
        )
    elif question_type == "discrete":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs, run_research
        )
    elif question_type == "multiple_choice":
        options = question_details["options"]
        print(f"Options: {options}")
        forecast, comment = await get_multiple_choice_gpt_prediction(
            question_details, num_runs, run_research
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")
    
    return {
        "question_id": question_id,
        "post_id": post_id,
        "title": title,
        "question_type": question_type,
        "url": f"https://www.metaculus.com/questions/{post_id}/",
        "forecast": forecast,
        "comment": comment,
    }


async def generate_predictions_for_tournament(tournament_code: str) -> None:
    """
    Generate predictions for all open questions in a tournament.
    Saves predictions to a local log file without posting them.
    
    Args:
        tournament_code: Tournament code or ID
    """
    # Get tournament ID
    tournament_id = get_tournament_id(tournament_code)
    
    # Create log file
    log_filename = create_log_filename(tournament_code)
    log_path = Path(log_filename)
    
    print(f"\n{'#'*80}")
    print(f"# Prediction Assistant for Tournament: {tournament_code}")
    print(f"# Tournament ID: {tournament_id}")
    print(f"# Log file: {log_filename}")
    print(f"# NOTE: Predictions will NOT be posted to Metaculus")
    print(f"{'#'*80}\n")
    
    # Get open questions
    print("Fetching open questions from tournament...")
    open_questions = get_open_question_ids_from_tournament(tournament_id)
    
    if not open_questions:
        message = f"No open questions found in tournament {tournament_id}"
        print(message)
        log_path.write_text(message)
        return
    
    print(f"\nFound {len(open_questions)} open question(s)\n")
    
    # Open log file for writing
    with log_path.open('w', encoding='utf-8') as log_file:
        # Write header
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Prediction Assistant Log\n")
        log_file.write(f"Tournament Code: {tournament_code}\n")
        log_file.write(f"Tournament ID: {tournament_id}\n")
        log_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total Questions: {len(open_questions)}\n")
        log_file.write(f"{'='*80}\n\n")
        
        # Process each question
        for idx, (question_id, post_id) in enumerate(open_questions, 1):
            print(f"\nProcessing question {idx}/{len(open_questions)}...")
            
            try:
                prediction = await generate_prediction_for_question(
                    question_id, post_id
                )
                
                # Write to log file
                log_file.write(f"\n{'='*80}\n")
                log_file.write(f"Question {idx}/{len(open_questions)}\n")
                log_file.write(f"{'='*80}\n")
                log_file.write(f"Question ID: {prediction['question_id']}\n")
                log_file.write(f"Post ID: {prediction['post_id']}\n")
                log_file.write(f"Title: {prediction['title']}\n")
                log_file.write(f"Type: {prediction['question_type']}\n")
                log_file.write(f"URL: {prediction['url']}\n")
                log_file.write(f"\nForecast:\n")
                
                # Format forecast based on type
                if prediction['question_type'] in ['numeric', 'discrete']:
                    log_file.write(f"{str(prediction['forecast'])[:1000]}...\n")
                else:
                    log_file.write(f"{prediction['forecast']}\n")
                
                log_file.write(f"\nComment:\n")
                log_file.write(f"{prediction['comment']}\n")
                log_file.write(f"\n{'='*80}\n")
                
                # Flush to ensure data is written
                log_file.flush()
                
                print(f"✓ Successfully generated prediction for question {question_id}")
                
            except Exception as e:
                error_msg = f"Error processing question {question_id}: {str(e)}"
                print(f"✗ {error_msg}")
                log_file.write(f"\n{'='*80}\n")
                log_file.write(f"ERROR - Question {idx}/{len(open_questions)}\n")
                log_file.write(f"{'='*80}\n")
                log_file.write(f"Question ID: {question_id}\n")
                log_file.write(f"Post ID: {post_id}\n")
                log_file.write(f"Error: {error_msg}\n")
                log_file.write(f"{'='*80}\n\n")
                log_file.flush()
    
    print(f"\n{'#'*80}")
    print(f"# Prediction generation complete!")
    print(f"# Predictions saved to: {log_filename}")
    print(f"# Total questions processed: {len(open_questions)}")
    print(f"# NOTE: No predictions were posted to Metaculus")
    print(f"{'#'*80}\n")


def main():
    """Main entry point for the prediction assistant."""
    if len(sys.argv) < 2:
        print("Error: Tournament code is required")
        print()
        print("Usage: python prediction_assistant.py <tournament_code>")
        print()
        print("Examples:")
        print("  python prediction_assistant.py 32916")
        print("  python prediction_assistant.py q1-2025-cup")
        sys.exit(1)
    
    tournament_code = sys.argv[1]
    
    try:
        asyncio.run(generate_predictions_for_tournament(tournament_code))
    except KeyboardInterrupt:
        print("\n\nPrediction generation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
