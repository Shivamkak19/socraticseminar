import random
from typing import Tuple, List, Union
from datasets import load_dataset
import json

from debateTypes import (
    TaskType,
    TaskInput,
    ArithmeticInput,
    GSM8KInput,
    ChessInput,
    BiographiesInput,
    MMLUInput,
    ChessValidityInput,
    GeneralInput,
    Task,
    Agent,
)


def generate_arithmetic_input() -> Tuple[int, List[int]]:
    """
    Generates a sequence of 6 random digits returned as type ArithmeticInput

    Sequence of operations from paper:
    {}+{}*{}+{}-{}*{}

    Returns:
        Tuple[int, List[int]]: A tuple containing:
            - The result of the arithmetic operation
            - A list of 6 random integers between 0 and 99
    """
    numbers = []
    for i in range(6):
        numbers.append(random.randint(0, 99))

    result = (
        numbers[0] + (numbers[1] * numbers[2]) + numbers[3] - (numbers[4] * numbers[5])
    )

    return (result, numbers)


def get_random_gsm8k_question() -> Tuple[str, int]:
    """
    Draws a random question from the GSM8K test set and returns
    the question text and numeric answer as separate values.

    Returns:
        Tuple[str, int]: A tuple containing:
            - question_text: The text of the math question
            - answer: The numeric answer as an integer
    """
    # Load the dataset and get test split
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]

    # Get a random index from the test set
    random_idx = random.randint(0, len(test_data) - 1)

    # Get the question-answer pair
    qa_pair = test_data[random_idx]

    # The question is directly available
    question = qa_pair["question"]

    # Parse the answer - it's after the #### marker
    answer_text = qa_pair["answer"]
    numeric_answer = int(answer_text.split("####")[-1].strip())

    return question, numeric_answer


def get_random_chess_position() -> Tuple[str, List[str]]:
    """Get a random chess position and its valid next moves.

    Returns:
        Tuple[str, List[str]]: A tuple containing:
            - input_position: String containing the chess position in UCI notation
            - valid_moves: List of strings representing valid destination squares
    """
    # Sample test case from synthetic_short task
    with open("bigBenchChessRealMedium.json", "r") as f:
        task_data = json.load(f)

    # Get examples section
    examples = task_data["examples"]

    # Select random example
    example = random.choice(examples)

    # Get input string and target valid moves
    input_position = example["input"]
    valid_moves = example["target"]

    return input_position, valid_moves


def get_random_mmlu_question() -> Tuple[MMLUInput, str]:
    """
    Gets a random question from the MMLU dataset across all categories.

    Returns:
        Tuple[MMLUInput, str]: A tuple containing:
            - MMLUInput object with the question and choices
            - correct answer letter (A, B, C, or D)
    """
    # Load the full MMLU dataset
    dataset = load_dataset("cais/mmlu", "all")

    # Get the test split which contains the questions
    test_data = dataset["test"]

    # Get a random index
    random_idx = random.randint(0, len(test_data) - 1)

    # Get the question data
    question_data = test_data[random_idx]

    # Extract the question and choices
    question = question_data["question"]
    choices = {
        "A": question_data["choices"][0],
        "B": question_data["choices"][1],
        "C": question_data["choices"][2],
        "D": question_data["choices"][3],
    }

    # Create MMLUInput object
    mmlu_input = MMLUInput(question=question, choices=choices)

    # Get the correct answer (convert from numeric to letter)
    answer = chr(65 + question_data["answer"])  # Convert 0,1,2,3 to A,B,C,D

    return mmlu_input, answer
