from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import asyncio
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.oauth2 import service_account

from llamaapi import LlamaAPI
import os
from dotenv import load_dotenv

load_dotenv()


class TaskType(Enum):
    """Types of tasks that can be debated."""

    ARITHMETIC = "arithmetic"
    GSM8K = "gsm8k"
    CHESS = "chess"
    BIOGRAPHIES = "biographies"
    MMLU = "mmlu"
    CHESS_VALIDITY = "chess_validity"
    GENERAL = "general"  # For arbitrary user prompts


@dataclass
class TaskInput:
    """Base class for task-specific inputs"""

    pass


@dataclass
class ArithmeticInput(TaskInput):
    """Input for arithmetic tasks"""

    numbers: List[int]


@dataclass
class GSM8KInput(TaskInput):
    """Input for GSM8K math word problems"""

    problem: str


@dataclass
class ChessInput(TaskInput):
    """Input for chess move problems"""

    moves: List[str]


@dataclass
class BiographiesInput(TaskInput):
    """Input for biography generation"""

    person: str


@dataclass
class MMLUInput(TaskInput):
    """Input for MMLU questions"""

    question: str
    choices: Dict[str, str]  # e.g. {"A": "choice1", "B": "choice2", ...}


@dataclass
class ChessValidityInput(TaskInput):
    """Input for chess move validity problems"""

    game: str
    position: str


@dataclass
class GeneralInput(TaskInput):
    """Input for general tasks with arbitrary prompts"""

    prompt: str  # Complete prompt to be used directly


@dataclass
class Task:
    """Represents a task for debate with its associated prompts."""

    task_type: TaskType
    starting_prompt: str
    debate_prompt: str
    reflection_prompt: str

    def format_prompt(self, task_input: TaskInput) -> str:
        """Formats the starting prompt with the given task input"""
        if self.task_type == TaskType.ARITHMETIC:
            assert isinstance(task_input, ArithmeticInput)
            return self.starting_prompt.format(*task_input.numbers)
        elif self.task_type == TaskType.GSM8K:
            assert isinstance(task_input, GSM8KInput)
            return self.starting_prompt.format(problem=task_input.problem)
        elif self.task_type == TaskType.CHESS:
            assert isinstance(task_input, ChessInput)
            return self.starting_prompt.format(moves=" ".join(task_input.moves))
        elif self.task_type == TaskType.BIOGRAPHIES:
            assert isinstance(task_input, BiographiesInput)
            return self.starting_prompt.format(person=task_input.person)
        elif self.task_type == TaskType.MMLU:
            assert isinstance(task_input, MMLUInput)
            return self.starting_prompt.format(
                question=task_input.question,
                **{k: v for k, v in task_input.choices.items()},
            )
        elif self.task_type == TaskType.CHESS_VALIDITY:
            assert isinstance(task_input, ChessValidityInput)
            return self.starting_prompt.format(
                game=task_input.game, position=task_input.position
            )
        elif self.task_type == TaskType.GENERAL:
            assert isinstance(task_input, GeneralInput)
            return task_input.prompt  # Use the prompt directly
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")


# Example task definitions with input types
GENERAL_TASK = Task(
    task_type=TaskType.GENERAL,
    starting_prompt="",  # Empty as the prompt comes from the input
    debate_prompt="These are the recent/updated opinions from other agents: {context} Based on these opinions, can you give an updated response?",
    reflection_prompt="Here is your original response: {original_response}\n\nCan you reflect on this response and provide any updates or corrections if needed?",
)

ARITHMETIC_TASK = Task(
    task_type=TaskType.ARITHMETIC,
    starting_prompt="What is the result of {}+{}*{}+{}-{}*{}? Keep explanation short, make sure to state your answer at the end of the response. Return final answer integer X only",
    debate_prompt="These are the recent/updated opinions from other agents: {context} Use these opinions carefully as additional advice, can you provide an updated answer? Keep explanation short, make sure to state your answer at the end of the response. Return final answer integer X only",
    reflection_prompt="Here is your original response: {original_response}\n\nCan you verify that your answer is correct? Please reiterate your answer, making sure to state your answer at the end of the response.",
)

GSM8K_TASK = Task(
    task_type=TaskType.GSM8K,
    starting_prompt="Can you solve the following math problem? {problem} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.",
    debate_prompt="These are the solutions to the problem from other agents: {context} Using the solutions from other agents as additional information, can you provide your answer to the math problem? Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.",
    reflection_prompt="Here is your original response: {original_response}\n\nCan you double check that your answer is correct? Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}.",
)

CHESS_TASK = Task(
    task_type=TaskType.CHESS,
    starting_prompt="Here is the current sequence of moves in a chess game: {moves}. What is the best chess move I should execute next? Give a single move suggestion of the form 14. <XXX> and make sure the chess move is valid in the current board state.",
    debate_prompt="Here are other chess move suggestions from other agents: {context} Using the chess suggestions from other agents as additional advice, can you give me your updated thoughts on the best next chess move I should play given the chess sequence? Give a single move suggestion of the form 14. <XXX> and make sure the chess move is valid in the current board state.",
    reflection_prompt="Here is your original response: {original_response}\n\nGiven the board state and your previous move suggestion, can you confirm your best next chess move? Give a single move suggestion of the form 14. <XXX> and make sure the chess move is valid in the current board state.",
)

BIOGRAPHIES_TASK = Task(
    task_type=TaskType.BIOGRAPHIES,
    starting_prompt="Give a bullet point biography of {person} highlighting their contributions and achievements as a computer scientist, with each fact separated with a new line character.",
    debate_prompt="Here are some bullet point biographies of {person} given by other agents: {context} Closely examine your biography and the biography of other agents and provide an updated bullet point biography.",
    reflection_prompt="Here is your original response: {original_response}\n\nClosely examine your biography and provide an updated bullet point biography.",
)

MMLU_TASK = Task(
    task_type=TaskType.MMLU,
    starting_prompt="Can you answer the following question as accurately as possible? : {question} A) {a}, B) {b}, C) {c}, D) {d} Explain your answer, putting the answer in the form (X) at the end of your response.",
    debate_prompt="These are the solutions to the problem from other agents: {context} Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents. Put your answer in the form (X) at the end of your response.",
    reflection_prompt="Here is your original response: {original_response}\n\nCan you double check that your answer is correct? Put your final answer in the form (X) at the end of your response.",
)

CHESS_VALIDITY_TASK = Task(
    task_type=TaskType.CHESS_VALIDITY,
    starting_prompt="Given the chess game {game}, give one valid destination square for the chess piece at {position}. State the destination square in the form (X), where X follows the regex [a-h][1-8], for example (e5). Give a one line explanation of why your destination square is a valid move.",
    debate_prompt="Here are destination square suggestions from other agents: {context} Can you double check that your destination square is a valid move? Check the valid move justifications from other agents. State your final answer in a newline with a 2 letter response following the regex [a-h][1-8].",
    reflection_prompt="Here is your original response: {original_response}\n\nCan you double check that your answer is valid? State your final answer in a newline with a 2 letter response following the regex [a-h][1-8].",
)


class Agent:
    """Represents an individual debating agent."""

    def __init__(self, agent_id: int, model_name: str, use_llama_api: bool = False):
        self.agent_id = agent_id
        self.model_name = model_name
        self.current_response = None
        self.previous_responses = []
        self.use_llama_api = use_llama_api

        if not self.use_llama_api:
            # Initialize credentials from service account file
            self.credentials = service_account.Credentials.from_service_account_file(
                "service-account.json",
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            # Initialize AI Platform client
            self.client = aiplatform.gapic.PredictionServiceClient(
                client_options={
                    "api_endpoint": "us-central1-aiplatform.googleapis.com"
                },
                credentials=self.credentials,
            )
        else:
            # Initialize LlamaAPI client
            self.llama_client = LlamaAPI(os.getenv("LLAMA_API_KEY"))

    async def generate_response(
        self, prompt: str, context: Optional[str] = None
    ) -> str:
        """
        Generates a response from the agent using either the Vertex AI platform or LlamaAPI.
        Args:
            prompt: The prompt to send to the model
            context: Optional context from previous debate rounds
        Returns:
            The generated response
        """
        # Construct the full prompt with context if provided
        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        if self.use_llama_api:
            # Construct the API request for Llama
            api_request_json = {
                "model": "llama3.1-70b",  # Using the Llama 70B model
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant participating in a debate.",
                    },
                    {"role": "user", "content": full_prompt},
                ],
                "stream": False,
            }
            try:
                # Execute the request using asyncio.to_thread to make it async
                response = await asyncio.to_thread(
                    self.llama_client.run, api_request_json
                )
                response_json = response.json()

                # Extract the response content
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    generated_text = response_json["choices"][0]["message"]["content"]
                else:
                    generated_text = "No response generated"

            except Exception as e:
                print(f"Error in Llama API request: {e}")
                generated_text = "Error generating response"

        else:
            # Original Vertex AI implementation
            instance = {
                "prompt": full_prompt,
                "max_tokens": 2048,
            }

            # Convert instance to protobuf Value
            instances = [json_format.ParseDict(instance, Value())]
            parameters = json_format.ParseDict({}, Value())

            # Get the endpoint path
            endpoint = self.client.endpoint_path(
                project="sakana-ai-445705",
                location="us-central1",
                endpoint="6294933866947805184",
            )

            try:
                # Make the prediction request
                response = await asyncio.to_thread(
                    self.client.predict,
                    endpoint=endpoint,
                    instances=instances,
                    parameters=parameters,
                )

                # Extract the generated text from the response
                generated_text = str(response.predictions[0])

            except Exception as e:
                print(f"Error in Vertex AI request: {e}")
                generated_text = "Error generating response"

        # Store the response
        self.current_response = generated_text
        self.previous_responses.append(generated_text)

        return generated_text


async def test_llama_agent():
    try:
        # Create an agent with LlamaAPI
        bob = Agent(agent_id=1, model_name="llama70b", use_llama_api=True)

        # Test with a simple prompt
        prompt = "What is 2 + 2?"
        print(f"Sending prompt: {prompt}")

        response = await bob.generate_response(prompt=prompt)
        print(f"\nResponse from LlamaAPI:")
        print(response)

        # Test with a more complex prompt
        complex_prompt = """
        Can you explain how photosynthesis works in three sentences?
        """
        print(f"\nSending complex prompt: {complex_prompt}")

        response = await bob.generate_response(prompt=complex_prompt)
        print(f"\nResponse from LlamaAPI:")
        print(response)

    except Exception as e:
        print(f"Error occurred: {str(e)}")


# Run the test
# if __name__ == "__main__":
#     # Load environment variables
#     load_dotenv()

#     # Verify LLAMA_API_KEY is set
#     if not os.getenv("LLAMA_API_KEY"):
#         print("Error: LLAMA_API_KEY not found in environment variables")
#     else:
#         asyncio.run(test_llama_agent())
