import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import textwrap
import time  # For handling potential rate limits
from typing import Any, Optional

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Generative AI model
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error(
        "Google AI Studio API key not found. Please add it to your .env file.  "
        "You can obtain an API key from https://makersuite.google.com/."
    )
    st.stop()  # Stop if the API key is missing

st.title("Prompt Engineering Playground")
st.subheader("Experiment with Prompting Techniques")

# --- Helper Functions ---
def code_block(text: str, language: str = "text") -> None:
    """Displays text as a formatted code block in Streamlit."""
    st.markdown(f"```{language}\n{text}\n```", unsafe_allow_html=True)


def display_response(response: Any) -> None:
    """Displays the model's response, handling text, and error cases."""
    if response.text:
        st.subheader("Generated Response:")
        st.markdown(response.text)
    elif response.prompt_feedback:
        st.warning(f"Prompt Feedback: {response.prompt_feedback}")
    else:
        st.error("No response or feedback received from the model.")
        st.error(f"Full response object: {response}")  # Print the full response for debugging


def generate_with_retry(prompt: str, model_name: str, generation_config: genai.types.GenerationConfig, max_retries: int = 3, delay: int = 5) -> Any:
    """
    Generates content with retry logic to handle potential API errors (e.g., rate limits, model not found).
    Args:
        prompt: The prompt string.
        model_name: The name of the model to use.
        generation_config: The generation configuration.
        max_retries: Maximum number of retries.
        delay: Delay in seconds between retries.
    Returns:
        The generated response.
    Raises:
        Exception: If the generation fails after maximum retries or a critical error occurs.
    """
    for i in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)  # Use the selected model name
            response = model.generate_content(prompt, generation_config=generation_config)
            return response
        except Exception as e:
            error_message = str(e)
            st.warning(f"Error during generation (attempt {i + 1}/{max_retries}): {error_message}")
            if "404" in error_message and "not found" in error_message:
                st.error(
                    f"Model '{model_name}' is not available or not supported.  Please select a different model."
                )
                return None # Return None to indicate failure.  The calling code must handle this.
            elif i < max_retries - 1:
                st.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise  # Re-raise the exception after the last retry
    raise Exception("Failed to generate content after maximum retries")



# --- Model Selection and Initialization ---
available_models = []
try:
    available_models = genai.list_models()  # Get the list of available models
except Exception as e:
    st.error(f"Error listing models: {e}.  Please check your API key and network connection.")
    st.stop()

model_names = [
    model.name for model in available_models if "generateContent" in model.supported_generation_methods
]  #get models supporting generateContent

if not model_names:
    st.error(
        "No models supporting 'generateContent' found.  This application requires a model that supports this method."
    )
    st.stop()

default_model = "gemini-pro" if "gemini-pro" in model_names else model_names[0] #select default model

selected_model = st.selectbox("Select a Model:", model_names, index=model_names.index(default_model))  # Let user choose

# Re-initialize the model with the selected name.  This is done *outside* the generate_with_retry loop.
try:
    model = genai.GenerativeModel(selected_model)
except Exception as e:
    st.error(f"Error initializing model {selected_model}: {e}")
    st.stop()

# --- Prompting Techniques Section ---
st.header("Experiment with Prompts")

prompt_technique = st.selectbox(
    "Choose a Prompting Technique:",
    [
        "Simple Instruction",
        "Using Delimiters",
        "Structured Output (JSON)",
        "Checking Assumptions",
        "Few-Shot Prompting",
        "Temperature Control",
        "Chain of Thought (CoT)",
        "Prompt Templates",
        "System Prompt",
        "Retrieval Augmentation"
    ],
    index=0,
)

prompt_input = st.text_area("Enter your prompt here:", height=150)

# Temperature slider
temperature = st.slider(
    "Temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="Controls the randomness of the output. Lower values are more deterministic; higher values are more creative.",
)

if st.button("Generate Response"):
    if not prompt_input:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            generation_config = genai.types.GenerationConfig(temperature=temperature)

            try:
                if prompt_technique == "Using Delimiters":
                    delimiter = st.text_input("Enter your delimiter (e.g., ###, ---):", "###")
                    processed_prompt = f"Here is the input, with parts separated by '{delimiter}':\n{prompt_input}\n Please process each part separately."
                    response = generate_with_retry(processed_prompt, selected_model, generation_config)
                    if response:
                        display_response(response)

                elif prompt_technique == "Structured Output (JSON)":
                    json_format = st.text_input(
                        "Describe the desired JSON format (e.g.,  {'name': str, 'age': int}):",
                        "{'key1': type, 'key2': type}",
                    )
                    processed_prompt = f"Please provide the output in JSON format, following this structure: {json_format}.  Here is the information: {prompt_input}"
                    response = generate_with_retry(processed_prompt, selected_model, generation_config)
                    if response:
                        try:
                            json_output = json.loads(response.text)
                            st.subheader("Generated JSON Output:")
                            st.json(json_output)
                        except json.JSONDecodeError:
                            st.error("Failed to decode JSON. Raw response:")
                            code_block(response.text, "json")

                elif prompt_technique == "Checking Assumptions":
                    assumption = st.text_input(
                        "State the assumption you want the model to check:", "The text is about a historical event."
                    )
                    processed_prompt = f"First, check if the following assumption is true: '{assumption}'.  Then, answer the prompt: {prompt_input}"
                    response = generate_with_retry(processed_prompt, selected_model, generation_config)
                    if response:
                        display_response(response)

                elif prompt_technique == "Few-Shot Prompting":
                    example1_input = st.text_area("Example 1 Input:", height=50)
                    example1_output = st.text_area("Example 1 Output:", height=50)
                    example2_input = st.text_area("Example 2 Input (Optional):", height=50)
                    example2_output = st.text_area("Example 2 Output (Optional):", height=50)

                    processed_prompt = "Here are some examples:\n"
                    processed_prompt += f"Input: {example1_input}\nOutput: {example1_output}\n"
                    if example2_input and example2_output:
                        processed_prompt += f"Input: {example2_input}\nOutput: {example2_output}\n"
                    processed_prompt += f"\nNow, answer the following:\nInput: {prompt_input}"

                    response = generate_with_retry(processed_prompt, selected_model, generation_config)
                    if response:
                        display_response(response)

                elif prompt_technique == "Temperature Control":
                    response = generate_with_retry(prompt_input, selected_model, generation_config)
                    if response:
                        display_response(response)

                elif prompt_technique == "Chain of Thought (CoT)":
                    cot_prompt = f"Let's think step by step. {prompt_input}"
                    response = generate_with_retry(cot_prompt, selected_model, generation_config)
                    if response:
                        display_response(response)

                elif prompt_technique == "Prompt Templates":
                    template_name = st.selectbox(
                        "Choose a template:",
                        ["None", "Question and Answer", "Summarization", "Code Generation"],
                        index=0,
                    )
                    if template_name == "Question and Answer":
                        processed_prompt = f"Answer the following question: {prompt_input}"
                    elif template_name == "Summarization":
                        processed_prompt = f"Summarize the following text: {prompt_input}"
                    elif template_name == "Code Generation":
                        language = st.text_input("Specify the programming language", "Python")
                        processed_prompt = f"Generate {language} code for the following: {prompt_input}"
                    else:
                        processed_prompt = prompt_input

                    response = generate_with_retry(processed_prompt, selected_model, generation_config)
                    if response:
                        display_response(response)

                elif prompt_technique == "System Prompt":
                    system_prompt_text = st.text_area(
                        "Enter system prompt:", "You are a helpful and informative assistant.", height=100
                    )
                    user_prompt = f"{prompt_input}"

                    response = generate_with_retry(
                        contents=[
                            genai.Content(role="system", parts=[genai.Part(text=system_prompt_text)]),
                            genai.Content(role="user", parts=[genai.Part(text=user_prompt)]),
                        ],
                        model=selected_model, # Pass the model name here as well
                        generation_config=generation_config,
                    )
                    if response:
                        display_response(response)

                elif prompt_technique == "Retrieval Augmentation":
                    context_text = st.text_area(
                        "Enter context text (knowledge base):",
                        "This is the context the model can use to answer the question.",
                        height=150,
                    )
                    processed_prompt = f"Given the following context: \n\n {context_text} \n\n Answer the following question: {prompt_input}"
                    response = generate_with_retry(processed_prompt, selected_model, generation_config)
                    if response:
                        display_response(response)

                else:  # Simple Instruction
                    response = generate_with_retry(prompt_input, selected_model, generation_config)
                    if response:
                        display_response(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")

