from transformers import pipeline
import sys

# Load the simplifier model once at startup
simplifier = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

# Simplification Wrapper - Draft 2
# Author: [Your Name]
# Date: [Today's Date]
# Description: Simplifies complicated legal questions using an open-source LLM.

def simplify_legal_question(question):
    """
    Simplifies a complicated legal question using Hugging Face's transformers library.
    """
    try:
        response = simplifier(f"paraphrase: {question}", max_length=60, num_return_sequences=1)
        simplified_question = response[0]['generated_text']
        return simplified_question
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    """
    Main function to execute the script.
    """
    print("Welcome to the Simplification Wrapper!")
    print("Enter a complicated legal question, and I'll simplify it for you.")
    
    while True:
        question = input("\nEnter your legal question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        
        simplified = simplify_legal_question(question)
        print("\nSimplified Question:")
        print(simplified)

if __name__ == "__main__":
    print("Python executable in use:", sys.executable)
    main()
