from transformers import pipeline

# Simplification Wrapper - Draft 2
# Author: [Your Name]
# Date: [Today's Date]
# Description: Simplifies complicated legal questions using an open-source LLM.

def simplify_legal_question(question):
    """
    Simplifies a complicated legal question using Hugging Face's transformers library.
    """
    try:
        # Load a summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        response = summarizer(question, max_length=50, min_length=25, do_sample=False)
        simplified_question = response[0]['summary_text']
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
    main()
