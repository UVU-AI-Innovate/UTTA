"""Simple chat interface for the teacher training simulator."""
from ai_agent import AIAgent

def main():
    print("=== Second Grade Teacher Training Simulator ===")
    print("\nWelcome to your personalized teaching practice session!")
    
    agent = AIAgent()
    agent.setup_teacher_profile()
    
    print("\nNow let's start practicing with some teaching scenarios.")
    print("Type 'quit' to end the session, 'new' for a new scenario, or 'help' for commands.")
    
    agent.start_interactive_session()

if __name__ == "__main__":
    main() 