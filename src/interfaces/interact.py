"""Interactive interface for teacher training simulation."""
from ai_agent import AIAgent

def main():
    print("Welcome to the Second Grade Teacher Training Simulator!")
    print("\nThis simulator will help you practice handling various classroom scenarios.")
    print("You'll interact with a simulated second-grade student with realistic behaviors and responses.")
    
    # Create AI agent
    agent = AIAgent()
    
    while True:
        print("\nAvailable Scenario Categories:")
        print("1. Attention Issues")
        print("2. Participation Challenges")
        print("3. Random Scenario")
        print("4. Quit")
        
        choice = input("\nChoose a category (1-4): ").strip()
        
        if choice == "1":
            agent.start_interactive_session("attention")
        elif choice == "2":
            agent.start_interactive_session("participation")
        elif choice == "3":
            agent.start_interactive_session()
        elif choice == "4":
            print("\nThank you for using the Teacher Training Simulator!")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main() 