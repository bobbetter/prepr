#!/usr/bin/env python3
"""
Interview Preparation Agentic Tool
Uses FunctionCallingAgent with tools for flexible, intelligent interview preparation.
"""

import warnings
# Suppress all deprecation warnings to avoid LlamaIndex noise
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Alternative specific suppressions in case the above doesn't catch everything
warnings.simplefilter("ignore", DeprecationWarning)

from agent import InterviewPrepAgent, PreprContext
import asyncio
from llama_index.core.workflow import (
    InputRequiredEvent,
    HumanResponseEvent,
    Context
)

async def interactive_chat():
    """Run an interactive chat session with the agent."""
    print("🚀 Starting Interview Prep Agent...")
    print("=" * 50)
    
    # Create agent instance
    prep_agent = InterviewPrepAgent()
    
    # After agent initialization, re-apply warning filters
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.simplefilter("ignore", DeprecationWarning)

    print("\n🤖 Agent: Hello! I'm your interview preparation assistant.")
    print("Let me check what context information we have...")
    
    # Check initial status
    status = prep_agent.get_context_status()
    print(f"\n{status}")
    
    print("\nI can help you with:")
    print("- Loading your CV (PDF file)")
    print("- Loading job description (text file)")
    print("- Loading interviewer information (text file)")
    print("- Generating interview questions")
    print("\nJust tell me what you'd like to do in natural language!")
    print("(Type 'quit' to exit)")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("🤖 Agent: Goodbye! Good luck with your interview preparation!")
            break
            
        # Let the agent handle everything through natural language
        try:
            handler = prep_agent.agent.run(user_input)
            async for event in handler.stream_events():
                if isinstance(event, InputRequiredEvent):
                    response = input(f"{event.prefix}: ")
                    handler.ctx.send_event(
                        HumanResponseEvent(response=response)
                    )
            
            result = await handler
            print(f"\n🤖 Agent: {result}")
            
        except Exception as e:
            print(f"\n🤖 Agent: I encountered an error: {str(e)}")
            print("Please try again or rephrase your request.")

async def main():
    """Main function to run the interactive agent."""
    try:
        await interactive_chat()
    except KeyboardInterrupt:
        print("\n\n🤖 Agent: Session interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
