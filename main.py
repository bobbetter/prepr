#!/usr/bin/env python3
"""
Interview Preparation Agentic Tool
Uses FunctionCallingAgent with tools for flexible, intelligent interview preparation.
"""

from agent import InterviewPrepAgent, PreprContext
import asyncio
from llama_index.core.workflow import (
    InputRequiredEvent,
    HumanResponseEvent,
    Context
)


# async def main():
#     """Main function to run the interview preparation agent."""
#     try:
#         agent = InterviewPrepAgent()
#         await agent.chat()
#     except Exception as e:
#         print(f"❌ Error initializing agent: {str(e)}")
#         return 1




async def main():
    print("🚀 Starting Interview Prep Workflow...")
    print("=" * 50)
    ctx = PreprContext()
    prep_agent = InterviewPrepAgent(ctx)
    handler = prep_agent.agent.run("Lets start interview prep!")

    try:
        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                response = input(event.prefix)
                handler.ctx.send_event(
                    HumanResponseEvent(response=response)
                )


                        # Get the final result
        result = await handler
        print(f"\n🎉 Workflow completed successfully!")
        
        # Get the final state to show what was loaded
        final_state = await handler.ctx.store.get_state()
        return final_state
    
    except Exception as e:
        print(f"❌ Error running workflow: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())
