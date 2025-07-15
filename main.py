#!/usr/bin/env python3
"""
Interview Preparation Agentic Tool
Uses FunctionCallingAgent with tools for flexible, intelligent interview preparation.
"""

from agent import InterviewPrepAgent
import asyncio


async def main():
    """Main function to run the interview preparation agent."""
    try:
        agent = InterviewPrepAgent()
        await agent.chat()
    except Exception as e:
        print(f"❌ Error initializing agent: {str(e)}")
        return 1

if __name__ == "__main__":
    asyncio.run(main())
