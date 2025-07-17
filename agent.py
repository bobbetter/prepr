import os
import re
from typing import List
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from mem0 import MemoryClient
from llama_index.core.workflow import (
    InputRequiredEvent,
    HumanResponseEvent,
    Context,
)

# Load environment variables
load_dotenv()

class ContextType(Enum):
    CV = "cv_text"
    JOB_DESCRIPTION = "job_description"
    INTERVIEWER_INFO = "interviewer_info"

class PreprContext(BaseModel):
    cv_loaded: bool = False
    job_loaded: bool = False
    interviewer_loaded: bool = False
    cv_text: str = None
    job_description: str = None
    interviewer_info: str = None

class InterviewPrepAgent:
    """Agentic interview preparation assistant using function calling."""
    
    def __init__(self, ctx: Context[PreprContext]):
        self.setup_apis()
        self.setup_llm()
        self.context = {}  # Store parsed data
        self.tools = self.create_tools()
        self.agent = self.create_agent()
        
        # Restore context from memory on startup
        print("🔄 Checking for existing context in memory...")  
        self._restore_all_context_from_memory(ctx)
        
    def setup_apis(self):
        """Setup API keys and clients."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")
        self.mem0_api_key = os.getenv("MEM0_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not self.llama_parse_api_key:
            raise ValueError("LLAMA_PARSE_API_KEY not found in environment variables")
        if not self.mem0_api_key:
            raise ValueError("MEM0_API_KEY not found in environment variables")
            
        # Initialize LlamaParse
        self.parser = LlamaParse(
            api_key=self.llama_parse_api_key,
            result_type="text",
            verbose=True
        )
        
        # Initialize direct Mem0 Memory client for explicit context storage
        self.direct_memory = MemoryClient(
            api_key=self.mem0_api_key
        )
        
        # User ID for memory storage
        self.user_id = "interview_prep_agent"
        
    def setup_llm(self):
        """Setup LlamaIndex LLM configuration."""
        self.llm = OpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=self.openai_api_key
        )
    
    def _store_context_in_memory(self, content: str, context_type: ContextType):
        """Store specific context items in memory with metadata."""
        try:
            messages = [
                {"role": "user", "content": content}
            ]
            self.direct_memory.add(
                messages,
                user_id=self.user_id,
                metadata={"context_type": context_type.value},
                infer=False
            )
            print(f"✅ Stored {context_type} in memory")
        except Exception as e:
            print(f"❌ Error storing {context_type} in memory: {str(e)}")
    
    def _filter_memories(self, memories, context_type: ContextType):
        return [x for x in memories if x["metadata"] == {'context_type': context_type.value}]
    
    async def _restore_all_context_from_memory(self, ctx: Context[PreprContext]):
        """Restore all context items from memory on startup."""
        for context_type in ContextType:
            await self._restore_context_from_memory(ctx, context_type)

    async def _restore_context_from_memory(self, ctx: Context[PreprContext], context_type: ContextType):
        """Restore specific context item from memory."""
        try:
            # Search for stored context items
            all_memories = self.direct_memory.get_all(
                user_id=self.user_id,
            )
            # Helper function needed, as SDK filter doesnt work
            memory = self._filter_memories(all_memories, context_type)
            state = await ctx.store.get_state()

            if memory is None:
                return
            match context_type:

                case ContextType.CV:
                    state.cv_text = memory
                    state.cv_loaded = True
                case ContextType.JOB_DESCRIPTION:
                    state.job_description = memory
                    state.job_loaded = True
                case ContextType.INTERVIEWER_INFO:
                    state.interviewer_info = memory
                    state.interviewer_loaded = True
            print(f"🎉 Restored {context_type} from memory")
            await ctx.store.set_state(state)
        except Exception as e:
            print(f"❌ Error restoring context from memory: {str(e)}")
    
    async def parse_cv(self, ctx: Context[PreprContext]) -> None:
        """Parse CV PDF using LlamaParse and anonymize the content."""
        question = "What is the path to the CV file?"
        response = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_id=question,
            waiter_event=InputRequiredEvent(
                prefix=question,
            ),
        )
        cv_file_path = response.response
        
        if not os.path.exists(cv_file_path):
            return f"❌ Error: CV file not found: {cv_file_path}"
            
        try:
            print(f"🔄 Parsing CV: {cv_file_path}")
            
            # Parse the PDF
            documents = self.parser.load_data(cv_file_path)
            
            # Extract text from parsed documents
            cv_text = ""
            for doc in documents:
                cv_text += doc.text + "\n"
                
            # Anonymize the text
            anonymized_cv_text = self.anonymize_text(cv_text.strip())
            
            # Store in context
            state = await ctx.store.get_state()
            state.cv_text = anonymized_cv_text
            state.cv_loaded = True
            await ctx.store.set_state(state)
                
            # Store in memory for persistence
            self._store_context_in_memory(anonymized_cv_text, ContextType.CV)
            
            print("✅ CV parsed, anonymized, and stored in memory!")
            
        except Exception as e:
            print(f"❌ Error parsing CV: {str(e)}")
    
    async def load_job_description(self, ctx: Context[PreprContext]) -> None:
        """Load job description from text file."""
        question = "What is the path to the job description file?"
        response = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_id=question,
            waiter_event=InputRequiredEvent(
                prefix=question,
            ),
        )
        job_desc_file_path = response.response
        if not os.path.exists(job_desc_file_path):
            print(f"❌ Error: Job description file not found: {job_desc_file_path}")
            return None
            
        try:
            with open(job_desc_file_path, 'r', encoding='utf-8') as f:
                job_description = f.read().strip()
                
            # Store in context
            state = await ctx.store.get_state()
            state.job_description = job_description
            state.job_loaded = True
            await ctx.store.set_state(state)
            
            # Store in memory for persistence
            self._store_context_in_memory(job_description, ContextType.JOB_DESCRIPTION)
            print("✅ Job description loaded and stored in memory!")
            
        except Exception as e:
            print(f"❌ Error reading job description file: {str(e)}")
    
    async def load_interviewer_info(self, ctx: Context[PreprContext]) -> None:
        """Load interviewer information from text file."""
        question = "What is the path to the interviewer info file?"
        response = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_id=question,
            waiter_event=InputRequiredEvent(
                prefix=question,
            ),
        )
        interviewer_file_path = response.response
        if not os.path.exists(interviewer_file_path):
            print(f"❌ Error: Interviewer info file not found: {interviewer_file_path}")
            return None
            
        try:
            with open(interviewer_file_path, 'r', encoding='utf-8') as f:
                interviewer_info = f.read().strip()
                
            # Store in context
            state = await ctx.store.get_state()
            state.interviewer_info = interviewer_info
            state.interviewer_loaded = True
            await ctx.store.set_state(state)
            
            # Store in memory for persistence
            self._store_context_in_memory(interviewer_info, ContextType.INTERVIEWER_INFO)
            print("✅ Interviewer information loaded and stored in memory!")
            
        except Exception as e:
            print(f"❌ Error reading interviewer info file: {str(e)}")
    
    async def generate_question(self, ctx: Context[PreprContext]) -> str:
        """Generate an interview question based on available context."""
        print("🤔 Generating interview question...")
        # Check what context we have using centralized keys
        missing_context = []
        state = await ctx.store.get_state()
        for context_type in ContextType:
            if not getattr(state, context_type.value):
                missing_context.append(context_type.name.replace('_', ' ').title())
            
        if missing_context:
            print(f"❌ Cannot generate question. Missing context: {', '.join(missing_context)}. Please load the missing files first.")
            return None
        

        question = "What type of question do you want to be asked?"
        response = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_id=question,
            waiter_event=InputRequiredEvent(
                prefix=question,
            ),
        )
        question_type = response.response
        try:
            print("\n🤔 Generating interview question...")
            
            # Generate question based on type
            question_prompts = {
                "technical": """
                Generate ONE challenging technical interview question that:
                1. Tests specific technical skills mentioned in the job description
                2. Relates to the candidate's technical experience
                3. Would be appropriate for the interviewer's technical background
                4. Requires problem-solving or system design thinking
                
                Return ONLY the question, no additional explanation.
                """,
                "behavioral": """
                Generate ONE behavioral interview question that:
                1. Tests soft skills and experience mentioned in the job description
                2. Relates to the candidate's background and experience
                3. Would help assess cultural fit
                4. Uses the STAR method format
                
                Return ONLY the question, no additional explanation.
                """,
                "mixed": """
                Generate ONE thoughtful interview question that:
                1. Tests relevant technical or behavioral skills mentioned in the job description
                2. Relates to the candidate's experience
                3. Would be appropriate for the interviewer's style/background
                4. Is challenging but fair
                
                Return ONLY the question, no additional explanation.,
                """,
                "open": """
                Generate ONE open-ended interview question that:
                1. Tests the candidate's ability to demonstrate their product sense and thinking about business impact
                2. Relates to the candidate's experience
                3. Would be appropriate for the interviewer's style/background
                4. Is challenging but fair
                
                Return ONLY the question, no additional explanation.
                """
            }
            
            # Create comprehensive prompt with all context
            full_prompt = f"""
            You are an expert interviewer helping to create personalized interview questions.
            
            CANDIDATE'S CV (Anonymized):
            {state.cv_text}
            
            JOB DESCRIPTION:
            {state.job_description}
            
            INTERVIEWER INFORMATION:
            {state.interviewer_info}
            
            TASK:
            {question_prompts.get(question_type, question_prompts["mixed"])}
            """
            
            # Generate question using direct LLM call
            response = self.llm.complete(full_prompt)
            question = str(response).strip()
                            
            return f"🎯 INTERVIEW QUESTION ({question_type.upper()}):\n\n{question}\n\n💭 Please provide your answer, and I'll give you feedback!"
            
        except Exception as e:
            return f"❌ Error generating question: {str(e)}"
    
    async def get_context_status(self, ctx: Context[PreprContext]) -> str:
        """Get current status of loaded context."""
        context_status = "📊 CURRENT CONTEXT STATUS:\n\n"
        state = await ctx.store.get_state()
        # Check each context item using centralized keys

        if state.cv_loaded:
            context_status += "✅ CV: Loaded\n"
        else:
            context_status += "❌ CV: Not loaded\n"

        if state.job_loaded:
            context_status += "✅ Job description: Loaded\n"
        else:   
            context_status += "❌ Job description: Not loaded\n"

        if state.interviewer_loaded:
            context_status += "✅ Interviewer info: Loaded\n"
        else:
            context_status += "❌ Interviewer info: Not loaded\n"
            
        return context_status
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize company names and personal information in the text."""
        anonymized_text = text
        
        # Common patterns for names (basic anonymization)
        name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last name pattern
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',  # First M. Last pattern
        ]
        
        for pattern in name_patterns:
            anonymized_text = re.sub(pattern, '[NAME]', anonymized_text)
        
        # Common company patterns and known tech companies
        company_patterns = [
            r'\bGoogle\b', r'\bMicrosoft\b', r'\bApple\b', r'\bAmazon\b', r'\bMeta\b',
            r'\bFacebook\b', r'\bNetflix\b', r'\bTesla\b', r'\bUber\b', r'\bAirbnb\b',
            r'\bStripe\b', r'\bSpotify\b', r'\bSlack\b', r'\bZoom\b', r'\bDropbox\b',
            r'\bSalesforce\b', r'\bOracle\b', r'\bIBM\b', r'\bIntel\b', r'\bNVIDIA\b',
            r'\bAdobe\b', r'\bTwitter\b', r'\bLinkedIn\b', r'\bSquare\b', r'\bPayPal\b', 
            r'\bSiriusXM\b', r'\bPandora\b'
        ]
        
        for pattern in company_patterns:
            anonymized_text = re.sub(pattern, '[COMPANY]', anonymized_text, flags=re.IGNORECASE)
        
        # Generic company patterns
        company_suffixes = [
            r'\b\w+\s+Inc\.?\b', r'\b\w+\s+LLC\.?\b', r'\b\w+\s+Corp\.?\b',
            r'\b\w+\s+Corporation\b', r'\b\w+\s+Limited\b', r'\b\w+\s+Ltd\.?\b',
            r'\b\w+\s+Company\b', r'\b\w+\s+Co\.?\b'
        ]
        
        for pattern in company_suffixes:
            anonymized_text = re.sub(pattern, '[COMPANY]', anonymized_text, flags=re.IGNORECASE)
        
        # Email addresses
        anonymized_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', anonymized_text)
        
        # Phone numbers
        phone_patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
            r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (123) 456-7890
            r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
            r'\b\+\d{1,3}\s*\d{3}\s*\d{3}\s*\d{4}\b'  # +1 123 456 7890
        ]
        
        for pattern in phone_patterns:
            anonymized_text = re.sub(pattern, '[PHONE]', anonymized_text)
        
        return anonymized_text
    
    def create_tools(self) -> List[FunctionTool]:
        """Create tools for the agent."""
        tools = [
            FunctionTool.from_defaults(
                fn=self.parse_cv,
                name="parse_cv",
                description="Parse and anonymize a CV PDF file. Requires the full file path to the CV PDF."
            ),
            FunctionTool.from_defaults(
                fn=self.load_job_description,
                name="load_job_description",
                description="Load job description from a text file. Requires the full file path to the job description text file."
            ),
            FunctionTool.from_defaults(
                fn=self.load_interviewer_info,
                name="load_interviewer_info",
                description="Load interviewer information from a text file. Requires the full file path to the interviewer info text file."
            ),
            FunctionTool.from_defaults(
                fn=self.generate_question,
                name="generate_question",
                description="Generate an interview question based on loaded context. Ask the user what type of question they want to be asked: 'technical', 'behavioral', 'mixed', or 'open'."
            ),
            # FunctionTool.from_defaults(
            #     fn=self.provide_feedback,
            #     name="provide_feedback",
            #     description="Provide feedback on the user's answer to the current interview question. Requires the user's answer as input."
            # ),
            FunctionTool.from_defaults(
                fn=self.get_context_status,
                name="get_context_status",
                description="Get the current status of loaded context (CV, job description, interviewer info, current question)."
            ),
            # FunctionTool.from_defaults(
            #     fn=self.clear_context_memory,
            #     name="clear_context_memory",
            #     description="Clear all context items from memory and local storage."
            # )
        ]
        return tools
    
    def create_agent(self) -> FunctionAgent:
        """Create the function calling agent."""
        system_prompt = """You are an intelligent interview preparation assistant. You help users prepare for job interviews by:

1. Parsing their CV and anonymizing it
2. Loading job descriptions and interviewer information
3. Generating relevant interview questions


So, you have to load the context first, then you can generate questions.
To be clear, the full context is:
- CV
- Job description
- Interviewer information
You can use the tool 'get_context_status' to check what context is currently loaded.

Only if you have all that context, you can generate questions. Otherwise, you will need to load that context first, 
by asking the user for the path to the file and using the tools to load it.

You have access to tools that can:
- parse_cv: Parse and anonymize CV PDF files
- load_job_description: Load job description from text files
- load_interviewer_info: Load interviewer information from text files
- generate_question: Generate interview questions. Ask the user what type of question they want to be asked.
- get_context_status: Check what information is currently loaded


IMPORTANT GUIDELINES:
- Always check the if you have all the context (CV, job description, interviewer info) before generating questions
- Ask for missing information when needed
- Suggest next steps based on the current state
- Be conversational and helpful

When a user first interacts with you, explain what you can do and guide them through the process. If they don't have all the required files, help them understand what they need."""

        return FunctionAgent(
            tools=self.tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True
        )
    
