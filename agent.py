import os
import re
from typing import List
from enum import Enum
from dotenv import load_dotenv

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
#from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from mem0 import MemoryClient

# Load environment variables
load_dotenv()

class ContextType(Enum):
    CV = "cv_text"
    JOB_DESCRIPTION = "job_description"
    INTERVIEWER_INFO = "interviewer_info"

class InterviewPrepAgent:
    """Agentic interview preparation assistant using function calling."""
    
    def __init__(self):
        self.setup_apis()
        self.setup_llm()
        self.context = {}  # Store parsed data
        self.tools = self.create_tools()
        self.agent = self.create_agent()
        
        # Restore context from memory on startup
        print("🔄 Checking for existing context in memory...")
        self.restore_all_context_from_memory()
        
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

        # # Initialize Mem0Memory for conversational memory (LlamaIndex integration)
        # context = {"user_id": "interview_prep_agent"}
        # self.memory = Mem0Memory.from_client(
        #     api_key=self.mem0_api_key,
        #     verbose=True,
        #     context=context
        # )
        
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
    
    def store_context_in_memory(self, content: str, context_type: ContextType):
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
    
    def restore_all_context_from_memory(self):
        """Restore all context items from memory on startup."""
        for context_type in ContextType:
            self.restore_context_from_memory(context_type)

    def restore_context_from_memory(self, context_type: ContextType):
        """Restore specific context item from memory."""
        try:
            # Search for stored context items
            all_memories = self.direct_memory.get_all(
                user_id=self.user_id,
            )
            # Helper function needed, as SDK filter doesnt work
            filtered_memories = self._filter_memories(all_memories, context_type)

            if len(filtered_memories) == 0:
                print(f"📝 No {context_type} found in memory")
                return
            elif len(filtered_memories) > 1:
                print(f"📝 Multiple {context_type} found in memory, using the most recent one")
                filtered_memories = filtered_memories[-1]
            else:
                filtered_memories = filtered_memories[0]
            
            self.context[context_type.value] = filtered_memories["memory"]
            print(f"🎉 Restored {context_type} from memory")
            
        except Exception as e:
            print(f"❌ Error restoring context from memory: {str(e)}")
    
    def parse_cv(self, cv_file_path: str) -> str:
        """Parse CV PDF using LlamaParse and anonymize the content."""
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
            self.context[ContextType.CV.value] = anonymized_cv_text
                
            # Store in memory for persistence
            self.store_context_in_memory(anonymized_cv_text, ContextType.CV)
            
            print("✅ CV parsed, anonymized, and stored in memory!")
            
            return f"CV successfully parsed, anonymized, and stored in memory.\n\nAnonymized CV Preview:\n{anonymized_cv_text[:500]}..."
            
        except Exception as e:
            return f"❌ Error parsing CV: {str(e)}"
    
    def load_job_description(self, job_desc_file_path: str) -> str:
        """Load job description from text file."""
        if not os.path.exists(job_desc_file_path):
            return f"❌ Error: Job description file not found: {job_desc_file_path}"
            
        try:
            with open(job_desc_file_path, 'r', encoding='utf-8') as f:
                job_description = f.read().strip()
                
            # Store in context
            self.context[ContextType.JOB_DESCRIPTION.value] = job_description
            
            # Store in memory for persistence
            self.store_context_in_memory(job_description, ContextType.JOB_DESCRIPTION)
            
            return f"✅ Job description loaded and stored in memory!\n\nJob Description Preview:\n{job_description[:500]}..."
            
        except Exception as e:
            return f"❌ Error reading job description file: {str(e)}"
    
    def load_interviewer_info(self, interviewer_file_path: str) -> str:
        """Load interviewer information from text file."""
        if not os.path.exists(interviewer_file_path):
            return f"❌ Error: Interviewer info file not found: {interviewer_file_path}"
            
        try:
            with open(interviewer_file_path, 'r', encoding='utf-8') as f:
                interviewer_info = f.read().strip()
                
            # Store in context
            self.context[ContextType.INTERVIEWER_INFO.value] = interviewer_info
  
            
            # Store in memory for persistence
            self.store_context_in_memory(interviewer_info, ContextType.INTERVIEWER_INFO)
            
            return f"✅ Interviewer information loaded and stored in memory!\n\nInterviewer Info Preview:\n{interviewer_info[:500]}..."
            
        except Exception as e:
            return f"❌ Error reading interviewer info file: {str(e)}"
    
    def generate_question(self, question_type: str = "mixed") -> str:
        """Generate an interview question based on available context."""
        
        # Check what context we have using centralized keys
        missing_context = []
        for context_type in ContextType:
            if context_type.value not in self.context:
                missing_context.append(context_type.name.replace('_', ' ').title())
            
        if missing_context:
            return f"❌ Cannot generate question. Missing context: {', '.join(missing_context)}. Please load the missing files first."
        
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
            {self.context[ContextType.CV.value]}
            
            JOB DESCRIPTION:
            {self.context[ContextType.JOB_DESCRIPTION.value]}
            
            INTERVIEWER INFORMATION:
            {self.context[ContextType.INTERVIEWER_INFO.value]}
            
            TASK:
            {question_prompts.get(question_type, question_prompts["mixed"])}
            """
            
            # Generate question using direct LLM call
            response = self.llm.complete(full_prompt)
            question = str(response).strip()
            
            # Store question in context
            self.context["current_question"] = question
            self.context["question_type"] = question_type
            
            return f"🎯 INTERVIEW QUESTION ({question_type.upper()}):\n\n{question}\n\n💭 Please provide your answer, and I'll give you feedback!"
            
        except Exception as e:
            return f"❌ Error generating question: {str(e)}"
    
    def provide_feedback(self, answer: str) -> str:
        """Provide feedback on the user's answer to the current question."""
        
        if "current_question" not in self.context:
            return "❌ No current question to provide feedback on. Please generate a question first."
        
        try:
            print("\n🔄 Analyzing your answer...")
            
            # Create comprehensive prompt with all context
            full_prompt = f"""
            You are an expert interviewer providing constructive feedback on interview answers.
            
            INTERVIEW QUESTION: {self.context['current_question']}
            
            CANDIDATE'S ANSWER: {answer}
            
            CANDIDATE'S CV (Anonymized):
            {self.context.get(ContextType.CV.value, 'Not available')}
            
            JOB DESCRIPTION:
            {self.context.get(ContextType.JOB_DESCRIPTION.value, 'Not available')}
            
            INTERVIEWER INFORMATION:
            {self.context.get(ContextType.INTERVIEWER_INFO.value, 'Not available')}
            
            TASK:
            Please provide constructive feedback on this interview answer. Consider:
            1. How well the answer addresses the question
            2. Technical accuracy (if applicable)
            3. Communication clarity
            4. Areas for improvement
            5. What the answer demonstrates about the candidate's fit for the role
            6. Specific suggestions for improvement
            
            Provide specific, actionable feedback that would help the candidate improve.
            """
            
            # Generate feedback using direct LLM call
            response = self.llm.complete(full_prompt)
            feedback = str(response).strip()
            
            # Store the answer and feedback
            self.context["last_answer"] = answer
            self.context["last_feedback"] = feedback
            
            return f"📝 FEEDBACK:\n\n{feedback}\n\n🎉 Feedback complete! You can ask for another question or more specific feedback."
            
        except Exception as e:
            return f"❌ Error providing feedback: {str(e)}"
    
    def get_context_status(self) -> str:
        """Get current status of loaded context."""
        status = "📊 CURRENT CONTEXT STATUS:\n\n"
        
        # Check each context item using centralized keys
        for context_type in ContextType:
            if context_type.value in self.context:
                # Get preview of content (first 100 chars)
                content_preview = str(self.context[context_type.value])[:100].replace('\n', ' ')
                status += f"✅ {context_type.name.replace('_', ' ').title()}: Loaded\n"
                status += f"   Preview: {content_preview}...\n\n"
            else:
                status += f"❌ {context_type.name.replace('_', ' ').title()}: Not loaded\n\n"
        
        # Check current question status
        if "current_question" in self.context:
            question_type = self.context.get("question_type", "unknown")
            question_preview = self.context["current_question"][:100].replace('\n', ' ')
            status += f"✅ Current Question ({question_type}): {question_preview}...\n"
        else:
            status += "❌ Current Question: None\n"
            
        return status
    
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
            FunctionTool.from_defaults(
                fn=self.provide_feedback,
                name="provide_feedback",
                description="Provide feedback on the user's answer to the current interview question. Requires the user's answer as input."
            ),
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
    
    def create_agent(self) -> FunctionCallingAgent:
        """Create the function calling agent."""
        system_prompt = """You are an intelligent interview preparation assistant. You help users prepare for job interviews by:

1. Parsing their CV and anonymizing it
2. Loading job descriptions and interviewer information
3. Generating relevant interview questions
4. Providing constructive feedback on answers

You have access to tools that can:
- parse_cv: Parse and anonymize CV PDF files
- load_job_description: Load job description from text files
- load_interviewer_info: Load interviewer information from text files
- generate_question: Generate interview questions. Ask the user what type of question they want to be asked.
- provide_feedback: Analyze and provide feedback on interview answers
- get_context_status: Check what information is currently loaded
- clear_context_memory: Clear all context items from memory and local storage

IMPORTANT GUIDELINES:
- Always check the current context status before generating questions
- Be encouraging and constructive in your feedback
- Ask for missing information when needed
- Suggest next steps based on the current state
- Be conversational and helpful

When a user first interacts with you, explain what you can do and guide them through the process. If they don't have all the required files, help them understand what they need."""

        agent = FunctionCallingAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            system_prompt=system_prompt,
            # memory=self.memory,
            verbose=True
        )
        return agent
    
    async def chat(self):
        """Start the interactive chat session."""
        print("🎯 Welcome to the Interview Preparation Assistant!")
        print("I can help you prepare for job interviews using your CV, job description, and interviewer information.")
        print("Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye! Good luck with your interview preparation!")
                    break
                
                if not user_input:
                    continue
                
                # Get response from agent
                response = await self.agent.achat(user_input)
                print(f"\n🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Good luck with your interview preparation!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")